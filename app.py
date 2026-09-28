from __future__ import annotations

import json
import math
import shutil
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QStandardPaths, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None


SOUND_GROUPS = ["А/Я", "О/Ё", "У/Ю", "И/Ы", "Е/Э", "Тишина"]
CALIBRATION_VOWELS = ["А", "О", "У", "Ы", "Э"]
VOWELS = CALIBRATION_VOWELS
GROUP_MEMBERS = {
    "А/Я": ("А", "Я"),
    "О/Ё": ("О", "Ё"),
    "У/Ю": ("У", "Ю"),
    "И/Ы": ("И", "Ы"),
    "Е/Э": ("Е", "Э"),
    "Тишина": (),
}
VOWEL_TO_GROUP = {vowel: group for group, vowels in GROUP_MEMBERS.items() for vowel in vowels}
SAMPLE_RATE = 16_000
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DEFAULT_FORMANTS = {
    "А": (800, 1500, 2600),
    "О": (500, 1000, 2500),
    "У": (350, 850, 2200),
    "Ы": (450, 1400, 2500),
    "Э": (600, 1750, 2600),
}


def extract_mfcc_features(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray | None:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size < int(sample_rate * 0.12):
        return None
    samples = samples - np.mean(samples)
    peak = float(np.max(np.abs(samples)))
    if peak < 1e-5:
        return None
    samples /= peak

    frame_size = int(sample_rate * 0.025)
    hop_size = int(sample_rate * 0.010)
    if samples.size < frame_size:
        samples = np.pad(samples, (0, frame_size - samples.size))
    frames = np.stack(
        [samples[start : start + frame_size] for start in range(0, samples.size - frame_size + 1, hop_size)]
    )
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    voiced = frames[rms > max(0.025, float(np.max(rms)) * 0.12)]
    if voiced.shape[0] < 5:
        return None

    windowed = voiced * np.hamming(frame_size).astype(np.float32)
    power = np.abs(np.fft.rfft(windowed, n=512, axis=1)) ** 2
    mel_low = 2595.0 * math.log10(1.0 + 80.0 / 700.0)
    mel_high = 2595.0 * math.log10(1.0 + (sample_rate / 2) / 700.0)
    mel_points = np.linspace(mel_low, mel_high, 28)
    hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((513 * hz_points) / sample_rate).astype(int)
    filters = np.zeros((26, 257), dtype=np.float32)
    for index in range(26):
        left, center, right = bins[index : index + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for bin_index in range(left, min(center, 257)):
            filters[index, bin_index] = (bin_index - left) / (center - left)
        for bin_index in range(center, min(right, 257)):
            filters[index, bin_index] = (right - bin_index) / (right - center)
    mel_energy = np.maximum(power @ filters.T, 1e-10)
    log_mel = np.log(mel_energy)
    mel_indices = np.arange(26, dtype=np.float32)
    coefficient_indices = np.arange(13, dtype=np.float32)[:, None]
    dct = np.cos((np.pi / 26) * (mel_indices + 0.5) * coefficient_indices)
    cepstra = log_mel @ dct.T
    sections = np.array_split(cepstra, 3)
    return np.concatenate(
        [np.mean(cepstra, axis=0), np.std(cepstra, axis=0)]
        + [np.mean(section, axis=0) for section in sections]
    )


def make_default_prototypes() -> dict[str, list[list[float]]]:
    prototypes = {}
    sample_count = int(SAMPLE_RATE * 0.45)
    time_axis = np.arange(sample_count, dtype=np.float32) / SAMPLE_RATE
    frequency_axis = np.fft.rfftfreq(sample_count, 1 / SAMPLE_RATE)

    for vowel, formants in DEFAULT_FORMANTS.items():
        variants = []
        for pitch, vocal_scale in zip((105, 145, 185), (0.94, 1.0, 1.06)):
            source = np.zeros(sample_count, dtype=np.float32)
            for harmonic in range(1, SAMPLE_RATE // (2 * pitch)):
                source += np.sin(2 * np.pi * pitch * harmonic * time_axis) / harmonic
            envelope = np.full_like(frequency_axis, 0.01)
            for index, formant in enumerate(formants):
                center = formant * vocal_scale
                width = (100, 150, 220)[index]
                envelope += (1.0 / (index + 1)) * np.exp(-0.5 * ((frequency_axis - center) / width) ** 2)
            synthesized = np.fft.irfft(np.fft.rfft(source) * envelope, n=sample_count).astype(np.float32)
            fade_length = int(SAMPLE_RATE * 0.02)
            synthesized[:fade_length] *= np.linspace(0, 1, fade_length)
            synthesized[-fade_length:] *= np.linspace(1, 0, fade_length)
            features = extract_mfcc_features(synthesized)
            if features is not None:
                variants.append(features)
        prototypes[vowel] = [np.mean(variants, axis=0).tolist()]
    return prototypes


class AudioInput:
    def __init__(self) -> None:
        self.stream = None
        self.lock = threading.Lock()
        self.chunks: deque[np.ndarray] = deque()
        self.waveform_levels: deque[float] = deque([0.0] * 48, maxlen=48)
        self.max_samples = SAMPLE_RATE // 2
        self.level = 0.0
        self.input_sample_rate = SAMPLE_RATE

    def start(self, device: int | None) -> None:
        self.stop()
        with self.lock:
            self.chunks.clear()
            self.waveform_levels = deque([0.0] * 48, maxlen=48)
        self.input_sample_rate = int(sd.query_devices(device, "input")["default_samplerate"])
        self.stream = sd.InputStream(
            device=device,
            samplerate=self.input_sample_rate,
            channels=1,
            dtype="float32",
            blocksize=256,
            callback=self._on_audio,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _on_audio(self, indata, _frames, _time, status) -> None:
        block = indata[:, 0].copy()
        level = float(np.sqrt(np.mean(block * block)))
        if self.input_sample_rate != SAMPLE_RATE:
            output_size = max(1, round(block.size * SAMPLE_RATE / self.input_sample_rate))
            block = np.interp(
                np.linspace(0, block.size - 1, output_size),
                np.arange(block.size),
                block,
            ).astype(np.float32)
        with self.lock:
            self.chunks.append(block)
            while sum(chunk.size for chunk in self.chunks) > self.max_samples:
                self.chunks.popleft()
            self.level = level
            self.waveform_levels.append(level)

    def snapshot(self, duration: float = 0.9) -> np.ndarray:
        with self.lock:
            if not self.chunks:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(tuple(self.chunks))
        return audio[-int(SAMPLE_RATE * duration) :]

    def waveform_snapshot(self) -> list[float]:
        with self.lock:
            return list(self.waveform_levels)


class WaveformDisplay(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(150)
        self.setFixedHeight(40)
        self.levels = [0.0] * 48

    def set_levels(self, levels: list[float]) -> None:
        self.levels = levels
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        center_y = self.height() // 2
        count = len(self.levels)
        step = self.width() / max(count, 1)
        for index, level in enumerate(self.levels):
            strength = min(1.0, level * 260.0)
            bar_height = max(2, int(strength * (self.height() - 6)))
            color = QColor("#00e5c0")
            color.setAlpha(90 + int(strength * 165))
            painter.fillRect(
                int(index * step + step * 0.3),
                center_y - bar_height // 2,
                max(1, int(step * 0.48)),
                bar_height,
                color,
            )
        painter.end()


def make_panel() -> tuple[QFrame, QVBoxLayout]:
    panel = QFrame()
    panel.setObjectName("panel")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(10)
    return panel, layout


class VoicecamWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("voicecam")
        self.resize(1120, 760)
        self.app_dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
        self.asset_dir = self.app_dir / "images"
        self.settings_file = self.app_dir / "settings.json"
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.settings = self._load_settings()
        raw_training = self.settings.get("training", {})
        self.user_training: dict[str, list[list[float]]] = {
            vowel: raw_training.get(vowel, []) for vowel in CALIBRATION_VOWELS
        }
        self.training = make_default_prototypes()
        self.training.update({vowel: samples for vowel, samples in self.user_training.items() if samples})
        self.images = self._migrate_images(self.settings.get("images", {}))
        self.audio = AudioInput()
        self.virtual_camera = None
        self.current_group = "Тишина"
        self.gallery_group = "А/Я"
        self.target_index = 0
        self.samples_for_target = 0
        self.calibration_complete = False
        self.capture_pending = False
        self.testing = False
        self.last_prediction = ""
        self.prediction_streak = 0
        self.last_emitted = ""
        self.last_emitted_at = 0.0
        self.current_image_path = ""
        self.output_pixmap = QPixmap()
        self.virtual_frame_buffer = None
        self.image_label = None
        self.output_status = None
        self.header_status = None
        self.output_button = None
        self.waveform = None
        self.level_bar = None
        self.threshold_slider = None
        self.calibration_letter = None
        self.calibration_group = None
        self.calibration_progress = None
        self.capture_button = None
        self.camera_view = None
        self.gallery = None
        self.letter_buttons: dict[str, QPushButton] = {}

        self._build_ui()
        next_target = next(
            (index for index, vowel in enumerate(CALIBRATION_VOWELS) if not self.user_training[vowel]),
            None,
        )
        self.calibration_complete = next_target is None
        self.target_index = next_target or 0
        self.samples_for_target = 0
        self._update_calibration_progress()
        self._populate_devices()
        self._refresh_gallery()
        self._set_output_image(self._first_image("Тишина"))

        self.audio_timer = QTimer(self)
        self.audio_timer.timeout.connect(self._update_audio_ui)
        self.audio_timer.start(20)
        self.output_timer = QTimer(self)
        self.output_timer.timeout.connect(self._send_virtual_frame)
        self.output_timer.start(16)
        self._start_audio()

    @staticmethod
    def _migrate_images(images: dict) -> dict[str, list[str]]:
        migrated = {group: [] for group in SOUND_GROUPS}
        for key, paths in images.items():
            group = key if key in SOUND_GROUPS else VOWEL_TO_GROUP.get(key)
            if group:
                migrated[group].extend(paths)
        return migrated

    def _load_settings(self) -> dict:
        try:
            return json.loads(self.settings_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"images": {}, "training": {}, "threshold": 5, "device": None}

    def _save_settings(self) -> None:
        self.settings["images"] = self.images
        self.settings["training"] = self.user_training
        self.settings["threshold"] = self.threshold_slider.value() if self.threshold_slider else 5
        self.settings["device"] = self.device_combo.currentData() if hasattr(self, "device_combo") else None
        self.settings_file.write_text(json.dumps(self.settings, ensure_ascii=False), encoding="utf-8")

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(62)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 8, 18, 8)
        header_layout.setSpacing(10)
        logo = QLabel()
        logo.setFixedSize(42, 42)
        logo.setStyleSheet("background: transparent; border: 0;")
        logo.setPixmap(
            QPixmap(str(Path(__file__).with_name("waveform-logo.svg"))).scaled(
                logo.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        header_layout.addWidget(logo)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        brand_title = QLabel("VOICECAM")
        brand_title.setObjectName("brandTitle")
        brand_subtitle = QLabel("ВИРТУАЛЬНАЯ КАМЕРА")
        brand_subtitle.setObjectName("brandSubtitle")
        brand.addWidget(brand_title)
        brand.addWidget(brand_subtitle)
        header_layout.addLayout(brand)
        header_layout.addStretch(1)
        self.header_status = QLabel("ОЖИДАНИЕ")
        self.header_status.setObjectName("statusBadge")
        header_layout.addWidget(self.header_status)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(236)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(16, 18, 16, 16)
        side_layout.setSpacing(8)
        self.nav_camera = QPushButton("НАСТРОЙКА КАМЕРЫ")
        self.nav_mic = QPushButton("НАСТРОЙКА МИКРОФОНА")
        for button in (self.nav_camera, self.nav_mic):
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setMinimumHeight(42)
            side_layout.addWidget(button)
        self.nav_camera.clicked.connect(lambda: self._show_page(0))
        self.nav_mic.clicked.connect(lambda: self._show_page(1))
        side_layout.addStretch(1)
        note = QLabel("ОБРАБОТКА\nЛОКАЛЬНО")
        note.setObjectName("sideNote")
        side_layout.addWidget(note)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._make_camera_page())
        self.pages.addWidget(self._make_microphone_page())
        content_layout.addWidget(sidebar)
        content_layout.addWidget(self.pages, 1)
        root_layout.addWidget(header)
        root_layout.addWidget(content, 1)
        self.setCentralWidget(root)
        self.nav_camera.setChecked(True)

        self.setStyleSheet("""
            QWidget { color: #e6f6f4; font-family: 'Segoe UI'; font-size: 13px; }
            QWidget#root { background: #0c1214; }
            QFrame#header { background: #131c1f; border-bottom: 2px solid #00e5c0; }
            QLabel#brandTitle { font-size: 15px; font-weight: 600; color: #e6f6f4; }
            QLabel#brandSubtitle { color: #00e5c0; font-size: 10px; }
            QLabel#statusBadge { color: #00e5c0; background: #112826; border: 1px solid #00e5c0; padding: 4px 9px; font-size: 10px; font-weight: 600; }
            QWidget#sidebar { background: #0c1214; border-right: 1px solid #203137; }
            QLabel#sideNote { color: #56736d; font-size: 10px; letter-spacing: 1px; }
            QPushButton#navButton { text-align: left; color: #8aa6a1; background: transparent; border: 1px solid transparent; padding: 0 10px; font-size: 11px; }
            QPushButton#navButton:checked, QPushButton#navButton:hover { color: #00e5c0; background: #131c1f; border-color: #203137; }
            QFrame#panel { background: #131c1f; border: 1px solid #203137; }
            QLabel#pageTitle { font-size: 19px; font-weight: 600; }
            QLabel#sectionTitle { color: #00e5c0; font-size: 10px; font-weight: 600; }
            QFrame#sectionDivider { background: #203137; border: 0; max-height: 1px; }
            QLabel#muted { color: #8aa6a1; }
            QLabel#preview { background: #080d0f; border: 1px solid #203137; color: #56736d; }
            QLabel#bigLetter { color: #00e5c0; font-size: 68px; font-weight: 600; }
            QPushButton, QComboBox { background: #1a272b; border: 1px solid #294047; padding: 8px 10px; min-height: 20px; }
            QPushButton:hover, QComboBox:hover { border-color: #008775; }
            QPushButton:checked, QPushButton#primary { color: #0c1214; background: #00e5c0; border-color: #00e5c0; font-weight: 700; }
            QPushButton:disabled { color: #56736d; background: #131c1f; border-color: #203137; }
            QComboBox::drop-down { border: 0; width: 24px; }
            QComboBox QAbstractItemView { background: #131c1f; selection-background-color: #00594d; }
            QListWidget { background: #0c1214; border: 1px solid #203137; padding: 3px; }
            QListWidget::item { padding: 7px; }
            QListWidget::item:selected { color: #00e5c0; background: #112826; }
            QSlider::groove:horizontal { height: 4px; background: #203137; }
            QSlider::handle:horizontal { width: 12px; margin: -5px 0; background: #00e5c0; }
            QProgressBar { height: 10px; background: #1a272b; border: 0; text-align: center; }
            QProgressBar::chunk { background: #00e5c0; }
        """)

    def _make_camera_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(14)
        heading = QLabel("Настройка камеры")
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)

        preview_panel, preview_layout = make_panel()
        top = QHBoxLayout()
        title = self._section_heading("ПРЯМОЙ ВЫХОД")
        self.output_status = QLabel("Виртуальная камера выключена")
        self.output_status.setObjectName("muted")
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.output_status)
        preview_layout.addLayout(top)
        self.camera_view = QLabel("Загрузите изображения для звуковых групп")
        self.camera_view.setObjectName("preview")
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setMinimumHeight(260)
        self.camera_view.setMaximumHeight(390)
        preview_layout.addWidget(self.camera_view, 1)
        controls = QHBoxLayout()
        self.image_label = QLabel("Ожидание звука")
        self.image_label.setObjectName("muted")
        controls.addWidget(self.image_label)
        self.waveform = WaveformDisplay()
        controls.addWidget(self.waveform, 1)
        self.output_button = QPushButton("ЗАПУСТИТЬ ВИРТУАЛЬНУЮ КАМЕРУ")
        self.output_button.setObjectName("primary")
        self.output_button.clicked.connect(self._toggle_virtual_camera)
        controls.addWidget(self.output_button)
        preview_layout.addLayout(controls)
        layout.addWidget(preview_panel, 3)

        assets_panel, assets_layout = make_panel()
        section = self._section_heading("ИЗОБРАЖЕНИЯ ПО ЗВУКАМ")
        assets_layout.addWidget(section)
        alphabet = QGridLayout()
        alphabet.setSpacing(3)
        for index, group in enumerate(SOUND_GROUPS):
            button = QPushButton(group)
            button.setCheckable(True)
            button.setMinimumHeight(36)
            button.clicked.connect(lambda checked=False, value=group: self._select_gallery_group(value))
            self.letter_buttons[group] = button
            alphabet.addWidget(button, index // 3, index % 3)
        self.letter_buttons[self.gallery_group].setChecked(True)
        assets_layout.addLayout(alphabet)

        gallery_row = QHBoxLayout()
        self.gallery = QListWidget()
        self.gallery.setMaximumHeight(100)
        self.gallery.currentRowChanged.connect(self._gallery_selection_changed)
        gallery_row.addWidget(self.gallery, 1)
        action_col = QVBoxLayout()
        add_button = QPushButton("ДОБАВИТЬ ИЗОБРАЖЕНИЯ")
        add_button.clicked.connect(self._add_images)
        remove_button = QPushButton("УДАЛИТЬ ВЫБРАННОЕ")
        remove_button.clicked.connect(self._remove_image)
        action_col.addWidget(add_button)
        action_col.addWidget(remove_button)
        action_col.addStretch(1)
        gallery_row.addLayout(action_col)
        assets_layout.addLayout(gallery_row)
        hint = QLabel("Громкость выбирает вариант внутри звуковой группы")
        hint.setObjectName("muted")
        assets_layout.addWidget(hint)
        layout.addWidget(assets_panel, 2)
        return page

    def _make_microphone_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(14)
        heading = QLabel("Настройка микрофона")
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)

        input_panel, input_layout = make_panel()
        title = self._section_heading("ВХОДНОЕ УСТРОЙСТВО")
        input_layout.addWidget(title)
        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._device_changed)
        device_row.addWidget(self.device_combo, 1)
        self.test_button = QPushButton("ПРОВЕРИТЬ МИКРОФОН")
        self.test_button.clicked.connect(self._toggle_test)
        device_row.addWidget(self.test_button)
        input_layout.addLayout(device_row)
        level_row = QHBoxLayout()
        level_title = QLabel("Уровень сигнала")
        level_title.setObjectName("muted")
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setTextVisible(False)
        self.level_value = QLabel("0 %")
        self.level_value.setMinimumWidth(42)
        level_row.addWidget(level_title)
        level_row.addWidget(self.level_bar, 1)
        level_row.addWidget(self.level_value)
        input_layout.addLayout(level_row)
        threshold_row = QHBoxLayout()
        threshold_title = QLabel("Порог голоса / тишины")
        threshold_title.setObjectName("muted")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 25)
        self.threshold_slider.setValue(int(self.settings.get("threshold", 5)))
        self.threshold_slider.valueChanged.connect(self._save_settings)
        self.threshold_value = QLabel(f"{self.threshold_slider.value()} %")
        self.threshold_slider.valueChanged.connect(lambda value: self.threshold_value.setText(f"{value} %"))
        threshold_row.addWidget(threshold_title)
        threshold_row.addWidget(self.threshold_slider, 1)
        threshold_row.addWidget(self.threshold_value)
        input_layout.addLayout(threshold_row)
        self.mic_status = QLabel("Выберите микрофон и проверьте сигнал")
        self.mic_status.setObjectName("muted")
        input_layout.addWidget(self.mic_status)
        layout.addWidget(input_panel)

        calibration_panel, calibration_layout = make_panel()
        title = self._section_heading("КАЛИБРОВКА ГЛАСНЫХ")
        calibration_layout.addWidget(title)
        instruction = QLabel("Калибровка необязательна: типовые значения уже работают. Для настройки произнесите А, О, У, Ы и Э по одному разу.")
        instruction.setWordWrap(True)
        instruction.setObjectName("muted")
        calibration_layout.addWidget(instruction)
        self.calibration_letter = QLabel("А")
        self.calibration_letter.setObjectName("bigLetter")
        self.calibration_letter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        calibration_layout.addWidget(self.calibration_letter)
        self.calibration_group = QLabel()
        self.calibration_group.setObjectName("sectionTitle")
        self.calibration_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        calibration_layout.addWidget(self.calibration_group)
        self.calibration_progress = QLabel()
        self.calibration_progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        calibration_layout.addWidget(self.calibration_progress)
        action_row = QHBoxLayout()
        self.capture_button = QPushButton("ЗАПИСАТЬ ОБРАЗЕЦ")
        self.capture_button.setObjectName("primary")
        self.capture_button.clicked.connect(self._capture_calibration_sample)
        action_row.addWidget(self.capture_button)
        reset_button = QPushButton("СБРОСИТЬ КАЛИБРОВКУ")
        reset_button.clicked.connect(self._reset_calibration)
        action_row.addWidget(reset_button)
        calibration_layout.addLayout(action_row)
        self.calibration_status = QLabel("Калибровка ещё не проводилась")
        self.calibration_status.setObjectName("muted")
        self.calibration_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        calibration_layout.addWidget(self.calibration_status)
        layout.addWidget(calibration_panel, 1)
        layout.addStretch(1)
        self._update_calibration_progress()
        return page

    def _show_page(self, page_index: int) -> None:
        self.pages.setCurrentIndex(page_index)
        self.nav_camera.setChecked(page_index == 0)
        self.nav_mic.setChecked(page_index == 1)

    def _section_heading(self, text: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        divider = QFrame()
        divider.setObjectName("sectionDivider")
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(label)
        layout.addWidget(divider, 1)
        return row

    def _populate_devices(self) -> None:
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        try:
            devices = sd.query_devices()
            default = sd.default.device[0]
            for index, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    self.device_combo.addItem(device["name"], index)
                    if index == self.settings.get("device") or (self.settings.get("device") is None and index == default):
                        self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
        except Exception as error:
            self.mic_status.setText(f"Не удалось получить список устройств: {error}")
        self.device_combo.blockSignals(False)

    def _start_audio(self) -> None:
        if self.device_combo.count() == 0:
            self.mic_status.setText("Входные устройства не найдены")
            return
        try:
            self.audio.start(self.device_combo.currentData())
            self.mic_status.setText("Микрофон подключён · сигнал обрабатывается локально")
        except Exception as error:
            self.mic_status.setText(f"Не удалось открыть микрофон: {error}")

    def _device_changed(self, _index: int) -> None:
        if not hasattr(self, "audio") or self.device_combo.currentData() is None:
            return
        self._start_audio()
        self._save_settings()

    def _update_audio_ui(self) -> None:
        level = self.audio.level
        self.waveform.set_levels(self.audio.waveform_snapshot())
        percent = min(100, int(level * 250))
        self.level_bar.setValue(percent)
        self.level_value.setText(f"{percent} %")
        if self.testing:
            self.mic_status.setText("Сигнал есть" if percent >= 2 else "Говорите в микрофон")
        if self.capture_pending:
            return
        threshold = self.threshold_slider.value() / 250.0
        if level < threshold:
            self.prediction_streak = 0
            if self.current_group != "Тишина" or self.last_emitted != "Тишина":
                self.last_emitted = "Тишина"
                self._show_group("Тишина", 0)
            return
        if not self.training:
            return
        audio = self.audio.snapshot(0.14)
        if audio.size < int(SAMPLE_RATE * 0.12):
            return
        features = extract_mfcc_features(audio)
        prediction = self._predict(features)
        if prediction is None:
            return
        group = VOWEL_TO_GROUP[prediction]
        self.last_prediction = prediction
        self.prediction_streak = 1
        if group != self.last_emitted:
            self.last_emitted = group
        self.last_emitted_at = time.monotonic()
        self._show_group(group, percent)

    def _predict(self, features: np.ndarray | None) -> str | None:
        if features is None:
            return None
        prototypes = []
        labels = []
        for letter in VOWELS:
            samples = self.training.get(letter, [])
            valid = [np.asarray(sample, dtype=np.float32) for sample in samples]
            if valid:
                prototypes.append(np.mean(valid, axis=0))
                labels.append(letter)
        if not prototypes:
            return None
        matrix = np.stack(prototypes)
        scale = np.maximum(np.std(matrix, axis=0), 0.35)
        distances = np.mean(((matrix - features) / scale) ** 2, axis=1)
        return labels[int(np.argmin(distances))]

    def _show_group(self, group: str, volume_percent: int) -> None:
        self.current_group = group
        paths = self._existing_images(group)
        if not paths:
            if self.current_image_path:
                self._set_output_image("")
            status = f"{group} · изображение не добавлено"
            if self.image_label.text() != status:
                self.image_label.setText(status)
            return
        normalized = min(0.999, max(0.0, (volume_percent - self.threshold_slider.value()) / max(1, 100 - self.threshold_slider.value())))
        selected = paths[min(len(paths) - 1, int(normalized * len(paths)))]
        if selected != self.current_image_path:
            self._set_output_image(selected)
        detail = "тишина" if group == "Тишина" else f"уровень {volume_percent} %"
        self.image_label.setText(f"{group} · {detail}")

    def _toggle_test(self) -> None:
        self.testing = not self.testing
        self.test_button.setText("ОСТАНОВИТЬ ПРОВЕРКУ" if self.testing else "ПРОВЕРИТЬ МИКРОФОН")
        if not self.testing:
            self.mic_status.setText("Проверка остановлена")

    def _capture_calibration_sample(self) -> None:
        if self.audio.stream is None:
            self.calibration_status.setText("Сначала подключите рабочий микрофон")
            return
        if self.capture_pending:
            return
        self.capture_pending = True
        self.capture_button.setEnabled(False)
        self.capture_button.setText("ЗАПИСЬ · ГОВОРИТЕ СЕЙЧАС")
        self.calibration_status.setText("Запись образца, произнесите показанную гласную...")
        QTimer.singleShot(900, self._finish_calibration_capture)

    def _finish_calibration_capture(self) -> None:
        audio = self.audio.snapshot(0.82)
        minimum_level = self.threshold_slider.value() / 250.0
        if audio.size and float(np.sqrt(np.mean(audio * audio))) < minimum_level:
            audio = np.array([], dtype=np.float32)
        features = extract_mfcc_features(audio)
        self.capture_pending = False
        self.capture_button.setEnabled(True)
        self.capture_button.setText("ЗАПИСАТЬ ОБРАЗЕЦ")
        if features is None:
            self.calibration_status.setText("Сигнал тихий или слишком короткий. Попробуйте ещё раз.")
            return
        vowel = CALIBRATION_VOWELS[self.target_index]
        sample = features.tolist()
        self.user_training[vowel] = [sample]
        self.training[vowel] = [sample]
        self._save_settings()
        self.target_index += 1
        if self.target_index >= len(CALIBRATION_VOWELS):
            self.target_index = 0
            self.calibration_complete = True
        self._update_calibration_progress()

    def _update_calibration_progress(self) -> None:
        if self.calibration_complete:
            self.calibration_letter.setText("✓")
            self.calibration_group.setText("ПЕРСОНАЛЬНАЯ КАЛИБРОВКА ГОТОВА")
            self.calibration_progress.setText("А · О · У · Ы · Э · по одному образцу")
            self.calibration_status.setText("Тишина определяется по уровню сигнала")
            self.capture_button.setEnabled(False)
            return
        vowel = CALIBRATION_VOWELS[self.target_index]
        group = VOWEL_TO_GROUP[vowel]
        self.calibration_letter.setText(vowel)
        self.calibration_group.setText(f"ГРУППА {group.upper()}")
        self.calibration_progress.setText(
            f"Гласная {self.target_index + 1} из {len(CALIBRATION_VOWELS)} · один образец"
        )
        self.calibration_status.setText("Калибровка необязательна · сейчас используются типовые значения")
        self.capture_button.setEnabled(True)

    def _reset_calibration(self) -> None:
        answer = QMessageBox.question(self, "Сбросить калибровку", "Удалить все записанные образцы гласных?")
        if answer == QMessageBox.StandardButton.Yes:
            self.user_training = {vowel: [] for vowel in CALIBRATION_VOWELS}
            self.training = make_default_prototypes()
            self.target_index = 0
            self.samples_for_target = 0
            self.calibration_complete = False
            self.last_emitted = ""
            self._save_settings()
            self._update_calibration_progress()

    def _select_gallery_group(self, group: str) -> None:
        self.gallery_group = group
        for key, button in self.letter_buttons.items():
            button.setChecked(key == group)
        self._refresh_gallery()

    def _existing_images(self, letter: str) -> list[str]:
        return [path for path in self.images.get(letter, []) if Path(path).is_file()]

    def _first_image(self, letter: str) -> str:
        paths = self._existing_images(letter)
        return paths[0] if paths else ""

    def _refresh_gallery(self) -> None:
        self.gallery.blockSignals(True)
        self.gallery.clear()
        for path in self._existing_images(self.gallery_group):
            self.gallery.addItem(Path(path).name)
        if self.gallery.count():
            self.gallery.setCurrentRow(0)
        self.gallery.blockSignals(False)
        if self.gallery.count():
            self._gallery_selection_changed(0)
        elif self.gallery_group == self.current_group:
            self._set_output_image("")

    def _gallery_selection_changed(self, row: int) -> None:
        paths = self._existing_images(self.gallery_group)
        if 0 <= row < len(paths):
            self.last_emitted = ""
            self._set_output_image(paths[row])
            self.image_label.setText(f"Предпросмотр: {self.gallery_group}")

    def _add_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Добавить изображения", "", "Изображения (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not files:
            return
        added = self.images.setdefault(self.gallery_group, [])
        for source in files:
            suffix = Path(source).suffix.lower()
            destination = self.asset_dir / f"{uuid.uuid4().hex[:10]}{suffix}"
            shutil.copy2(source, destination)
            added.append(str(destination))
        self._save_settings()
        self._refresh_gallery()

    def _remove_image(self) -> None:
        row = self.gallery.currentRow()
        paths = self._existing_images(self.gallery_group)
        if not 0 <= row < len(paths):
            return
        path = paths[row]
        self.images[self.gallery_group].remove(path)
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass
        self._set_output_image(self._first_image(self.gallery_group))
        self._save_settings()
        self._refresh_gallery()

    def _set_output_image(self, path: str) -> None:
        self.current_image_path = path if path and Path(path).is_file() else ""
        if self.current_image_path:
            self.output_pixmap = QPixmap(self.current_image_path)
            self.camera_view.setPixmap(self.output_pixmap.scaled(self.camera_view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.camera_view.setText("")
        else:
            self.output_pixmap = QPixmap()
            self.camera_view.setPixmap(QPixmap())
            self.camera_view.setText("Загрузите изображения для звуковых групп")
        self._build_virtual_frame()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.current_image_path:
            self._set_output_image(self.current_image_path)

    def _toggle_virtual_camera(self) -> None:
        if self.virtual_camera is not None:
            self.virtual_camera.close()
            self.virtual_camera = None
            self.output_button.setText("ЗАПУСТИТЬ ВИРТУАЛЬНУЮ КАМЕРУ")
            self.output_status.setText("Виртуальная камера выключена")
            self.header_status.setText("КАМЕРА ВЫКЛ")
            return
        if pyvirtualcam is None:
            QMessageBox.warning(self, "Нет pyvirtualcam", "Установите зависимости командой: python -m pip install -r requirements.txt")
            return
        try:
            self.virtual_camera = pyvirtualcam.Camera(
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
                fps=60,
                fmt=pyvirtualcam.PixelFormat.RGB,
            )
            self.output_button.setText("ОСТАНОВИТЬ ВИРТУАЛЬНУЮ КАМЕРУ")
            self.output_status.setText(f"В эфире · {self.virtual_camera.device}")
            self.header_status.setText("КАМЕРА В ЭФИРЕ")
        except Exception as error:
            self.virtual_camera = None
            QMessageBox.warning(
                self,
                "Виртуальная камера недоступна",
                f"Не удалось создать устройство. В Windows установите и запустите OBS Virtual Camera.\n\n{error}",
            )

    def _send_virtual_frame(self) -> None:
        if self.virtual_camera is None or self.virtual_frame_buffer is None:
            return
        try:
            self.virtual_camera.send(self.virtual_frame_buffer)
        except Exception as error:
            self.output_status.setText(f"Ошибка вывода: {error}")
            self.virtual_camera.close()
            self.virtual_camera = None

    def _build_virtual_frame(self) -> None:
        frame = QImage(FRAME_WIDTH, FRAME_HEIGHT, QImage.Format.Format_RGB888)
        frame.fill(QColor("#080d0f"))
        painter = QPainter(frame)
        if not self.output_pixmap.isNull():
            scaled = self.output_pixmap.scaled(FRAME_WIDTH, FRAME_HEIGHT, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap((FRAME_WIDTH - scaled.width()) // 2, (FRAME_HEIGHT - scaled.height()) // 2, scaled)
        else:
            painter.setPen(QPen(QColor("#56736d")))
            painter.drawText(frame.rect(), Qt.AlignmentFlag.AlignCenter, "VOICECAM")
        painter.end()
        image = frame.convertToFormat(QImage.Format.Format_RGB888)
        pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(FRAME_HEIGHT, image.bytesPerLine())
        self.virtual_frame_buffer = pixels[:, : FRAME_WIDTH * 3].reshape(FRAME_HEIGHT, FRAME_WIDTH, 3).copy()

    def closeEvent(self, event) -> None:
        self.audio.stop()
        if self.virtual_camera is not None:
            self.virtual_camera.close()
        self._save_settings()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setOrganizationName("voicecam")
    app.setApplicationName("voicecam")
    window = VoicecamWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()