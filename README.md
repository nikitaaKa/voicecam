# voicecam

Desktop-приложение для виртуальной камеры. voicecam анализирует звук с микрофона, выбирает изображения по произнесённым гласным и переключается на отдельную группу при тишине.

## Возможности

- Пять звуковых групп: А/Я, О/Ё, У/Ю, И/Ы, Е/Э; шестая группа используется для тишины.
- Несколько изображений для каждой группы. Громкость голоса выбирает изображение внутри группы.
- Предпросмотр выходного кадра и визуализация уровня микрофона в реальном времени.
- Выбор микрофона, проверка сигнала и настройка порога тишины.
- Распознавание работает сразу на типовых акустических профилях. Персональная калибровка необязательна: можно записать по одному образцу А, О, У, Ы и Э.
- Настройки и изображения хранятся локально; аудио не отправляется в сеть.

## Установка

**Запустите exe файл ЛИБО:**

Требуется Python 3.10 или новее.

1. Склонируйте репозиторий или скачайте ZIP и распакуйте его.
2. Откройте терминал в папке `voicecam`.
3. Установите зависимости командой `python -m pip install -r requirements.txt`.
4. Запустите приложение командой `python app.py`.

Для виртуального устройства камеры в Windows установите OBS Studio и запустите **OBS Virtual Camera**. В других системах `pyvirtualcam` требует совместимый виртуальный драйвер. Без драйвера локальный предпросмотр продолжит работать.

## Распознавание

voicecam вычисляет MFCC-признаки короткого участка аудио и сравнивает их с типовыми либо записанными пользователем профилями гласных. Это экспериментальная классификация отдельных гласных, а не универсальное распознавание речи; результат зависит от микрофона, окружения и произношения. Персональная калибровка может улучшить результат.

## Конфиденциальность

Обработка аудио выполняется локально. Приложение не отправляет звук или изображения на внешние серверы.

---

# voicecam

A desktop virtual-camera app. voicecam analyzes microphone input, selects images based on spoken vowels, and switches to a dedicated group when the room is quiet.

## Features

- Five sound groups: А/Я, О/Ё, У/Ю, И/Ы, Е/Э, plus a sixth group for silence.
- Multiple images per group. Voice volume selects an image within the detected group.
- Live output preview and microphone-level visualization.
- Microphone selection, input-level test, and adjustable silence threshold.
- Works immediately with generic acoustic profiles. Optional calibration records one sample each for А, О, У, Ы, and Э.
- Settings and images stay on the local machine; audio is not uploaded.

## Installation

Python 3.10 or newer is required.

1. Clone the repository or download and extract its ZIP archive.
2. Open a terminal in the `voicecam` folder.
3. Install dependencies with `python -m pip install -r requirements.txt`.
4. Start the app with `python app.py`.

For a virtual camera device on Windows, install OBS Studio and start **OBS Virtual Camera**. On other systems, `pyvirtualcam` requires a compatible virtual-camera driver. The local preview remains available without a driver.

## Recognition

voicecam extracts MFCC features from a short audio segment and compares them with generic or user-recorded vowel profiles. This is experimental isolated-vowel classification, not general-purpose speech recognition. Results depend on the microphone, environment, and pronunciation; personal calibration may improve accuracy.

## Privacy

Audio is processed locally. The app does not send audio or images to external servers.