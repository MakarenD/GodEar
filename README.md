# Real-time Speech-to-Text and Translation | Распознавание и перевод речи в реальном времени

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
## English

This project provides a fast, offline speech-to-text and translation tool. It can translate your microphone OR the audio coming from your computer (e.g., from Zoom, YouTube, or a browser).

### Setup

For quick setup on macOS/Linux:
```bash
chmod +x setup.sh
./setup.sh
```

Or manually:
1. **Install Dependencies**: Python 3.9+ is required.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Download Models**:
   ```bash
   python3 setup_models.py
   ```

### Capturing System Audio (Translate your Interviewer)
By default, the script listens to your microphone. To translate your собеседник (interviewer/speaker), you need to route system audio to a virtual input:

#### 🍏 macOS
1. Install **BlackHole 2ch**: `brew install blackhole-2ch`.
2. Open **Audio MIDI Setup** app -> Create **Multi-Output Device**.
3. Check both your speakers/headphones AND BlackHole 2ch.
4. Set this Multi-Output Device as your system **Output** in Sound Settings.

#### 🪟 Windows
1. Install **VB-Audio Virtual Cable**.
2. Set "CABLE Input" as your Default Playback Device.
3. In this app, use `--device [ID]` where ID corresponds to "CABLE Output".
*Alternative:* Enable **"Stereo Mix"** in Sound Control Panel -> Recording tab.

#### 🐧 Linux (PulseAudio/PipeWire)
Use the monitor source of your output device:
```bash
pactl list sources | grep ".monitor"
```

### Usage
1. Find the ID of your virtual device:
   ```bash
   python3 list_devices.py
   ```
2. Run the translator with the device ID:
   ```bash
   python3 main.py en ru --device [ID]
   ```
   *Example:* `python3 main.py en ru --device 2`

---

<a name="russian"></a>
## Русский

Этот проект — быстрый офлайн-инструмент для распознавания и перевода речи. Он может переводить как ваш микрофон, так и звук, исходящий из компьютера (например, из Zoom, YouTube или браузера).

### Установка

Для быстрой установки на macOS/Linux:
```bash
chmod +x setup.sh
./setup.sh
```

Или вручную:
1. **Установка зависимостей**: Требуется Python 3.9+.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Загрузка моделей**:
   ```bash
   python3 setup_models.py
   ```

### Перехват системного звука (Перевод собеседника)
По умолчанию скрипт слушает микрофон. Чтобы переводить голос собеседника, нужно направить системный звук на виртуальный вход:

#### 🍏 macOS
1. Установите **BlackHole 2ch**: `brew install blackhole-2ch`.
2. Откройте приложение **«Настройка Audio-MIDI»** -> Создайте **«Устройство с несколькими выходами»**.
3. Отметьте ваши динамики/наушники И BlackHole 2ch.
4. Выберите это устройство как основной **Выход** (Output) в системных настройках звука.

#### 🪟 Windows
1. Установите **VB-Audio Virtual Cable**.
2. Выберите "CABLE Input" как устройство воспроизведения по умолчанию.
3. В приложении используйте `--device [ID]`, где ID соответствует "CABLE Output".
*Альтернатива:* Включите **«Стерео микшер»** (Stereo Mix) в Панели управления звуком -> Вкладка «Запись».

#### 🐧 Linux (PulseAudio/PipeWire)
Используйте "monitor" версию вашего устройства вывода:
```bash
pactl list sources | grep ".monitor"
```

### Использование
1. Найдите ID виртуального устройства:
   ```bash
   python3 list_devices.py
   ```
2. Запустите переводчик с указанием ID:
   ```bash
   python3 main.py en ru --device [ID]
   ```
   *Пример:* `python3 main.py en ru --device 2`

## Requirements / Требования
- `vosk`, `argostranslate`, `silero-vad`, `sounddevice`, `numpy`, `torch`, `torchaudio`, `tqdm`, `requests`
