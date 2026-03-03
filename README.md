# Real-time Speech-to-Text and Translation | Распознавание и перевод речи в реальном времени

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
## English

This project provides a fast speech-to-text tool with real-time translation. Speech recognition (Vosk) runs offline; translation uses Google Translate (internet required). It can translate your microphone OR the audio coming from your computer (e.g., from Zoom, YouTube, or a browser).

### Setup

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

Or manually:
1. **Install Dependencies**: Python 3.9+ is required.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Download Models** (en, ru, de by default; add more in setup_models.py):
   ```bash
   python3 setup_models.py
   ```

### Capturing System Audio (Discord, Zoom, YouTube, etc.)
By default, the script listens to your microphone. To translate audio from Discord calls, Zoom, YouTube, or other apps:

#### 🪟 Windows – **Easiest: `--loopback` (no virtual cable)**
Captures directly from your default playback device (wherever Discord outputs):
```powershell
python main.py ru de --loopback
```
You hear Discord normally; the script captures the same audio. No VB-Audio or routing needed.

#### 🪟 Windows – Alternative: Virtual Cable
If `--loopback` doesn't work or you need different routing:
1. Install **VB-Audio Virtual Cable** (free): https://vb-audio.com/Cable/
2. After install, you'll have **CABLE Input** (playback) and **CABLE Output** (recording).
3. **Route Discord to the cable:** In Discord: User Settings → Voice & Video → Output Device → **CABLE Input**
4. **Hear the call while capturing:** Right-click the speaker icon → Sound settings → More sound settings → Recording tab → double-click **CABLE Output** → Listen tab → check "Listen to this device" → set "Playback through" to your headphones/speakers.
5. Run:
   ```powershell
   python list_devices.py
   python main.py ru de --device [ID]   # use ID for "CABLE Output"
   ```

#### 🍏 macOS
1. Install **BlackHole 2ch**: `brew install blackhole-2ch`.
2. Open **Audio MIDI Setup** app -> Create **Multi-Output Device**.
3. Check both your speakers/headphones AND BlackHole 2ch.
4. Set this Multi-Output Device as your system **Output** in Sound Settings.


#### 🐧 Linux (PulseAudio/PipeWire)
Use the monitor source of your output device:
```bash
pactl list sources | grep ".monitor"
```

### Usage
**Interactive (recommended):** Run without arguments to choose audio source and languages:
```bash
python main.py
```

**Command line:**
```bash
python main.py en ru                    # Microphone
python main.py en ru --loopback         # Discord/system audio (Windows)
python main.py en ru --device [ID]      # Specific device
python main.py en ru --no-menu          # Skip menu, use defaults
```

---

<a name="russian"></a>
## Русский

Этот проект — быстрый инструмент для распознавания и перевода речи в реальном времени. Распознавание (Vosk) работает офлайн; перевод использует Google Translate (требуется интернет). Он может переводить как ваш микрофон, так и звук, исходящий из компьютера (например, из Zoom, YouTube или браузера).

### Установка

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows (PowerShell):**
```powershell
.\setup.ps1
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

### Перехват системного звука (Discord, Zoom, YouTube и др.)
По умолчанию скрипт слушает микрофон. Чтобы переводить звук из Discord, Zoom, YouTube и других приложений:

#### 🪟 Windows – **Проще всего: `--loopback` (без виртуального кабеля)**
Захват напрямую с устройства воспроизведения по умолчанию (куда выводит Discord):
```powershell
python main.py ru de --loopback
```
Вы слышите Discord как обычно; скрипт захватывает тот же звук. VB-Audio и перенаправление не нужны.

#### 🪟 Windows – Альтернатива: виртуальный кабель
Если `--loopback` не работает или нужна другая маршрутизация:
1. Установите **VB-Audio Virtual Cable**: https://vb-audio.com/Cable/
2. В Discord: Настройки → Голос и видео → Устройство вывода → **CABLE Input**
3. ПКМ по иконке динамика → Параметры звука → Вкладка «Запись» → **CABLE Output** → «Прослушать» → включите воспроизведение через наушники.
4. `python list_devices.py` и `python main.py ru de --device [ID]`

#### 🍏 macOS
1. Установите **BlackHole 2ch**: `brew install blackhole-2ch`.
2. Откройте приложение **«Настройка Audio-MIDI»** -> Создайте **«Устройство с несколькими выходами»**.
3. Отметьте ваши динамики/наушники И BlackHole 2ch.
4. Выберите это устройство как основной **Выход** (Output) в системных настройках звука.

#### 🐧 Linux (PulseAudio/PipeWire)
Используйте "monitor" версию вашего устройства вывода:
```bash
pactl list sources | grep ".monitor"
```

### Использование
**Интерактивно:** `python main.py` — выберите источник звука и языки.

**Из командной строки:** `python main.py en ru`, `--loopback`, `--device [ID]`, `--no-menu`

## Requirements / Требования
- `vosk`, `deep-translator`, `silero-vad`, `sounddevice`, `numpy`, `torch`, `torchaudio`, `tqdm`, `requests`

**Note:** Translation uses deep-translator (Google Translate API) and requires an internet connection. Speech recognition (Vosk) remains offline.
