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
Captures directly from your default playback device:
```powershell
python main.py ru de --loopback
```

#### 🍏 macOS – **New: `--loopback` (ScreenCaptureKit)**
No virtual cable needed on macOS 12.3+! Captures system audio directly:
```bash
python main.py ru de --loopback
```
*Note: You must grant **Screen Recording** permission to your terminal when prompted.*

#### 🐧 Linux – **New: `--loopback` (Monitor source)**
Automatically finds and uses the "monitor" source of your PulseAudio/PipeWire output:
```bash
python main.py ru de --loopback
```

#### Alternative: Virtual Cables (All OS)
If `--loopback` doesn't work for your setup:
- **Windows**: Use **VB-Audio Virtual Cable**.
- **macOS**: Use **BlackHole 2ch**.
- **Linux**: Select a specific device ID from `python list_devices.py`.

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

#### 🪟 Windows – **Проще всего: `--loopback`**
Захват напрямую с устройства воспроизведения:
```powershell
python main.py ru de --loopback
```

#### 🍏 macOS – **Новое: `--loopback` (ScreenCaptureKit)**
Не нужен виртуальный кабель на macOS 12.3+! Прямой захват звука системы:
```bash
python main.py ru de --loopback
```
*Внимание: нужно разрешить «Запись экрана» (Screen Recording) для вашего терминала при появлении запроса.*

#### 🐧 Linux – **Новое: `--loopback` (Monitor source)**
Автоматический поиск и захват с "monitor" источника вашего PulseAudio/PipeWire:
```bash
python main.py ru de --loopback
```

#### Альтернатива: виртуальные кабели (Все ОС)
Если `--loopback` не работает:
- **Windows**: **VB-Audio Virtual Cable**.
- **macOS**: **BlackHole 2ch**.
- **Linux**: Выберите ID конкретного устройства через `python list_devices.py`.

### Использование
**Интерактивно:** `python main.py` — выберите источник звука и языки.

**Из командной строки:** `python main.py en ru`, `--loopback`, `--device [ID]`, `--no-menu`

## Requirements / Требования
- `vosk`, `deep-translator`, `silero-vad`, `sounddevice`, `numpy`, `torch`, `torchaudio`, `tqdm`, `requests`

**Note:** Translation uses deep-translator (Google Translate API) and requires an internet connection. Speech recognition (Vosk) remains offline.
