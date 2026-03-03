# Real-time Speech-to-Text and Translation | Распознавание и перевод речи в реальном времени

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
## English

This project provides a fast speech-to-text tool with real-time translation. It supports multiple recognition engines including **Vosk** (fast, offline) and **OpenAI Whisper** (high accuracy). It can translate your microphone OR the audio coming from your computer (e.g., from Zoom, YouTube, or a browser).

### Key Features
- **Multiple Engines**: Choose between `Vosk` (real-time partials), `Whisper Medium` (highest accuracy), and `Whisper Lite/Base` (balanced).
- **Fast Response**: Optimized silence detection (~0.22s) for quicker translation delivery.
- **Native Loopback**: Capture system audio directly on macOS (ScreenCaptureKit), Windows (WASAPI), and Linux (Monitor).

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
2. **Download Models** (for Vosk):
   ```bash
   python3 setup_models.py
   ```
   *Note: Whisper models are downloaded automatically on first run.*

### Recognition Engines
You can choose the engine via the interactive menu or the `--engine` flag:
- **Vosk** (`--engine vosk`): Extremely fast, low latency, shows partial results as you speak.
- **Whisper Medium** (`--engine whisper`): Best accuracy, requires more CPU/GPU resources.
- **Whisper Lite** (`--engine whisper-lite`): Faster than Medium, uses the `base` model.

### Capturing System Audio (Discord, Zoom, YouTube, etc.)
Use the `--loopback` flag to capture audio directly from your speakers/playback:

#### 🪟 Windows (WASAPI)
```powershell
python main.py ru de --loopback
```

#### 🍏 macOS (ScreenCaptureKit)
Works on macOS 12.3+ without virtual cables:
```bash
python main.py ru de --loopback
```
*Note: Grant **Screen Recording** permission to your terminal.*

#### 🐧 Linux (Monitor source)
```bash
python main.py ru de --loopback
```

### Usage
**Interactive (recommended):**
```bash
python main.py
```

**Command line:**
```bash
python main.py en ru --engine whisper --loopback       # Full Whisper (Medium model)
python main.py en ru --engine whisper-lite --loopback  # Whisper Lite (Base model)
python main.py en ru --engine vosk --no-menu           # Vosk (Fast)
```

---

<a name="russian"></a>
## Русский

Этот проект — инструмент для распознавания и перевода речи в реальном времени. Поддерживает несколько движков: **Vosk** (быстрый, офлайн) и **OpenAI Whisper** (высокая точность). Переводит как микрофон, так и системный звук (Zoom, YouTube, игры).

### Основные возможности
- **Несколько движков**: Выбор между `Vosk` (мгновенно), `Whisper Medium` (максимальная точность) и `Whisper Lite/Base` (быстрее оригинала).
- **Быстрый отклик**: Оптимизированное детектирование пауз (~0.22 сек) для частого вывода перевода.
- **Нативный захват**: Прямой перехват звука системы на macOS (ScreenCaptureKit), Windows (WASAPI) и Linux (Monitor).

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
1. **Установка зависимостей**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Загрузка моделей** (для Vosk):
   ```bash
   python3 setup_models.py
   ```
   *Примечание: Модели Whisper скачиваются автоматически при первом запуске.*

### Движки распознавания
Выбор через интерактивное меню или флаг `--engine`:
- **Vosk** (`--engine vosk`): Очень быстрый, выводит текст "на лету" (частичные результаты).
- **Whisper Medium** (`--engine whisper`): Самая высокая точность, требует больше ресурсов CPU/GPU.
- **Whisper Lite** (`--engine whisper-lite`): Быстрее версии Medium, использует модель `base`.

### Перехват системного звука (Discord, Zoom, YouTube и др.)
Используйте флаг `--loopback` для захвата звука напрямую из системы:

#### 🪟 Windows (WASAPI)
```powershell
python main.py ru de --loopback
```

#### 🍏 macOS (ScreenCaptureKit)
Работает на macOS 12.3+ без виртуальных кабелей:
```bash
python main.py ru de --loopback
```
*Внимание: разрешите «Запись экрана» (Screen Recording) для терминала.*

#### 🐧 Linux (Monitor source)
```bash
python main.py ru de --loopback
```

### Использование
**Интерактивно (рекомендуется):**
```bash
python main.py
```

**Из командной строки:**
```bash
python main.py en ru --engine whisper --loopback       # Полный Whisper (модель Medium)
python main.py en ru --engine whisper-lite --loopback  # Whisper Lite (модель Base)
python main.py en ru --engine vosk --no-menu           # Vosk (Быстрый)
```

---

## Requirements / Требования
- `vosk`, `openai-whisper`, `deep-translator`, `silero-vad`, `sounddevice`, `numpy`, `torch`, `torchaudio`, `tqdm`, `requests`

**Note:** Translation uses Google Translate API and requires internet. Speech recognition remains local.
