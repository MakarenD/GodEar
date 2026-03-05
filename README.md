# Real-time Speech-to-Text and Translation | Распознавание и перевод речи в реальном времени

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
## English

This project provides a fast speech-to-text tool with real-time translation and voice synthesis. It supports multiple recognition engines including **Vosk** (fast, offline) and **OpenAI Whisper** (high accuracy). It can translate your microphone OR the audio coming from your computer (e.g., from Zoom, YouTube, or a browser), and even speak the translation back to you.

### Key Features
- **Voice Translation (TTS)**: Turn the app into a simultaneous interpreter. The program can speak the translated text in real-time using high-quality local voices (Silero TTS). Supports output to any audio device, including virtual cables for Discord/Zoom.
- **Speaker Diarization**: Intelligently identify different speakers in a conversation. Each speaker is assigned a unique name and color in the overlay, making it easy to follow meetings and dialogues (e.g., "[Speaker 1] Hello", "[Speaker 2] Hi").
- **Multiple Engines**: Choose between `Vosk` (real-time partials), `Whisper Medium` (highest accuracy), and `Whisper Lite/Base` (balanced).
- **Offline Translation**: Local, privacy-first translation using CTranslate2 and NLLB-200 (600M) without requiring internet access. Switchable on-the-fly in settings.
- **Translation Overlay & GUI**: Real-time, transparent, always-on-top window to display translations directly over your apps. Includes a System Tray icon and a graphical Settings window for on-the-fly adjustments.
- **Global Hotkeys**: Control the app while in-game or full-screen. Easily toggle the overlay visibility, enable drag mode to reposition the window, or mute/unmute audio capture directly from your keyboard.
- **Fast Response**: Optimized silence detection (~0.22s) for quicker translation delivery.
- **Native Loopback**: Capture system audio directly on macOS (ScreenCaptureKit), Windows (WASAPI), and Linux (Monitor).

### 🚀 Quick Start (Pre-built Binaries)
If you don't want to install Python and dependencies, you can download a standalone version:
1. Go to the **Releases** section on GitHub.
2. Download the archive for your operating system (e.g., `speech-to-text-macos.zip` or `speech-to-text-windows.zip`).
3. Extract the archive and run the executable file (`speech-to-text`).
*Note: The first run will automatically download the necessary speech recognition models.*

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
   python3 -m venv venv39
   source venv39/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Download Models** (for Vosk):
   ```bash
   python3 setup_models.py
   ```
   *Note: Whisper, Diarization, and TTS models are downloaded automatically on first run.*

### Recognition Engines
You can choose the engine via the interactive menu or the `--engine` flag:
- **Vosk** (`--engine vosk`): Extremely fast, low latency, shows partial results as you speak.
- **Whisper Medium** (`--engine whisper`): Best accuracy, requires more CPU/GPU resources.
- **Whisper Lite** (`--engine whisper-lite`): Faster than Medium, uses the `base` model.
- **Vosk + Whisper Hybrid** (`--engine vosk+whisper`): Combines Vosk's instant response with Whisper's high accuracy. Shows fast partials and updates them with perfect translation upon pause.
- **Vosk + Whisper Lite Hybrid** (`--engine vosk+whisper-lite`): Same hybrid logic but with the faster `base` Whisper model.

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
python main.py en ru --overlay                         # Show translation overlay
python main.py en ru --diarization                     # Enable speaker identification
python main.py en ru --engine vosk+whisper --loopback  # Hybrid Mode with system audio
python main.py en ru --engine whisper --loopback       # Full Whisper (Medium model)
python main.py en ru --engine whisper-lite --loopback  # Whisper Lite (Base model)
python main.py en ru --engine vosk --no-menu           # Vosk (Fast)
```

---

<a name="russian"></a>
## Русский

Этот проект — инструмент для распознавания, перевода и озвучки речи в реальном времени. Поддерживает несколько движков: **Vosk** (быстрый, офлайн) и **OpenAI Whisper** (высокая точность). Переводит как микрофон, так и системный звук (Zoom, YouTube, игры), а также может озвучивать перевод голосом.

### Основные возможности
- **Голосовой переводчик (TTS)**: Превратите программу в полноценного синхронного переводчика. Система может озвучивать перевод в реальном времени, используя качественные локальные голоса (Silero TTS). Поддерживается вывод на любое устройство, включая виртуальные кабели для трансляции в Discord/Zoom.
- **Разделение спикеров (Diarization)**: Умное распознавание разных участников диалога. Каждому спикеру назначается имя и уникальный цвет в оверлее, что позволяет легко следить за ходом совещаний (например, "[Speaker 1] Привет", "[Speaker 2] Как дела?").
- **Несколько движков**: Выбор между `Vosk` (мгновенно), `Whisper Medium` (максимальная точность) и `Whisper Lite/Base` (быстрее оригинала).
- **Офлайн Перевод**: Локальный, приватный перевод с помощью CTranslate2 и NLLB-200 (600M) без подключения к интернету. Переключение "на лету" в настройках.
- **Оверлей и GUI**: Прозрачное окно поверх всех приложений для вывода перевода. Включает иконку в системном трее и графическое окно настроек для смены параметров "на лету".
- **Глобальные хоткеи**: Управляйте программой прямо во время игры или конференции. Настройте горячие клавиши для скрытия/показа оверлея, режима перемещения окна (drag mode) или паузы перевода (mute).
- **Быстрый отклик**: Оптимизированное детектирование пауз (~0.22 сек) для частого вывода перевода.
- **Нативный захват**: Прямой перехват звука системы на macOS (ScreenCaptureKit), Windows (WASAPI) и Linux (Monitor).

### 🚀 Быстрый старт (Готовые сборки)
Если вы не хотите устанавливать Python и зависимости, вы можете скачать готовую версию приложения:
1. Перейдите в раздел **Releases** (Релизы) на GitHub.
2. Скачайте архив для вашей операционной системы (например, `speech-to-text-macos.zip` или `speech-to-text-windows.zip`).
3. Распакуйте архив и запустите исполняемый файл (`speech-to-text`).
*Примечание: При первом запуске приложение автоматически скачает необходимые модели для распознавания речи.*

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
   python3 -m venv venv39
   source venv39/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Загрузка моделей** (для Vosk):
   ```bash
   python3 setup_models.py
   ```
   *Примечание: Модели Whisper, Diarization и TTS скачиваются автоматически при первом запуске.*

### Движки распознавания
Выбор через интерактивное меню или флаг `--engine`:
- **Vosk** (`--engine vosk`): Очень быстрый, выводит текст "на лету" (частичные результаты).
- **Whisper Medium** (`--engine whisper`): Самая высокая точность, требует больше ресурсов CPU/GPU.
- **Whisper Lite** (`--engine whisper-lite`): Быстрее версии Medium, использует модель `base`.
- **Гибридный Vosk + Whisper** (`--engine vosk+whisper`): Совмещает мгновенный отклик Vosk и идеальную точность Whisper. Сначала выводит быстрый перевод, а в паузах заменяет его на точный.
- **Гибридный Vosk + Whisper Lite** (`--engine vosk+whisper-lite`): Та же гибридная логика, но с более легкой моделью `base`.

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
python main.py en ru --overlay                         # Показать перевод в оверлее
python main.py en ru --diarization                     # Включить распознавание спикеров
python main.py en ru --engine vosk+whisper --loopback  # Гибридный режим с системным звуком
python main.py en ru --engine whisper --loopback       # Полный Whisper (модель Medium)
python main.py en ru --engine whisper-lite --loopback  # Whisper Lite (модель Base)
python main.py en ru --engine vosk --no-menu           # Vosk (Быстрый)
```

---

## Requirements / Требования
- `vosk`, `openai-whisper`, `deep-translator`, `silero-vad`, `sounddevice`, `numpy`, `torch`, `torchaudio`, `tqdm`, `requests`, `PyQt6`, `pynput`, `ctranslate2`, `transformers`, `huggingface-hub`, `sentencepiece`, `speechbrain`, `scipy`, `scikit-learn`, `omegaconf`

**Note:** Online translation uses Google Translate API and requires internet. Offline translation (CTranslate2/NLLB) and Speech recognition remain entirely local.
