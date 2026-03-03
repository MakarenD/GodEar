# Real-time Speech-to-Text and Translation | Распознавание и перевод речи в реальном времени

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
## English

This project provides a fast, offline speech-to-text and translation tool using:
- **Vosk**: Speech recognition (ASR) using lightweight models for speed.
- **Silero VAD**: Voice Activity Detection to accurately segment sentences.
- **Argos Translate**: Fully offline machine translation based on CTranslate2.

### Setup

For quick setup on macOS/Linux:
```bash
chmod +x setup.sh
./setup.sh
```

Or manually:
1. **Install Dependencies**:
   Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Download Models**:
   Run the setup script to download Vosk and Argos Translate models.
   ```bash
   python3 setup_models.py
   ```

### Usage

Run the main script:
```bash
python3 main.py [source_lang] [target_lang]
```
Default is `en ru` (English to Russian).

Example:
```bash
python3 main.py en ru
```

- **Partial Results**: Shown as `>> text...` while you speak.
- **Final Translation**: Printed as `USER: [original]` and `TRAN: [translated]` after a short pause (VAD detected).

### Audio Issues
If you have multiple microphones, use `list_devices.py` to see available IDs.
```bash
python3 list_devices.py
```

---

<a name="russian"></a>
## Русский

Этот проект представляет собой быстрый офлайн-инструмент для распознавания и перевода речи с использованием:
- **Vosk**: Распознавание речи (ASR) с использованием легковесных моделей для максимальной скорости.
- **Silero VAD**: Детектор активности голоса для точного разделения на предложения.
- **Argos Translate**: Полностью автономный машинный перевод на базе CTranslate2.

### Установка

Для быстрой установки на macOS/Linux:
```bash
chmod +x setup.sh
./setup.sh
```

Или вручную:
1. **Установка зависимостей**:
   Убедитесь, что у вас установлен Python 3.9+. Рекомендуется использовать виртуальное окружение.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Загрузка моделей**:
   Запустите скрипт настройки для загрузки моделей Vosk и Argos Translate.
   ```bash
   python3 setup_models.py
   ```

### Использование

Запустите основной скрипт:
```bash
python3 main.py [исходный_язык] [целевой_язык]
```
По умолчанию используется `en ru` (Английский -> Русский).

Пример:
```bash
python3 main.py en ru
```

- **Промежуточные результаты**: Отображаются как `>> текст...` во время речи.
- **Финальный перевод**: Выводится в формате `USER: [оригинал]` и `TRAN: [перевод]` после обнаружения паузы (VAD).

### Проблемы со звуком
Если у вас несколько микрофонов, используйте `list_devices.py`, чтобы увидеть доступные ID устройств.
```bash
python3 list_devices.py
```

## Requirements / Требования
- `vosk`
- `argostranslate`
- `silero-vad`
- `sounddevice`
- `numpy`
- `torch`
- `torchaudio`
- `tqdm`
- `requests`
