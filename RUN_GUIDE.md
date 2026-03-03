
# Execution Guide | Руководство по запуску

This guide explains how to set up and run the Speech-to-Text & Translation tool on different operating systems.
Это руководство объясняет, как настроить и запустить инструмент распознавания и перевода речи на различных операционных системах.

---

## English

### 1. Prerequisites
- **Python 3.9 or higher** must be installed.
- **Internet connection** is required for the first run to download models.

### 2. Setup (Virtual Environment)

#### **macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### **Windows**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Models
Run this script once to download the necessary neural networks (Vosk and Argos Translate):
```bash
python setup_models.py
```

### 4. Running the Application
To start translating from **English to Russian**:
```bash
python main.py en ru
```

To start translating from **Russian to English**:
```bash
python main.py ru en
```

---

## Русский

### 1. Требования
- Должен быть установлен **Python 3.9 или выше**.
- **Интернет-соединение** требуется только для первого запуска, чтобы скачать модели.

### 2. Настройка (Виртуальное окружение)

#### **macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### **Windows**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Загрузка моделей
Запустите этот скрипт один раз, чтобы скачать необходимые нейросети (Vosk и Argos Translate):
```bash
python setup_models.py
```

### 4. Запуск приложения
Чтобы начать перевод с **английского на русский**:
```bash
python main.py en ru
```

Чтобы начать перевод с **русского на английский**:
```bash
python main.py ru en
```

---

## Troubleshooting / Решение проблем
- **Microphone not working**: Run `python list_devices.py` to see available devices.
- **Микрофон не работает**: Запустите `python list_devices.py`, чтобы увидеть доступные устройства.
