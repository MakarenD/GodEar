#!/bin/bash

# Остановить скрипт при ошибке
set -e

echo "=== Speech-to-Text Setup ==="

# 1. Проверка Python
if ! command -v python3 &> /dev/null
then
    echo "Python3 не найден. Пожалуйста, установите его."
    exit 1
fi

# 2. Создание виртуального окружения
echo "Создание виртуального окружения (venv)..."
python3 -m venv venv

# 3. Активация окружения
source venv/bin/activate

# 4. Обновление pip
echo "Обновление pip..."
pip install --upgrade pip

# 5. Установка зависимостей
echo "Установка зависимостей из requirements.txt..."
pip install -r requirements.txt

# 6. Скачивание моделей
echo "Скачивание моделей Vosk и Argos Translate..."
python3 setup_models.py

echo "=== Установка завершена! ==="
echo "Для запуска проекта используйте:"
echo "source venv/bin/activate"
echo "python3 main.py"
