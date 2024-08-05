# warp-implementation

1. Клонирование репозитория
Сначала клонируйте репозиторий на вашу локальную машину:
git clone https://github.com/ваш-аккаунт/ваш-репозиторий.git
cd warp_implementation

2. Установка зависимостей
Создайте и активируйте виртуальное окружение (рекомендуется использовать venv или conda):

# Для venv
python -m venv env
source env/bin/activate  # Для Windows используйте `env\Scripts\activate`

# Для conda
conda create --name myenv python=3.8
conda activate myenv
Установите необходимые зависимости из файла requirements.txt:

pip install -r requirements.txt

3. Запуск обучения
Запустите основной скрипт обучения:
python reward_training.py
python warp_training.py

4. Проверка результатов
После завершения обучения вы можете найти сохраненную модель в директории results/warp_model, вписать ее в конфиг config.yaml
python evaluate.py

Результат эксперимента с изменением T можно найти в ноутбуке exp.ipynb
