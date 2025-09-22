# Импортируем необходимые библиотеки
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Шаг 1: Загружаем датасет
# Датасет содержит различные химические параметры вина и оценку качества (quality)
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')

# Шаг 2: Предобработка
# Создаем бинарную целевую переменную quality_binary:
# Если качество >= 6, считаем хорошим и обозначаем 1, иначе 0
df['quality_binary'] = (df['quality'] >= 6).astype(int)

# Отделяем признаки от целевой переменной
X = df.drop(['quality', 'quality_binary'], axis=1)
y = df['quality_binary']

# Шаг 3: Разбиваем данные на обучающую и тестовую выборки
# use random_state для воспроизводимости результата
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Шаг 4: Обучаем модель случайного леса
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Шаг 5: Оцениваем точность модели на тестовой выборке
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.3f}")

# Шаг 6: Сохраняем обученную модель в файл
# Получаем директорию скрипта train.py
script_dir = os.path.dirname(os.path.abspath(__file__))

# Путь к корню проекта (поднимаемся на два уровня вверх из code/models)
project_root = os.path.abspath(os.path.join(script_dir, '../../'))

# Путь к директории models в корне проекта
model_dir = os.path.join(project_root, 'models')

# Полный путь к файлу для сохранения модели
model_path = os.path.join(model_dir, 'wine_model.pkl')

# Сохраняем модель
joblib.dump(model, model_path)
print(f"Модель сохранена в {model_path}")
