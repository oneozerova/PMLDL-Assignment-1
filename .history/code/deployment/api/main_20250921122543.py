from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Определяем модель входных данных с 11 характеристиками винного датасета
class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# Загружаем модель, путь относительно файла main.py
model_path = os.path.join(os.path.dirname(__file__), '../../../models/wine_model.pkl')

# model = joblib.load(model_path)

absolute_model_path = os.path.abspath(model_path)
print("Model path:", absolute_model_path)

app = FastAPI()

# Эндпоинт для предсказания
@app.post("/predict")
def predict(input: WineInput):
    # Создаем np.array из всех входных признаков в правильном порядке
    features = np.array([
        input.fixed_acidity,
        input.volatile_acidity,
        input.citric_acid,
        input.residual_sugar,
        input.chlorides,
        input.free_sulfur_dioxide,
        input.total_sulfur_dioxide,
        input.density,
        input.pH,
        input.sulphates,
        input.alcohol
    ])
    # Делаем предсказание модели
    pred = model.predict([features])[0]
    # Возвращаем результат в читаемом виде
    return {"prediction": "Good wine" if pred == 1 else "Bad wine"}
