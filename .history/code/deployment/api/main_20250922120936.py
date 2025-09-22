from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Define Model
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

# Load Model
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
model_path = os.path.join(BASE_DIR, "models", "wine_model.pkl")

model = joblib.load(model_path)


app = FastAPI()

# Endpoint for the prediction
@app.post("/predict")
def predict(input: WineInput):
    # creatw np.array out of all features
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
    # make prediction
    pred = model.predict([features])[0]
    # returen the result
    return {"prediction": "Good wine" if pred == 1 else "Bad wine"}
