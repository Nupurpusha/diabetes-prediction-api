from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

# --- Input Schema ---
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# --- Predict Endpoint ---
@app.post("/predict_diabetes/")
async def predict_diabetes(data: DiabetesInput):
    """
    Predict diabetes based on input features using a pre-trained Random Forest classifier.
    """

    # Get model path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "diabetes_model.pkl")

    # Load model
    with open(model_path, "rb") as file:
        rf_classifier = pickle.load(file)

    # Prepare features
    input_features = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

    # Prediction and probability
    prediction = rf_classifier.predict(input_features)[0]
    probability = rf_classifier.predict_proba(input_features)[0][1]  # probability of having diabetes

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

# Run: uvicorn app:app --reload
