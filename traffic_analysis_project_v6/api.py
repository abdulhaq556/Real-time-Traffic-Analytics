from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json

app = FastAPI()

# ==========================
# Load Models
# ==========================

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
loc_encoder = joblib.load("models/loc_encoder.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

with open("models/meta.json", "r") as f:
    meta = json.load(f)

# ==========================
# Request Schema
# ==========================

class TrafficRequest(BaseModel):
    location: str
    hour: int
    minute: int
    day_of_week: int
    lat: float
    lon: float
   

# ==========================
# Home
# ==========================

@app.get("/")
def home():
    return {"message": "API is running"}

# ==========================
# Predict
# ==========================

@app.post("/predict")
def predict(data: TrafficRequest):

    loc_encoded = loc_encoder.transform([data.location])[0]

    is_weekend = 1 if data.day_of_week in [5, 6] else 0

    is_peak_hour = 1 if (
        (7 <= data.hour <= 10) or
        (16 <= data.hour <= 19)
    ) else 0

    df = pd.DataFrame([{
        "hour": data.hour,
        "minute": data.minute,
        "day_of_week": data.day_of_week,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "lat": data.lat,
        "lon": data.lon,
        "loc_encoded": loc_encoded
    }])

    X = scaler.transform(df)

    prediction = model.predict(X)

    result = target_encoder.inverse_transform(prediction)

    return {
        "prediction": str(result[0])
    }
@app.get("/locations")
def get_locations():
    return {
        "locations": loc_encoder.classes_.tolist()
    }