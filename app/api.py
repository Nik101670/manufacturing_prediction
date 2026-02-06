from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

from app.inference import load_bundle, predict_from_dict

app = FastAPI(title="Manufacturing Prediction API")

# Load bundle
BASE_DIR = Path(__file__).resolve().parent.parent
BUNDLE_PATH = BASE_DIR / "model" / "model_bundle.pkl"

model, scaler, feature_columns = load_bundle(str(BUNDLE_PATH))


class PredictRequest(BaseModel):
    # accept any key-values
    data: dict


@app.get("/")
def home():
    return {"message": "API is running!", "status": "ok"}


@app.get("/features")
def features():
    return {"feature_columns": feature_columns}


@app.post("/predict")
def predict(req: PredictRequest):
    prediction = predict_from_dict(model, scaler, feature_columns, req.data)
    return {"prediction": prediction}
