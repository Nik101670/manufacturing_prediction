from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import logging

from app.inference import load_bundle, predict_from_dict

app = FastAPI(title="Manufacturing Prediction API")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load bundle
BASE_DIR = Path(__file__).resolve().parent.parent
BUNDLE_PATH = BASE_DIR / "model" / "model_bundle_perfect.pkl"

try:
    model, scaler_X, scaler_Y, feature_columns = load_bundle(str(BUNDLE_PATH))
    logger.info(f" Model loaded successfully from {BUNDLE_PATH}")
    logger.info(f" Features: {len(feature_columns)}")
except Exception as e:
    logger.error(f" Failed to load model: {e}")
    raise RuntimeError(f"Model loading failed: {e}")

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
    prediction = predict_from_dict(model, scaler_X, scaler_Y, feature_columns, req.data)
    return {"prediction": prediction}
