from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from pathlib import Path
import logging
import joblib
import pandas as pd

logger = logging.getLogger("uvicorn.error")
MODEL_PATH = Path("models/price_prediction_elasticnet.joblib")
COLS = ["X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station", "X4 number of convenience stores", "X5 latitude", "X6 longitude"]
class PredictRequest(BaseModel):
    transaction_date: float = Field(ge=1900, le=2100)
    house_age: float = Field(ge=0.0, le=300)
    distance_to_MRT: float = Field(ge=0.0)
    number_of_CS: int = Field(ge=0)
    latitude: float = Field(ge=22, le=26)
    longitude: float = Field(ge=120, le=123)

# We load the model on startup, in case of error raise exception
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Launching application. Loading models...")
    if not MODEL_PATH.exists():
        logger.error("Model file not found. Stopping.")
        raise RuntimeError(f"Model file not found in {MODEL_PATH.resolve()}")
    app.state.model = joblib.load(MODEL_PATH)
    logger.info("Model loaded.")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    logger.info("Health request received.")
    return{"status": "OK"}

@app.post("/predict")
async def get_prediction(input: PredictRequest):
    logger.info("Price prediction called.")
    rows = {
        "X1 transaction date":input.transaction_date,
        "X2 house age":input.house_age,
        "X3 distance to the nearest MRT station":input.distance_to_MRT,
        "X4 number of convenience stores":input.number_of_CS,
        "X5 latitude":input.latitude,
        "X6 longitude":input.longitude
    }
    features = pd.DataFrame([rows], columns = COLS)
    logger.info("Values loaded for prediction.")
    pred = app.state.model.predict(features)
    logger.info("Returning prediction")
    return{"predicted_price": f"{float(pred[0]):.2f}"}