import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_PATH = Path("models/price_prediction_elasticnet.joblib")
COLS = ["X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station", "X4 number of convenience stores", "X5 latitude", "X6 longitude"]

def test_model_loads():
    assert MODEL_PATH.exists(), f"Missing model file: {MODEL_PATH.resolve()}"
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")

def test_model_returns_one_row_gt_0():
    model = joblib.load(MODEL_PATH)
    input = pd.DataFrame([{
        "X1 transaction date":2013.8,
        "X2 house age": 12.0,
        "X3 distance to the nearest MRT station": 500.0,
        "X4 number of convenience stores":0,
        "X5 latitude": 24.96500,
        "X6 longitude": 121.53700
    }], columns=COLS)
    y = model.predict(input)
    assert len(y) == 1
    assert y[0] >= 0


def test_model_out_of_location():
    """
    Scenario to check how the model behaves out of the location zone.
    Expected to fail and return a negative price
    """
    model = joblib.load(MODEL_PATH)
    input = pd.DataFrame([{
        "X1 transaction date":2013.8,
        "X2 house age": 12.0,
        "X3 distance to the nearest MRT station": 500.0,
        "X4 number of convenience stores":0,
        "X5 latitude": 45.96500,
        "X6 longitude": 21.53700
    }], columns=COLS)
    y = model.predict(input)
    assert len(y) == 1
    assert y[0] < 0