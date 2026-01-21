from fastapi.testclient import TestClient
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.endpoints import app

def test_health_endpoint():
    """
    Correct input, expected to work
    """
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}

def test_prediction_correct_input():
    """
    Correct payload, expects 200 and returning a price greater than 0
    """
    correct_payload = {
        "transaction_date":2013.8,
        "house_age": 12.0,
        "distance_to_MRT": 500.0,
        "number_of_CS":0,
        "latitude": 24.96500,
        "longitude": 121.53700
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=correct_payload)
        assert response.status_code == 200
        output = response.json()
        assert "predicted_price" in output
        value = float(output["predicted_price"])
        assert value > 0.0

def test_prediction_negative_input():
    """
    Includes negative distance and latitude out of the bounds.
    Is expected to fail.
    """
    negative_payload = {
        "transaction_date":2013.8,
        "house_age": 12.0,
        "distance_to_MRT": -500.0,
        "number_of_CS":0,
        "latitude": -24.96500,
        "longitude": 121.53700
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=negative_payload)
        assert response.status_code == 422

def test_prediction_empty_fields_input():
    """
    Includes None transaction date and distance to MRT.
    Is expected to fail.
    """
    empty_payload = {
        "transaction_date":None,
        "house_age": 12.0,
        "distance_to_MRT": None,
        "number_of_CS":0,
        "latitude": 24.96500,
        "longitude": 121.53700
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=empty_payload)
        assert response.status_code == 422

def test_prediction_incorrect_type_input():
    """
    Includes incorrect types for transaction date, number of Convenience Stores and longitude.
    Is expected to fail.
    """
    negative_payload = {
        "transaction_date":"2013-04",
        "house_age": 12.0,
        "distance_to_MRT": 500.0,
        "number_of_CS":4.5,
        "latitude": 24.96500,
        "longitude": '121.53700'
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=negative_payload)
        assert response.status_code == 422