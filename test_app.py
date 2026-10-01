# test_app.py
import pytest
from app import app  # Import the Flask app from app.py

@pytest.fixture
def client():
    app.config["TESTING"] = True  # Enable testing mode
    with app.test_client() as client:  # Create a test client for simulating requests
        yield client

def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Iris Flower Classifier" in response.data  # Check for HTML title



def test_predict_endpoint_valid_input(client):
    response = client.post(
        "/predict",  # Simulate a POST request to the predict route
        json={"features": [5.1, 3.5, 1.4, 0.2]}  # Valid input for setosa
    )
    assert response.status_code == 200  # Check if status is OK
    assert response.json == {"prediction": "setosa"}  # Check predicted class

def test_predict_endpoint_invalid_input(client):
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4]}  # Invalid: only 3 features
    )
    assert response.status_code == 400  # Check if status is Bad Request
    assert "error" in response.json  # Check for error key in response