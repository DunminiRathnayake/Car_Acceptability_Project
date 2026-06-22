import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.app import app

def test_prediction_api():
    client = TestClient(app)
    
    # Test case 1: Acceptable car configuration
    test_payload = {
        "buying": "vhigh",
        "maint": "high",
        "doors": "4",
        "persons": "more",
        "lug_boot": "med",
        "safety": "high"
    }
    
    print(f"Sending test payload: {test_payload}")
    response = client.post("/api/predict", json=test_payload)
    
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.json()}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data
    assert "confidence" in data
    assert "explanation" in data
    
    explanation_dict = data["explanation"]
    assert "summary" in explanation_dict
    assert "top_positive_features" in explanation_dict
    assert "top_negative_features" in explanation_dict
    assert "confidence_reason" in explanation_dict
    assert "decision_strength" in explanation_dict
    
    summary = explanation_dict["summary"].lower()
    assert "cost" in summary or "price" in summary or "buying" in summary or "maint" in summary
    
    # Test case 2: Unacceptable safety configuration
    safety_payload = {
        "buying": "low",
        "maint": "low",
        "doors": "4",
        "persons": "more",
        "lug_boot": "med",
        "safety": "low"
    }
    print(f"\nSending low safety payload: {safety_payload}")
    response = client.post("/api/predict", json=safety_payload)
    print(f"Response body: {response.json()}")
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "unacc"
    
    explanation_dict = data["explanation"]
    summary = explanation_dict["summary"].lower()
    assert "safety" in summary
    assert "low safety" in summary
    
    # Test case 3: Invalid category input validation
    invalid_payload = {
        "buying": "extra_high", # invalid option
        "maint": "low",
        "doors": "4",
        "persons": "more",
        "lug_boot": "med",
        "safety": "high"
    }
    print(f"\nSending invalid category payload: {invalid_payload}")
    response = client.post("/api/predict", json=invalid_payload)
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.json()}")
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "invalid value" in data["message"].lower()
    assert "valid options" in data["message"].lower()
    
    print("\nSUCCESS: Prediction API test, Explainable AI, and Input Validation tests passed successfully!")

if __name__ == "__main__":
    test_prediction_api()
