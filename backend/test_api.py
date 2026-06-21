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
    print("\nSUCCESS: Prediction API test passed successfully!")

if __name__ == "__main__":
    test_prediction_api()
