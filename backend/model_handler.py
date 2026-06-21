import os
import pickle
import pandas as pd
import numpy as np

# Define categorical mappings based on alphabetical sorting
MAPPINGS = {
    "buying": {"high": 0, "low": 1, "med": 2, "vhigh": 3},
    "maint": {"high": 0, "low": 1, "med": 2, "vhigh": 3},
    "doors": {"2": 0, "3": 1, "4": 2, "5more": 3},
    "persons": {"2": 0, "4": 1, "more": 2},
    "lug_boot": {"big": 0, "med": 1, "small": 2},
    "safety": {"high": 0, "low": 1, "med": 2}
}

CLASS_MAPPINGS = {
    0: "acc",
    1: "good",
    2: "unacc",
    3: "vgood"
}

CLASS_LABELS = {
    "acc": "Acceptable",
    "good": "Good",
    "unacc": "Unacceptable",
    "vgood": "Very Good"
}

class ModelHandler:
    def __init__(self):
        self.model = None
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(self.base_dir, "random_forest.pkl")
        self.load_model()

    def load_model(self):
        """Loads the serialized model pickle file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

    def validate_inputs(self, inputs: dict):
        """Validates that all expected features are present and have valid categories."""
        expected_features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
        for feature in expected_features:
            if feature not in inputs:
                raise ValueError(f"Missing required feature: '{feature}'")
            
            value = str(inputs[feature]).lower()
            if value not in MAPPINGS[feature]:
                valid_options = list(MAPPINGS[feature].keys())
                raise ValueError(
                    f"Invalid value '{inputs[feature]}' for feature '{feature}'. "
                    f"Valid options are: {valid_options}"
                )
            
    def predict(self, inputs: dict) -> dict:
        """
        Takes raw string inputs, validates, encodes them, runs prediction,
        and decodes the output back to class label and confidence score.
        """
        # Validate inputs first
        self.validate_inputs(inputs)

        # Encode inputs
        encoded_dict = {}
        for feature, mapping in MAPPINGS.items():
            val = str(inputs[feature]).lower()
            encoded_dict[feature] = mapping[val]

        # Convert to Pandas DataFrame to match original feature names
        df_input = pd.DataFrame([encoded_dict])

        # Run prediction
        pred_numeric = self.model.predict(df_input)[0]
        pred_proba = self.model.predict_proba(df_input)[0]

        # Get label and description
        pred_class = CLASS_MAPPINGS.get(int(pred_numeric), "unknown")
        pred_label = CLASS_LABELS.get(pred_class, "Unknown")

        # Get probability/confidence of the predicted class
        # Random Forest model output matches the sorted classes: acc (0), good (1), unacc (2), vgood (3)
        confidence = float(pred_proba[int(pred_numeric)]) * 100

        return {
            "prediction": pred_class,
            "prediction_label": pred_label,
            "confidence": round(confidence, 2)
        }
