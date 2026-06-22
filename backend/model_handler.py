import os
import pickle
import pandas as pd
import numpy as np
import shap

# We map outcome abbreviations to friendly labels for UI rendering
CLASS_LABELS = {
    "acc": "Acceptable",
    "good": "Good",
    "unacc": "Unacceptable",
    "vgood": "Very Good"
}

class ModelHandler:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.explainer = None
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, "models")
        
        self.model_path = os.path.join(self.models_dir, "random_forest.pkl")
        self.encoders_path = os.path.join(self.models_dir, "encoders.pkl")
        self.metadata_path = os.path.join(self.models_dir, "metadata.pkl")
        
        self.load_artifacts()

    def load_artifacts(self):
        """Loads serialized model, encoders, metadata, and initializes the SHAP TreeExplainer."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.encoders_path):
            raise FileNotFoundError(f"Encoders file not found at {self.encoders_path}")
        
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
            
        with open(self.encoders_path, "rb") as f:
            self.encoders = pickle.load(f)
            
        self.metadata = None
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
                
        # Initialize SHAP TreeExplainer once during startup
        self.explainer = shap.TreeExplainer(self.model)

    def get_metadata(self) -> dict:
        """Returns the model training metadata."""
        return self.metadata

    def validate_inputs(self, inputs: dict):
        """Validates that all expected features are present and contain valid category values."""
        expected_features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
        for feature in expected_features:
            if feature not in inputs:
                raise ValueError(f"Missing required feature: '{feature}'")
            
            value = str(inputs[feature]).lower()
            
            # Retrieve valid classes from the fitted LabelEncoder
            encoder = self.encoders.get(feature)
            if not encoder:
                raise ValueError(f"No LabelEncoder found for feature: '{feature}'")
                
            valid_options = list(encoder.classes_)
            if value not in valid_options:
                raise ValueError(
                    f"Invalid value '{inputs[feature]}' for feature '{feature}'. "
                    f"Valid options are: {valid_options}"
                )
            
    def predict(self, inputs: dict) -> dict:
        """
        Takes raw string inputs, validates them using the loaded encoders,
        transforms features to numerical form, runs model prediction,
        computes SHAP values, and decodes the output class back to its original label.
        """
        # Validate inputs first
        self.validate_inputs(inputs)

        # Encode inputs dynamically using each feature's LabelEncoder
        encoded_dict = {}
        for feature in ["buying", "maint", "doors", "persons", "lug_boot", "safety"]:
            val = str(inputs[feature]).lower()
            encoder = self.encoders[feature]
            encoded_dict[feature] = int(encoder.transform([val])[0])

        # Convert to Pandas DataFrame to match original feature names
        df_input = pd.DataFrame([encoded_dict])

        # Run prediction
        pred_numeric = self.model.predict(df_input)[0]
        pred_proba = self.model.predict_proba(df_input)[0]

        # Decode target class prediction using the 'class' LabelEncoder
        pred_class = self.encoders["class"].inverse_transform([pred_numeric])[0]
        pred_label = CLASS_LABELS.get(pred_class, pred_class.capitalize())

        # Calculate confidence using probability list matching pred_numeric
        confidence = float(pred_proba[int(pred_numeric)]) * 100
        
        # Calculate second best probability for decision strength margin
        sorted_proba = np.sort(pred_proba)
        second_confidence = float(sorted_proba[-2]) * 100 if len(sorted_proba) > 1 else 0.0

        # Compute SHAP values for the single input sample
        shap_vals = self.explainer.shap_values(df_input)
        class_shap = shap_vals[0, :, int(pred_numeric)]

        # Get feature names dynamically from metadata, fallback to default ordering
        feature_names = self.metadata.get("feature_names") if self.metadata else ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

        # Scale SHAP values into "influence scores"
        contributions = []
        for i, feat in enumerate(feature_names):
            influence = float(class_shap[i]) * 100
            contributions.append({"feature": feat, "influence": round(influence, 2)})

        return {
            "prediction": pred_class,
            "prediction_label": pred_label,
            "confidence": round(confidence, 2),
            "second_confidence": round(second_confidence, 2),
            "contributions": contributions
        }
