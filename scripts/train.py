import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def run_training_pipeline():
    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    data_path = os.path.join(base_dir, "data", "car.csv")
    encoded_data_path = os.path.join(base_dir, "data", "car_encoded.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Loading raw data from {data_path}...")
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv(data_path, names=columns)
    
    print("Fitting LabelEncoders and preprocessing data...")
    encoders = {}
    df_encoded = df.copy()
    for col in df.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  - Fitted encoder for '{col}': {list(le.classes_)}")
        
    # Save encoders.pkl
    encoders_path = os.path.join(models_dir, "encoders.pkl")
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)
    print(f"Successfully saved encoders to {encoders_path}")
    
    # Save car_encoded.csv
    df_encoded.to_csv(encoded_data_path, index=False)
    print(f"Successfully saved preprocessed encoded data to {encoded_data_path}")
    
    # Split features and target
    X = df_encoded.drop("class", axis=1)
    y = df_encoded["class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training RandomForestClassifier...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate model
    accuracy = rf.score(X_test, y_test)
    print(f"Model test accuracy: {accuracy * 100:.2f}%")
    
    # Save random_forest.pkl
    model_path = os.path.join(models_dir, "random_forest.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"Successfully saved model to {model_path}")
    
    # Save metadata.pkl
    metadata = {
        "accuracy": accuracy,
        "dataset_size": len(df),
        "feature_names": list(X.columns),
        "feature_importances": dict(zip(X.columns, rf.feature_importances_))
    }
    metadata_path = os.path.join(models_dir, "metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Successfully saved metadata to {metadata_path}")
    
    print("\nSUCCESS: Training pipeline execution finished successfully!")

if __name__ == "__main__":
    run_training_pipeline()
