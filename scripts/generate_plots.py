import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def generate_plots():
    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    data_path = os.path.join(base_dir, "data", "car.csv")
    models_dir = os.path.join(base_dir, "models")
    rf_path = os.path.join(models_dir, "random_forest.pkl")
    encoders_path = os.path.join(models_dir, "encoders.pkl")
    
    # 1. Generate model accuracy comparison plot
    models = [
        'DT-3', 'DT-5', 'DT-10', 'Random Forest', 
        'Logistic Regression', 'KNN', 'Naive Bayes', 'SVM'
    ]
    accuracies = [
        74.28, 86.99, 95.38, 97.40, 
        65.90, 88.73, 62.43, 91.33
    ]
    
    plt.figure(figsize=(10, 5.5))
    sns.set_style("darkgrid")
    
    # Color highlight the final selected model (Random Forest) in emerald green, others in blue
    colors = ['#3b82f6', '#3b82f6', '#3b82f6', '#10b981', '#3b82f6', '#3b82f6', '#3b82f6', '#3b82f6']
    bars = plt.bar(models, accuracies, color=colors, width=0.6)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(0, 110)
    plt.xticks(rotation=30, fontsize=10)
    plt.tight_layout()
    
    # Save model_comparison.png in frontend assets
    mc_paths = [
        os.path.join(base_dir, "frontend", "assets", "img", "model_comparison.png")
    ]
    
    for p in mc_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=150)
        print(f"Saved accuracy comparison chart to {p}")
        
    plt.close()
    
    # 2. Generate Random Forest Confusion Matrix Plot
    print("Loading model and data for evaluation metrics...")
    if not os.path.exists(rf_path) or not os.path.exists(encoders_path) or not os.path.exists(data_path):
        print("Required artifacts missing. Skipping evaluation metrics plot.")
        return
        
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
        
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv(data_path, names=columns)
    
    df_encoded = df.copy()
    for col in df.columns:
        df_encoded[col] = encoders[col].transform(df[col])
        
    X = df_encoded.drop("class", axis=1)
    y = df_encoded["class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    y_pred = rf.predict(X_test)
    
    # Decode classes for display labels
    class_encoder = encoders["class"]
    class_names = [c.upper() for c in class_encoder.classes_]
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, 
                cbar=False, annot_kws={"size": 12, "weight": "bold"})
    
    plt.title("Random Forest Confusion Matrix (Validation)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=10)
    plt.ylabel("True Class", fontsize=11, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    out_paths = [
        os.path.join(base_dir, "frontend", "assets", "img", "confusion_matrix.png")
    ]
    
    for p in out_paths:
        plt.savefig(p, dpi=150)
        print(f"Saved evaluation metrics confusion matrix to {p}")
        
    plt.close()
    print("All plots generated successfully!")

if __name__ == "__main__":
    generate_plots()
