# Car Acceptability Predictor & Explainable AI (XAI) Dashboard

An end-to-end Machine Learning web application that predicts vehicle acceptability using physical attributes and provides real-time Explainable AI (XAI) insights. Built using **FastAPI** on the backend, **Vanilla HTML/CSS/JS** on the frontend, and a **Random Forest Classifier** trained on the UCI Car Evaluation dataset.

The system uses **SHAP (Shapley Additive exPlanations)** values to provide rigorous, mathematically correct explainability scores that detail exactly *why* the model made a specific evaluation.

---

## 🚀 Key Features

* **High-Accuracy Classification**: Utilizes a Random Forest Classifier trained to **97.4% test accuracy** across four categories: *Unacceptable*, *Acceptable*, *Good*, and *Very Good*.
* **Real Explainable AI (XAI)**: Transcends generic classification by computing SHAP values dynamically on each prediction request, delivering:
  * **Influence / Impact Scores**: Indicating the mathematical contribution strength of each feature.
  * **Interactive Feature Bar Charts**: Visually separating positive (supporting) and negative (opposing) factors.
  * **Semantic Explanations**: Custom, logic-driven natural language explanations generated from actual SHAP outputs.
* **Premium User Interface**: Implements a dark-mode glassmorphism dashboard, featuring:
  * Responsive layout matching desktop, tablet, and mobile displays.
  * Micro-animations including card floating transitions, status indicator pulses, and slide-in logo elements.
  * Prediction history drawer panel for comparative logging.
* **Model Comparison & Insights**: Includes pre-rendered accuracy benchmarks comparing 8 training runs and validation confusion matrices.

---

## 📂 Repository Structure

```text
├── backend/
│   ├── app.py                  # FastAPI server hosting static files and prediction endpoint
│   ├── model_handler.py        # Object wrapper for model inference and feature encoding
│   ├── explanation_engine.py   # XAI engine compiling dynamic explanations from SHAP contributions
│   ├── test_api.py             # Automated backend integration and explainability test suite
│   └── requirements.txt        # Backend dependencies
├── data/
│   ├── car.csv                 # Raw UCI Car Evaluation dataset
│   └── car_encoded.csv         # Processed label-encoded dataset
├── frontend/
│   ├── index.html              # Dashboard landing page
│   ├── css/
│   │   └── style.css           # Styling system (glassmorphism variables, media queries)
│   ├── js/
│   │   └── app.js              # State manager (form submissions, AJAX requests, history cache)
│   └── assets/
│       └── img/
│           ├── confusion_matrix.png # Random Forest validation confusion matrix
│           └── model_comparison.png # Accuracy comparison chart of all trained models
├── models/
│   ├── encoders.pkl            # Serialized label encoders
│   ├── metadata.pkl            # Model accuracy, size, and feature importances
│   └── random_forest.pkl       # Serialized Random Forest model artifact
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_decision_tree.ipynb
│   ├── 05_random_forest.ipynb
│   ├── 06_logistic_regression.ipynb
│   ├── 07_knn.ipynb
│   ├── 08_naive_bayes.ipynb
│   ├── 09_svm.ipynb
│   ├── 10_model_comparison.ipynb
│   ├── 11_save_model.ipynb
│   └── 12_test_model.ipynb     # Interactive model validation notebook
├── scripts/
│   ├── train.py                # Model training pipeline
│   └── generate_plots.py       # Plot generator script (saves to frontend assets)
├── start.bat                   # Batch script to launch FastAPI locally
└── .gitignore                  # Git exclusions for IDEs, environments, and logs
```

---

## ⚙️ Setup & Installation

### Prerequisites
* Python 3.9 or higher

### 1. Install Dependencies
Navigate to the root directory and install requirements:
```bash
pip install -r backend/requirements.txt
```

### 2. Run the Application
Start the uvicorn development server:
```bash
start.bat
```
Alternatively, launch manually:
```bash
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

---

## 📊 Machine Learning Pipeline

### Dataset
The model is trained on the **Car Evaluation Dataset** from the **UCI Machine Learning Repository**, which evaluates cars based on:
1. `buying` (Buying Price): `vhigh`, `high`, `med`, `low`
2. `maint` (Maintenance Cost): `vhigh`, `high`, `med`, `low`
3. `doors` (Doors Count): `2`, `3`, `4`, `5more`
4. `persons` (Persons Capacity): `2`, `4`, `more`
5. `lug_boot` (Luggage Boot Size): `small`, `med`, `big`
6. `safety` (Safety Rating): `low`, `med`, `high`

### Model Benchmarks
Multiple models were evaluated in Jupyter notebooks during development. The final **Random Forest Classifier** was selected for production deployment:

| Model | Accuracy (%) |
| :--- | :---: |
| **Random Forest (Selected)** | **97.40%** |
| Decision Tree (Max Depth = 10) | 95.38% |
| Support Vector Machine (SVM) | 91.33% |
| k-Nearest Neighbors (KNN) | 88.73% |
| Decision Tree (Max Depth = 5) | 86.99% |
| Decision Tree (Max Depth = 3) | 74.28% |
| Logistic Regression | 65.90% |
| Naive Bayes | 62.43% |

---

## 🧠 Explainable AI (XAI) Standards

Unlike simple predictions, this dashboard adheres to real Explainable AI standards:
1. **Mathematically Correct SHAP Values**: Computes exact Shapley contributions from the Random Forest model for the predicted class.
2. **No Misleading Probabilities**: Raw SHAP values represent margins of evidence in log-odds/probability space, and are properly displayed as **Influence Scores**, **Impact Scores**, or **Contribution Strengths**, avoiding mathematically incorrect percentage conversions.
3. **Contextual Natural Language Generation**: Synthesizes custom explanations directly from the highest-magnitude SHAP variables (e.g., explaining why low safety or high price dominated the evaluation).

---

## 🧪 Testing

To run the automated test suite verifying the API validation, SHAP scoring formats, and explanation rules:
```bash
python backend/test_api.py
```
