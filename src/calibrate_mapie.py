# src/calibrate_mapie.py
import joblib
import os
import warnings
import pandas as pd
from mapie.classification import MapieClassifier
from src.preprocess import preprocess_data # We use this to get the calibration set
from src.utils import load_heart_disease_dataset

# Suppress warnings
warnings.filterwarnings("ignore")

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# List of your trained base models
# Make sure you have trained and saved all these models
model_names = [
    "random_forest",
    "logistic_regression",
    "svm",
    "xgboost"
]

print("Loading calibration data...")
# Use the 'test' set from your preprocessing script as the calibration set
_, X_calib, _, y_calib = preprocess_data()

# --- TARGET VARIABLE FIX ---
# The UCI dataset target 'num' has values 0, 1, 2, 3, 4.
# For binary classification (disease vs. no disease), we must convert this.
# 0 = No Disease, 1,2,3,4 = Disease
print("Binarizing target variable for calibration...")
y_calib = (y_calib > 0).astype(int)
# ---------------------------

print(f"Loaded calibration set with {X_calib.shape[0]} samples.")

for name in model_names:
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    mapie_model_path = os.path.join(MODEL_DIR, f"mapie_{name}.pkl")
    
    if not os.path.exists(model_path):
        print(f"WARNING: Model {model_path} not found. Skipping.")
        continue

    print(f"Calibrating {name}...")
    
    # 1. Load the base trained model
    base_model = joblib.load(model_path)
    
    # 2. Wrap it with MapieClassifier
    mapie = MapieClassifier(estimator=base_model, method="score", cv="prefit")
    
    # 3. Fit MAPIE on the calibration set
    mapie.fit(X_calib, y_calib)
    
    # 4. Save the fitted, calibrated MAPIE model
    joblib.dump(mapie, mapie_model_path)
    print(f"Saved calibrated model to {mapie_model_path}")

print("\nAll models calibrated and saved.")
print("You can now run the Streamlit app.")