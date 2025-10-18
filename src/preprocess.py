# src/preprocess.py
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os

from src.utils import load_heart_disease_dataset

def preprocess_data(test_size=0.2, random_state=42):
    X, y = load_heart_disease_dataset()

    # Save feature order for the app
    feature_order = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Impute
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Save artifacts
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(imputer, os.path.join(model_dir, "imputer.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    with open(os.path.join(model_dir, "feature_order.json"), "w") as f:
        json.dump(feature_order, f)

    print("[preprocess] Saved imputer.pkl, scaler.pkl, feature_order.json to models/")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    preprocess_data()
