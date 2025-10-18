import joblib
import pandas as pd
import os
from src.utils import load_heart_disease_dataset
from sklearn.model_selection import train_test_split
from mapie.classification import MapieClassifier


def run_conformal_prediction(model , X):
    # Load dataset
    X, y = load_heart_disease_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load saved preprocessing + model
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imputer = joblib.load(os.path.join(BASE_DIR, "models", "imputer.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    model = joblib.load(os.path.join(BASE_DIR, "models", "random_forest.pkl"))

    # Apply preprocessing
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    X_train_scaled = scaler.transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Wrap trained model with MAPIE (prefit mode since model is already trained)
    mapie = MapieClassifier(estimator=model, method="score", cv="prefit")
    mapie.fit(X_train_scaled, y_train)

    # Predict with 90% confidence interval
    y_pred, y_pis = mapie.predict(X_test_scaled, alpha=0.1)

    # Display results
    print("Predicted labels:", y_pred[:5])
    print("Prediction intervals for first 5 samples:\n", y_pis[:5])

    return y_pred, y_pis


if __name__ == "__main__":
    run_conformal_prediction()
