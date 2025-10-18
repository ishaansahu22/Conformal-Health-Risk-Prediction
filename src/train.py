# src/train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib, os

from src.preprocess import preprocess_data

def train_model():
    X_train, X_test, y_train, y_test = preprocess_data()
    model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    print(f"[train] Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\n[train] Classification report:\n", classification_report(y_test, preds))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "random_forest.pkl")
    joblib.dump(model, path)
    print(f"[train] Saved model to {path}")

if __name__ == "__main__":
    train_model()
