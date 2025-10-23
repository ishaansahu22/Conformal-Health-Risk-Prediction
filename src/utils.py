import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

# Constants
DATA_DIR = r"C:\Users\ishaa\Desktop\Conformal Health Risk\data"
HEART_DATA_PATH = os.path.join(DATA_DIR, "heart.csv")

def load_heart_disease_dataset():
    """
    Load and preprocess the Heart Disease dataset.
    Converts target values > 0 into 1 for binary classification.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load or fetch dataset
    if os.path.exists(HEART_DATA_PATH):
        print(f"Loading dataset from {HEART_DATA_PATH}")
        df = pd.read_csv(HEART_DATA_PATH)
    else:
        print("Downloading dataset from UCI repository...")
        heart_disease = fetch_ucirepo(id=45)
        df = pd.concat([
            heart_disease.data.features,
            heart_disease.data.targets.rename(columns={'num': 'target'})
        ], axis=1)
        df.to_csv(HEART_DATA_PATH, index=False)
        print(f"Dataset saved to {HEART_DATA_PATH}")

    # ✅ FIX: Convert multi-class target (0–4) → binary (0 or 1)
    if 'target' in df.columns:
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Split into features & target
    X = df.drop(columns=['target'])
    y = df['target']

    print("\n✅ Dataset loaded successfully.")
    print(f"Target value counts:\n{y.value_counts()}")
    print(f"Shape: {df.shape}")

    return X, y


def get_data_path(filename):
    """Return absolute path to a file in the data folder."""
    return os.path.join(DATA_DIR, filename)


# Example test
if __name__ == "__main__":
    X, y = load_heart_disease_dataset()
    print("\nFeature Columns:\n", X.columns.tolist())
    print("\nSample Data:\n", X.head())
