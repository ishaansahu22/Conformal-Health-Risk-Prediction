import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

# -------------------------------------
# Constants
# -------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HEART_DATA_PATH = os.path.join(DATA_DIR, "heart.csv")


def load_heart_disease_dataset():
    """
    Load and preprocess the Heart Disease dataset.
    
    - Downloads dataset from UCI if not found locally.
    - Converts multi-class target (0–4) into binary:
        0 → No heart disease
        1 → Heart disease present (any severity)
    - Returns features (X) and target (y) as DataFrames.
    """
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # -------------------------------------
    # Step 1: Load dataset (local or fetch)
    # -------------------------------------
    if os.path.exists(HEART_DATA_PATH):
        print(f"[INFO] Loading dataset from {HEART_DATA_PATH}")
        df = pd.read_csv(HEART_DATA_PATH)
    else:
        print("[INFO] Downloading dataset from UCI repository...")
        heart_disease = fetch_ucirepo(id=45)  # UCI Heart Disease dataset (ID 45)
        df = pd.concat([
            heart_disease.data.features,
            heart_disease.data.targets.rename(columns={'num': 'target'})
        ], axis=1)

        df.to_csv(HEART_DATA_PATH, index=False)
        print(f"[INFO] Dataset saved locally to: {HEART_DATA_PATH}")

    # -------------------------------------
    # Step 2: Convert target → binary
    # -------------------------------------
    if 'target' not in df.columns:
        raise KeyError("❌ 'target' column not found in dataset!")

    # Convert target to binary (0 = healthy, 1 = any heart disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # -------------------------------------
    # Step 3: Split into features and target
    # -------------------------------------
    X = df.drop(columns=['target'])
    y = df['target']

    # -------------------------------------
    # Step 4: Console info (for debugging)
    # -------------------------------------
    print("\n✅ Dataset loaded successfully!")
    print(f"📊 Shape: {df.shape}")
    print(f"🎯 Target distribution:\n{y.value_counts()}")
    print(f"🧩 Feature columns: {list(X.columns)}")

    return X, y


def get_data_path(filename):
    """Return absolute path to a file in the data directory."""
    return os.path.join(DATA_DIR, filename)


# -------------------------------------
# Run test if executed directly
# -------------------------------------
if __name__ == "__main__":
    X, y = load_heart_disease_dataset()
    print("\n🔍 Preview of data:")
    print(X.head())
    print(y.head())
    