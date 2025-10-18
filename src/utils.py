import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

# Constants
DATA_DIR = r"C:\Users\ishaa\Desktop\Conformal Health Risk\data"
HEART_DATA_PATH = os.path.join(DATA_DIR, "heart.csv")

def load_heart_disease_dataset():
    """
    Load the Heart Disease dataset from local CSV if available,
    otherwise fetch from UCI repository and save locally.
    
    Returns:
        tuple: (X_features, y_target) pandas DataFrames
    """
    # Create directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check if local file exists
    if os.path.exists(HEART_DATA_PATH):
        print(f"Loading dataset from {HEART_DATA_PATH}")
        df = pd.read_csv(HEART_DATA_PATH)
    else:
        print("Downloading dataset from UCI repository...")
        heart_disease = fetch_ucirepo(id=45)  # ID 45 is Heart Disease dataset
        
        # Combine features and target
        df = pd.concat([
            heart_disease.data.features,
            heart_disease.data.targets.rename(columns={'num': 'target'})
        ], axis=1)
        
        # Save to CSV
        df.to_csv(HEART_DATA_PATH, index=False)
        print(f"Dataset saved to {HEART_DATA_PATH}")
    
    # Split into features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    return X, y

def get_data_path(filename):
    """
    Get absolute path for a file in the data directory
    
    Args:
        filename (str): Name of the file
        
    Returns:
        str: Full absolute path to the file
    """
    return os.path.join(DATA_DIR, filename)

# Example usage
if __name__ == "__main__":
    # Test the dataset loading
    X, y = load_heart_disease_dataset()
    
    print("\nFeature columns:")
    print(X.columns.tolist())
    
    print("\nFirst 5 rows:")
    print(X.head())
    
    print("\nTarget distribution:")
    print(y.value_counts())