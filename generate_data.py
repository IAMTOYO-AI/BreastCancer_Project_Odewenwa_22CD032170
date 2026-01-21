"""
Generate breast cancer dataset from sklearn's built-in dataset
This is the Wisconsin Diagnostic Breast Cancer (WDBC) dataset
"""
from sklearn.datasets import load_breast_cancer
import pandas as pd
import os


def generate_cancer_data():
    """Generate breast_cancer.csv from sklearn dataset"""

    print("Generating breast cancer dataset...")

    # Load the dataset from sklearn
    data = load_breast_cancer()

    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target  # 0 = malignant, 1 = benign

    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/breast_cancer.csv', index=False)

    print(f"✓ Created data/breast_cancer.csv with {len(df)} samples")
    print(f"  - Malignant (cancerous): {(df['diagnosis'] == 0).sum()}")
    print(f"  - Benign (non-cancerous): {(df['diagnosis'] == 1).sum()}")
    print(f"  - Features: {len(data.feature_names)}")


if __name__ == "__main__":
    generate_cancer_data()