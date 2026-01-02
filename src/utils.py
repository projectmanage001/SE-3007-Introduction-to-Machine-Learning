# Utility functions - directory creation, validation, etc.

import os
import sys
from pathlib import Path


def ensure_dir(directory):
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def validate_dataset(data_path):
    # Check if dataset exists, if not show error message and exit
    if not os.path.exists(data_path):
        print("\n" + "="*80)
        print("ERROR: Dataset not found!")
        print("="*80)
        print(f"\nExpected location: {data_path}")
        print("\nPlease ensure the 'creditcard.csv' dataset is placed in the data/ directory.")
        print("\nDataset Information:")
        print("  - Source: Kaggle Credit Card Fraud Detection Dataset")
        print("  - URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("  - File: creditcard.csv")
        print("  - Size: ~150 MB")
        print("\nInstructions:")
        print("  1. Download the dataset from Kaggle")
        print("  2. Place 'creditcard.csv' in the 'data/' directory")
        print("  3. Run this script again")
        print("\nFor more information, see: data/README.md")
        print("="*80 + "\n")
        sys.exit(1)
    
    return True


def get_model_paths(models_dir):
    # Returns dictionary with paths to model files
    return {
        'model': os.path.join(models_dir, 'final_model_pipeline.joblib'),
        'scaler': os.path.join(models_dir, 'scaler.joblib'),
        'threshold': os.path.join(models_dir, 'best_threshold.txt')
    }


def check_model_exists(models_dir):
    # Check if model file exists
    model_path = os.path.join(models_dir, 'final_model_pipeline.joblib')
    if not os.path.exists(model_path):
        print("\n" + "="*80)
        print("ERROR: Trained model not found!")
        print("="*80)
        print(f"\nExpected location: {model_path}")
        print("\nPlease train the model first by running:")
        print("  python src/train.py")
        print("="*80 + "\n")
        return False
    return True


def print_section(title):
    # Print formatted section header
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)
