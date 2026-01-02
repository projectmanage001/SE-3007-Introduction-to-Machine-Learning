# Training script for Credit Card Fraud Detection
# This script trains the XGBoost model with SMOTE and saves it to results/models/

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.utils import ensure_dir, validate_dataset, print_section
from src.preprocessing import load_and_clean_data, perform_eda_and_save_plots, feature_engineering, scale_data
from src.training import train_model_with_tuning


def save_training_plots(search, output_dir):
    # Save training convergence plot from hyperparameter search results
    ensure_dir(output_dir)
    
    # Extract CV results
    cv_results = search.cv_results_
    
    # Plot hyperparameter search results (proxy for training convergence)
    plt.figure(figsize=FIGSIZE_MEDIUM)
    
    mean_scores = cv_results['mean_test_score']
    std_scores = cv_results['std_test_score']
    
    iterations = np.arange(1, len(mean_scores) + 1)
    
    plt.plot(iterations, mean_scores, 'o-', linewidth=2, markersize=8, label='Mean CV F1 Score')
    plt.fill_between(iterations, 
                     mean_scores - std_scores, 
                     mean_scores + std_scores, 
                     alpha=0.3, label='±1 std dev')
    
    # Mark best iteration
    best_idx = np.argmax(mean_scores)
    plt.scatter(iterations[best_idx], mean_scores[best_idx], 
                color='red', s=200, marker='*', zorder=5, 
                label=f'Best Score: {mean_scores[best_idx]:.4f}')
    
    plt.xlabel('Hyperparameter Configuration', fontsize=12)
    plt.ylabel('F1 Score (Cross-Validation)', fontsize=12)
    plt.title('Training Convergence - Hyperparameter Search Results', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    loss_curve_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_curve_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved training convergence plot: {loss_curve_path}")


def main():
    print_section("CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")
    
    # Validate dataset exists
    print("\n1. Validating Dataset...")
    validate_dataset(DATA_PATH)
    print(f"  ✓ Dataset found: {DATA_PATH}")
    
    # Create output directories
    print("\n2. Setting Up Directories...")
    ensure_dir(MODELS_DIR)
    ensure_dir(PLOTS_DIR)
    ensure_dir(OLD_PLOTS_EDA_DIR)  # For EDA plots
    print("  ✓ Directories ready")
    
    # Load data
    print("\n3. Loading Data...")
    df = load_and_clean_data(DATA_PATH)
    print(f"  ✓ Loaded {len(df):,} transactions")
    print(f"  ✓ Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    
    # Perform EDA
    print("\n4. Performing Exploratory Data Analysis...")
    perform_eda_and_save_plots(df, OLD_PLOTS_EDA_DIR)
    print(f"  ✓ EDA plots saved to {OLD_PLOTS_EDA_DIR}")
    
    # Feature Engineering
    print("\n5. Feature Engineering...")
    df = feature_engineering(df)
    print(f"  ✓ Engineered features added")
    
    # Prepare features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Train-test split
    print("\n6. Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    print(f"  ✓ Training set: {len(X_train):,} samples")
    print(f"  ✓ Test set: {len(X_test):,} samples")
    
    # Scale data
    print("\n7. Scaling Features...")
    X_train, scaler = scale_data(X_train)
    X_test['Amount'] = scaler.transform(X_test['Amount'].values.reshape(-1, 1))
    print("  ✓ Features scaled")
    
    # Train model
    print("\n8. Training Model with SMOTE and Hyperparameter Tuning...")
    print("  (This may take several minutes...)")
    final_pipeline, search = train_model_with_tuning(X_train, y_train, return_search=True)
    print(f"\n  ✓ Best F1 Score (CV): {search.best_score_:.4f}")
    print(f"  ✓ Best Parameters:")
    for param, value in search.best_params_.items():
        print(f"      {param}: {value}")
    
    # Save model artifacts
    print("\n9. Saving Model Checkpoints...")
    model_path = os.path.join(MODELS_DIR, 'final_model_pipeline.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    
    joblib.dump(final_pipeline, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"  ✓ Model saved: {model_path}")
    print(f"  ✓ Scaler saved: {scaler_path}")
    
    # Save split data for evaluation
    test_data_path = os.path.join(MODELS_DIR, 'test_data.joblib')
    joblib.dump({
        'X_test': X_test,
        'y_test': y_test
    }, test_data_path)
    print(f"  ✓ Test data saved: {test_data_path}")
    
    # Save training plots
    print("\n10. Generating Training Plots...")
    save_training_plots(search, PLOTS_DIR)
    
    print_section("TRAINING COMPLETED SUCCESSFULLY!")
    print("\nNext Steps:")
    print("  1. Run evaluation: python src/evaluate.py")
    print("  2. Generate test examples: python src/inference.py")
    print("\n")


if __name__ == "__main__":
    main()
