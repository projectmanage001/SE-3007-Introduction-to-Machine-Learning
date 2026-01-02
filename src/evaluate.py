# Evaluation script - evaluates the trained model on test set
# Computes metrics and saves to results/metrics.json

import os
import sys
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, 
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.utils import ensure_dir, check_model_exists, print_section


def find_optimal_threshold(y_true, y_probs):
    """
    Find optimal classification threshold based on F1 score.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        
    Returns:
        float: Optimal threshold
        int: Index of optimal threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)
    
    best_idx = np.argmax(f1_scores)
    
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = thresholds[-1]
    
    return best_threshold, best_idx


def save_precision_recall_curve(y_true, y_probs, best_threshold, best_idx, output_dir):
    """Save Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    
    plt.figure(figsize=FIGSIZE_MEDIUM)
    plt.plot(recall, precision, marker='.', linewidth=2, label='XGBoost')
    plt.scatter(recall[best_idx], precision[best_idx], 
                marker='o', color='red', s=200, zorder=5,
                label=f'Optimal Threshold: {best_threshold:.3f}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'metric_curve.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return save_path


def save_roc_curve(y_true, y_probs, output_dir):
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=FIGSIZE_MEDIUM)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return save_path, roc_auc


def save_confusion_matrix(y_true, y_pred, threshold, output_dir):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=FIGSIZE_SMALL)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title(f'Confusion Matrix (Threshold: {threshold:.3f})', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return save_path, cm


def main():
    print_section("CREDIT CARD FRAUD DETECTION - MODEL EVALUATION")
    
    # Check if model exists
    print("\n1. Loading Model...")
    if not check_model_exists(MODELS_DIR):
        sys.exit(1)
    
    model_path = os.path.join(MODELS_DIR, 'final_model_pipeline.joblib')
    model = joblib.load(model_path)
    print(f"  ✓ Model loaded: {model_path}")
    
    # Load test data
    test_data_path = os.path.join(MODELS_DIR, 'test_data.joblib')
    if not os.path.exists(test_data_path):
        print(f"\n  ERROR: Test data not found at {test_data_path}")
        print("  Please run training first: python src/train.py")
        sys.exit(1)
    
    test_data = joblib.load(test_data_path)
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    print(f"  ✓ Test data loaded: {len(X_test):,} samples")
    
    # Create output directories
    print("\n2. Setting Up Directories...")
    ensure_dir(PLOTS_DIR)
    ensure_dir(RESULTS_DIR)
    
    # Get predictions
    print("\n3. Generating Predictions...")
    y_probs = model.predict_proba(X_test)[:, 1]
    print(f"  ✓ Predictions generated")
    
    # Find optimal threshold
    print("\n4. Finding Optimal Threshold...")
    best_threshold, best_idx = find_optimal_threshold(y_test, y_probs)
    print(f"  ✓ Optimal threshold: {best_threshold:.4f}")
    
    # Apply threshold
    y_pred = (y_probs >= best_threshold).astype(int)
    
    # Calculate metrics
    print("\n5. Computing Metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"  ✓ Accuracy:  {accuracy:.4f}")
    print(f"  ✓ Precision: {precision:.4f}")
    print(f"  ✓ Recall:    {recall:.4f}")
    print(f"  ✓ F1 Score:  {f1:.4f}")
    print(f"  ✓ AUC-ROC:   {roc_auc:.4f}")
    
    # Save metrics to JSON
    print("\n6. Saving Metrics...")
    metrics = {
        "test_accuracy": float(accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "auc_roc": float(roc_auc),
        "optimal_threshold": float(best_threshold),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "test_set_size": len(y_test),
        "fraud_cases": int(y_test.sum()),
        "normal_cases": int(len(y_test) - y_test.sum())
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  ✓ Metrics saved: {METRICS_PATH}")
    
    # Generate and save plots
    print("\n7. Generating Evaluation Plots...")
    
    # Precision-Recall curve
    pr_path = save_precision_recall_curve(y_test, y_probs, best_threshold, best_idx, PLOTS_DIR)
    print(f"  ✓ Precision-Recall curve: {pr_path}")
    
    # ROC curve
    roc_path, _ = save_roc_curve(y_test, y_probs, PLOTS_DIR)
    print(f"  ✓ ROC curve: {roc_path}")
    
    # Confusion matrix
    cm_path, _ = save_confusion_matrix(y_test, y_pred, best_threshold, PLOTS_DIR)
    print(f"  ✓ Confusion matrix: {cm_path}")
    
    # Save threshold
    threshold_path = os.path.join(MODELS_DIR, 'best_threshold.txt')
    with open(threshold_path, 'w') as f:
        f.write(str(best_threshold))
    print(f"  ✓ Threshold saved: {threshold_path}")
    
    # Print classification report
    print("\n8. Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, 
                                target_names=['Normal', 'Fraud'],
                                digits=4))
    
    print_section("EVALUATION COMPLETED SUCCESSFULLY!")
    print("\nNext Steps:")
    print("  - Run inference: python src/inference.py")
    print(f"  - View metrics: cat {METRICS_PATH}")
    print(f"  - View plots in: {PLOTS_DIR}")
    print("\n")


if __name__ == "__main__":
    main()
