# Inference script - generates example predictions on test samples
# Creates visualization cards and saves them to results/test_examples/

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.utils import ensure_dir, check_model_exists, print_section


def create_prediction_card(sample_features, actual_label, pred_prob, threshold, 
                           feature_names, sample_idx, output_path):
    # Create a visual prediction card for one transaction
    pred_label = 1 if pred_prob >= threshold else 0
    is_correct = (pred_label == actual_label)
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Title
    result_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    result_color = 'green' if is_correct else 'red'
    
    fig.suptitle(f'Transaction Prediction #{sample_idx}\n{result_text}', 
                 fontsize=16, fontweight='bold', color=result_color)
    
    # Prediction Summary (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    summary_text = f"""
    Actual Class: {'FRAUD' if actual_label == 1 else 'NORMAL'}
    Predicted Class: {'FRAUD' if pred_label == 1 else 'NORMAL'}
    
    Fraud Probability: {pred_prob:.4f}
    Threshold: {threshold:.4f}
    Confidence: {abs(pred_prob - threshold):.4f}
    """
    
    ax1.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Probability Bar (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    colors = ['green' if pred_prob < threshold else 'red']
    bars = ax2.barh(['Fraud Probability'], [pred_prob], color=colors, alpha=0.7)
    ax2.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    ax2.set_xlim([0, 1])
    ax2.set_xlabel('Probability', fontsize=11)
    ax2.set_title('Prediction Confidence', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    # Add probability value on bar
    for bar in bars:
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Top Features (middle, spans both columns)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Get non-zero features for visualization (Amount and engineered features)
    feature_dict = dict(zip(feature_names, sample_features))
    
    # Select interesting features to display
    display_features = []
    for fname in ['Amount', 'Amount_Log', 'Hour', 'Is_Night']:
        if fname in feature_dict:
            display_features.append((fname, feature_dict[fname]))
    
    # Add a few V features with highest absolute values
    v_features = [(k, v) for k, v in feature_dict.items() if k.startswith('V')]
    v_features_sorted = sorted(v_features, key=lambda x: abs(x[1]), reverse=True)[:6]
    display_features.extend(v_features_sorted)
    
    if display_features:
        feat_names, feat_values = zip(*display_features)
        y_pos = np.arange(len(feat_names))
        
        colors_bar = ['red' if v > 0 else 'blue' for v in feat_values]
        ax3.barh(y_pos, feat_values, color=colors_bar, alpha=0.6)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(feat_names, fontsize=9)
        ax3.set_xlabel('Normalized Feature Value', fontsize=11)
        ax3.set_title('Key Features', fontsize=12, fontweight='bold')
        ax3.axvline(0, color='black', linewidth=0.8)
        ax3.grid(axis='x', alpha=0.3)
    
    # Feature Statistics (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    stats_text = f"""
    Total Features: {len(sample_features)}
    Non-zero Features: {np.count_nonzero(sample_features)}
    Mean Value: {np.mean(sample_features):.4f}
    Std Dev: {np.std(sample_features):.4f}
    Max Value: {np.max(sample_features):.4f}
    Min Value: {np.min(sample_features):.4f}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Decision Explanation (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    if pred_label == 1:
        if actual_label == 1:
            explanation = "✓ True Positive\nCorrectly identified fraud"
        else:
            explanation = "✗ False Positive\nIncorrectly flagged as fraud"
    else:
        if actual_label == 0:
            explanation = "✓ True Negative\nCorrectly identified as normal"
        else:
            explanation = "✗ False Negative\nMissed fraud case"
    
    ax5.text(0.5, 0.5, explanation, fontsize=13, verticalalignment='center',
             horizontalalignment='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow' if not is_correct else 'lightgreen', alpha=0.5))
    
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def main():
    print_section("CREDIT CARD FRAUD DETECTION - INFERENCE EXAMPLES")
    
    # Check if model exists
    print("\n1. Loading Model...")
    if not check_model_exists(MODELS_DIR):
        sys.exit(1)
    
    model_path = os.path.join(MODELS_DIR, 'final_model_pipeline.joblib')
    model = joblib.load(model_path)
    print(f"  ✓ Model loaded")
    
    # Load threshold
    threshold_path = os.path.join(MODELS_DIR, 'best_threshold.txt')
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
        print(f"  ✓ Optimal threshold: {threshold:.4f}")
    else:
        threshold = 0.5
        print(f"  ⚠ Using default threshold: {threshold}")
    
    # Load test data
    print("\n2. Loading Test Data...")
    test_data_path = os.path.join(MODELS_DIR, 'test_data.joblib')
    if not os.path.exists(test_data_path):
        print(f"\n  ERROR: Test data not found at {test_data_path}")
        print("  Please run training first: python src/train.py")
        sys.exit(1)
    
    test_data = joblib.load(test_data_path)
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    print(f"  ✓ Test data loaded: {len(X_test):,} samples")
    
    # Create output directory
    print("\n3. Setting Up Output Directory...")
    ensure_dir(TEST_EXAMPLES_DIR)
    
    # Get predictions for all test samples
    print("\n4. Generating Predictions...")
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    print(f"  ✓ Predictions generated")
    
    # Select diverse examples
    print("\n5. Selecting Example Transactions...")
    
    # Get indices for different categories
    fraud_indices = np.where(y_test == 1)[0]
    normal_indices = np.where(y_test == 0)[0]
    
    # Select fraud examples with various probabilities
    fraud_probs = y_probs[fraud_indices]
    fraud_sorted_idx = np.argsort(fraud_probs)[::-1]  # Sort by probability descending
    
    # Select: high confidence, medium confidence, low confidence (if misclassified)
    n_fraud = min(N_FRAUD_EXAMPLES, len(fraud_indices))
    step = max(len(fraud_sorted_idx) // n_fraud, 1)
    selected_fraud_idx = fraud_indices[fraud_sorted_idx[::step][:n_fraud]]
    
    # Select normal examples
    normal_probs = y_probs[normal_indices]
    normal_sorted_idx = np.argsort(normal_probs)  # Sort by probability ascending
    
    n_normal = min(N_NORMAL_EXAMPLES, len(normal_indices))
    step = max(len(normal_sorted_idx) // n_normal, 1)
    selected_normal_idx = normal_indices[normal_sorted_idx[::step][:n_normal]]
    
    # Combine indices
    selected_indices = np.concatenate([selected_fraud_idx, selected_normal_idx])
    
    print(f"  ✓ Selected {len(selected_fraud_idx)} fraud + {len(selected_normal_idx)} normal examples")
    
    # Generate prediction cards
    print("\n6. Creating Prediction Visualizations...")
    feature_names = X_test.columns.tolist()
    
    for i, idx in enumerate(selected_indices, 1):
        sample_features = X_test.iloc[idx].values
        actual_label = y_test.iloc[idx]
        pred_prob = y_probs[idx]
        
        output_path = os.path.join(TEST_EXAMPLES_DIR, f'example_{i:02d}.png')
        
        create_prediction_card(
            sample_features=sample_features,
            actual_label=actual_label,
            pred_prob=pred_prob,
            threshold=threshold,
            feature_names=feature_names,
            sample_idx=i,
            output_path=output_path
        )
        
        status = "✓" if (pred_prob >= threshold) == actual_label else "✗"
        label_text = "FRAUD" if actual_label == 1 else "NORMAL"
        print(f"  {status} Example {i:02d}: {label_text} (prob: {pred_prob:.4f})")
    
    # Create summary statistics
    print("\n7. Inference Summary:")
    print(f"  - Total examples generated: {len(selected_indices)}")
    print(f"  - Fraud examples: {len(selected_fraud_idx)}")
    print(f"  - Normal examples: {len(selected_normal_idx)}")
    print(f"  - Saved to: {TEST_EXAMPLES_DIR}")
    
    # Calculate accuracy on selected samples
    selected_probs = y_probs[selected_indices]
    selected_labels = y_test.iloc[selected_indices]
    selected_preds = (selected_probs >= threshold).astype(int)
    accuracy = (selected_preds == selected_labels).mean()
    
    print(f"  - Accuracy on examples: {accuracy:.2%}")
    
    print_section("INFERENCE COMPLETED SUCCESSFULLY!")
    print(f"\nExample images saved to: {TEST_EXAMPLES_DIR}")
    print(f"View them with: open {TEST_EXAMPLES_DIR}")
    print("\n")


if __name__ == "__main__":
    main()
