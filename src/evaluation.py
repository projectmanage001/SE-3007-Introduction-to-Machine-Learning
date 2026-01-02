import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os  
# GÜNCELLEME: roc_curve ve auc eklendi
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report, roc_curve, auc

def evaluate_model(model, X_test, y_test, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get probabilities instead of hard predictions
    y_probs = model.predict_proba(X_test)[:, 1]

    # --- 1. Precision-Recall Curve & Threshold Tuning ---
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    # Handle NaN values usually occurring at the start/end
    f1_scores = np.nan_to_num(f1_scores)
    
    best_idx = np.argmax(f1_scores)
    
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = thresholds[-1]
    
    print(f"Optimal Threshold found: {best_threshold}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='XGBoost')
    plt.scatter(recall[best_idx], precision[best_idx], marker='o', color='black', label='Best Threshold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Best Threshold: {best_threshold:.2f})')
    plt.legend()
    plt.savefig(f"{output_dir}/precision_recall_curve.png")
    plt.close()

    # --- 2. ROC Curve (Task 8 - YENİ EKLENEN KISIM) ---
    # Raporda Task 8 altında "Plot ROC curves" istendiği için bu kısmı ekledik.
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()

    # --- 3. Confusion Matrix at Best Threshold ---
    y_pred_optimal = (y_probs >= best_threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred_optimal)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (Threshold: {best_threshold:.2f})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{output_dir}/confusion_matrix_optimal.png")
    plt.close()

    # Save metrics to a text file
    report = classification_report(y_test, y_pred_optimal)
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(f"Optimal Threshold: {best_threshold}\n\n")
        f.write(report)

    print(f"Evaluation plots and report saved to {output_dir}")
    return best_threshold