import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Required Models for Task 6
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Metrics and Validation
from sklearn.model_selection import train_test_split, cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler

# Local modules
from preprocessing import load_and_clean_data, feature_engineering

# Configuration
DATA_PATH = 'data/creditcard.csv'
OUTPUT_DIR = 'outputs/plots_results'

def compare_all_models():
  
    print("--- Task 6: Comparing 4 Different ML Models ---")
    
    # 1. Data Preparation
    print("1. Loading and Preprocessing Data...")
    df = load_and_clean_data(DATA_PATH)
    
    # We apply the same feature engineering steps (e.g., 'is_night', 'Hour') 
    # to ensure all models benefit from the added context.
    df = feature_engineering(df)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Train/Test Split
    # We use 'stratify=y' to ensure the proportion of Fraud cases is preserved 
    # in both training and testing sets, which is crucial for imbalanced data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Defining Models
    # We selected these 4 algorithms to compare a simple linear baseline 
    # against powerful ensemble tree methods.
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs'),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    }
    
    results = []
    
    print(f"2. Starting Training & Comparison Loop ({len(models)} models)...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 3. Training Loop
    # We iterate through each model to train and evaluate them under identical conditions.
    for name, model in models.items():
        print(f"   -> Training {name}...")
        
        # Engineering Decision: Use of ImbPipeline
        # We wrap the steps in a Pipeline to prevent 'Data Leakage'.
        # SMOTE is applied ONLY during training folds, never on the validation fold.
        # RobustScaler is used to handle outliers in 'Amount'.
        pipeline = ImbPipeline([
            ('scaler', RobustScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        # Cross-Validation
        # We use 'f1' and 'recall' as primary metrics because Accuracy is misleading
        # for fraud detection (Accuracy Paradox).
        scoring = ['precision', 'recall', 'f1', 'roc_auc']
        scores = cross_validate(pipeline, X_train, y_train, cv=3, scoring=scoring, n_jobs=-1)
        
        # Aggregating results
        results.append({
            'Model': name,
            'Test F1 Score': scores['test_f1'].mean(),
            'Test Recall': scores['test_recall'].mean(),
            'Test ROC-AUC': scores['test_roc_auc'].mean(),
            'Training Time (s)': scores['fit_time'].mean()
        })
    
    # 4. Results Processing
    # We sort the models by F1 Score to easily identify the top performer.
    results_df = pd.DataFrame(results).sort_values(by='Test F1 Score', ascending=False)
    
    print("\n--- Model Comparison Results ---")
    print(results_df)
    
    # Save results to CSV for the project report
    results_df.to_csv(f"{OUTPUT_DIR}/model_comparison_results.csv", index=False)
    
    # 5. Visualization
    # We generate a bar chart to visually demonstrate the performance gap 
    # between linear models and tree-based ensembles.
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Test F1 Score', y='Model', data=results_df, palette='viridis')
    plt.title('Model Comparison: F1 Score')
    plt.xlabel('F1 Score (Higher is Better)')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison_chart.png")
    print(f"\nChart saved to {OUTPUT_DIR}/model_comparison_chart.png")
    
    print("--- Comparison Completed ---")

if __name__ == "__main__":
    compare_all_models()