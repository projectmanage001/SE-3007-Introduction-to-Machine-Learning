import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import numpy as np

# Configuration / Settings
MODEL_PATH = 'outputs/models/final_model_pipeline.joblib'
DATA_PATH = 'data/creditcard.csv'
OUTPUT_DIR = 'outputs/plots_results'

def explain_model():
   
  
    print("--- Task 9 & 10: Explainability Analysis (SHAP & Feature Importance) ---")
    
    # 1. Load Model and Data
    if not os.path.exists(MODEL_PATH):
        print("Error: Trained model not found! Please run 'main_pipeline.py' first.")
        return

    print("1. Loading Model and Data...")
    pipeline = joblib.load(MODEL_PATH)
    
    # We need to extract the actual XGBoost model step from the pipeline.
    # The pipeline steps are: [('smote', SMOTE), ('xgb', XGBClassifier)]
    # SHAP requires the fitted model object, not the entire pipeline wrapper.
    model = pipeline.named_steps['xgb']
    
    # Load raw data for analysis
    df = pd.read_csv(DATA_PATH)
    
    # Re-apply Feature Engineering
    # We must replicate the exact feature engineering steps used in training
    # to ensure the input shape matches what the model expects.
    df['Hour'] = df['Time'].apply(lambda x: np.floor(x / 3600)) % 24
    df['is_night'] = df['Hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
    df = df.drop(['Time', 'Class'], axis=1) # Drop target and unused columns
    
    # Sampling for Performance
    # Calculating SHAP values for the entire dataset is computationally expensive.
    # We take a random sample of 1000 instances to speed up the explanation process.
    X_sample = df.sample(1000, random_state=42)
    
    # Load the pre-fitted Scaler
    # It is crucial to use the same scaler saved during training to maintain data consistency.
    scaler = joblib.load('outputs/models/scaler.joblib')
    
    # Apply Scaling to 'Amount'
    X_sample_scaled = X_sample.copy()
    X_sample_scaled['Amount'] = scaler.transform(X_sample[['Amount']])
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- TASK 9: Feature Importance ---
    print("2. Generating Feature Importance Plot...")
    
    # Extracting built-in feature importance from XGBoost
    importance = model.feature_importances_
    feature_names = X_sample.columns
    
    # Creating a DataFrame to organize and sort features
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10) # Top 10 features
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='magma')
    plt.title('Top 10 Most Important Features (XGBoost)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
    print(f"   -> Saved to {OUTPUT_DIR}/feature_importance.png")

    # --- TASK 10: SHAP Analysis ---
    print("3. Performing SHAP Analysis (This helps explain 'Why')...")
    
    # Initialize SHAP Explainer
    # TreeExplainer is optimized for tree-based models like XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample_scaled)
    
    # 1. Summary Plot (Beeswarm)
    # This plot shows the distribution of impact each feature has on the model output.
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample_scaled, show=False)
    plt.title("SHAP Summary Plot", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary_plot.png")
    plt.close()
    print(f"   -> Saved to {OUTPUT_DIR}/shap_summary_plot.png")
    
    # 2. Bar Plot (Mean Absolute Value)
    # This shows the average magnitude of impact for each feature.
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample_scaled, plot_type="bar", show=False)
    plt.title("Mean SHAP Values (Average Impact)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_bar_plot.png")
    plt.close()
    print(f"   -> Saved to {OUTPUT_DIR}/shap_bar_plot.png")

    print("--- Explainability Analysis Completed ---")

if __name__ == "__main__":
    explain_model()