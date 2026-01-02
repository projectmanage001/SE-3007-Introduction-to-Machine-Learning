import pandas as pd
import numpy as np
import joblib
import os

# Configuration
MODEL_PATH = 'outputs/models/final_model_pipeline.joblib'
SCALER_PATH = 'outputs/models/scaler.joblib'
OUTPUT_DIR = 'outputs/plots_results'

def run_scenarios():
    print("--- Task 15: Running Fraud Scenarios (Student Analysis) ---")
    print("We designed these scenarios to see if our model works like a human expert.")
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found! Please run 'main_pipeline.py' first.")
        return

    # 1. Loading System
    print("1. Loading the trained XGBoost model and Scaler...")
    pipeline = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    # Accessing the actual model inside the pipeline
    model = pipeline.named_steps['xgb']
    
    # 2. Defining Scenarios
    # We use a base template where all V features are 0 (average), 
    # and then we tweak specific values to simulate real-life situations.
    base_features = {f'V{i}': 0.0 for i in range(1, 29)}
    
    scenarios = [
        {
            "name": "Scenario 1: The 'Rich User' Test",
            "detail": "Amount: $25,000 | Time: 14:00 (Afternoon) | Features: Normal",
            "student_comment": "We want to check: Does the model flag a transaction just because the amount is huge? We hope NOT, because rich people also use credit cards.",
            "data": {**base_features, 'Amount': 25000.0, 'Hour': 14, 'is_night': 0}
        },
        {
            "name": "Scenario 2: The 'Insomniac' Test",
            "detail": "Amount: $15.50 | Time: 03:00 AM (Night) | Features: Normal",
            "student_comment": "We want to check: Does the model panic just because it's night time? Buying a snack at 3 AM shouldn't be fraud.",
            "data": {**base_features, 'Amount': 15.50, 'Hour': 3, 'is_night': 1}
        },
        {
            "name": "Scenario 3: The 'Hacker' Attack",
            "detail": "Amount: $5,000 | Time: 04:00 AM | V14 & V4: Highly Abnormal",
            "student_comment": "We manipulated V14 and V4 (based on our Feature Importance charts). This simulates a card being used in a weird location or device. We expect a FRAUD alert.",
            "data": {**base_features, 'Amount': 5000.0, 'Hour': 4, 'is_night': 1, 'V14': -12.0, 'V4': 8.5}
        }
    ]
    
    results = []

    # 3. Running Simulations
    print("3. Testing scenarios against the model...")
    
    for sc in scenarios:
        # Create a single-row DataFrame
        df = pd.DataFrame([sc['data']])
        
        # Apply Scaling (Crucial step!)
        # We must scale the 'Amount' because our model was trained on scaled data.
        df_scaled = df.copy()
        df_scaled['Amount'] = scaler.transform(df[['Amount']])
        
        # Predict Probability
        prob = model.predict_proba(df_scaled)[0, 1]
        
        # Decision Logic (Using 0.5 as threshold)
        is_fraud = prob > 0.5
        status = "🚨 FRAUD DETECTED" if is_fraud else "✅ Normal Transaction"
        
        # Generating a Student Interpretation based on result
        if sc['name'] == "Scenario 1: The 'Rich User' Test":
            interpretation = "Success! The model understands that high amount alone is NOT fraud." if not is_fraud else "Fail. The model is too sensitive to high amounts."
        elif sc['name'] == "Scenario 2: The 'Insomniac' Test":
            interpretation = "Good. The model knows that night transactions can be safe." if not is_fraud else "Warning. The model might be biased against night activity."
        else: # Scenario 3
            interpretation = "Perfect! The model caught the complex pattern hidden in V-features." if is_fraud else "Fail. The model missed the manipulated features."

        results.append({
            "Scenario Name": sc['name'],
            "Input Details": sc['detail'],
            "Our Intent (Why?)": sc['student_comment'],
            "Model Probability": f"{prob:.4f}",
            "Final Prediction": status,
            "Student Conclusion": interpretation
        })

    # 4. Saving and Displaying Results
    results_df = pd.DataFrame(results)
    
    print("\n--- Final Scenario Analysis Results ---")
    # We transpose the dataframe for better readability in terminal if needed, 
    # but here we just print columns or save to CSV.
    print(results_df[['Scenario Name', 'Final Prediction', 'Student Conclusion']])
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    results_df.to_csv(f"{OUTPUT_DIR}/scenario_test_results.csv", index=False)
    print(f"\n[INFO] Detailed results have been saved to: {OUTPUT_DIR}/scenario_test_results.csv")
    print("We can use this CSV table directly in our Project Report under Task 15.")

if __name__ == "__main__":
    run_scenarios()