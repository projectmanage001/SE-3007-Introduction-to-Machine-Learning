import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load artifacts
MODEL_PATH = 'outputs/models/final_model_pipeline.joblib'
SCALER_PATH = 'outputs/models/scaler.joblib'
THRESHOLD_PATH = 'outputs/models/best_threshold.txt'

@st.cache_resource
def load_system():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(THRESHOLD_PATH, 'r') as f:
        threshold = float(f.read())
    return model, scaler, threshold

try:
    model, scaler, threshold = load_system()
    st.sidebar.success("System Loaded Successfully")
except:
    st.error("Please run 'main_pipeline.py' first to generate models!")
    st.stop()

st.title("🛡️ Credit Card Fraud Detection System")
st.markdown(f"**Current Model Threshold:** `{threshold:.4f}`")
st.write("This dashboard simulates a real-time fraud detection interface developed by our team.")

# Tab Selection
tab1, tab2 = st.tabs(["Single Prediction (API Mode)", "Batch Analysis"])

with tab1:
    st.header("Real-time Transaction Check")
    
    # Simulating inputs
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
        hour = st.slider("Hour of Day", 0, 23, 14)
    with col2:
        # Just random features for V1-V28 simulation since we can't input 28 floats manually easily
        st.info("V1-V28 features are simulated as random for this demo.")
    
    if st.button("Analyze Transaction"):
        # Create input array
        input_data = pd.DataFrame(np.random.randn(1, 28), columns=[f'V{i}' for i in range(1, 29)])
        input_data['Amount'] = amount
        input_data['Hour'] = hour
        input_data['is_night'] = 1 if 0 <= hour <= 6 else 0
        
        # Scale
        input_data['Amount'] = scaler.transform(input_data[['Amount']])
        
        # Predict Probability
        prob = model.predict_proba(input_data)[0, 1]
        
        st.divider()
        if prob >= threshold:
            st.error(f"🚨 FRAUD DETECTED! (Probability: {prob:.2%})")
        else:
            st.success(f"✅ Transaction Valid (Probability: {prob:.2%})")

with tab2:
    st.header("Upload Transaction Batch (CSV)")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df_upload.head())
        
        if st.button("Run Batch Analysis"):
            # Preprocessing Steps (Simplified for demo)
            # Assuming CSV has Time, Amount and V1-V28
            process_df = df_upload.copy()
            if 'Time' in process_df.columns:
                process_df['Hour'] = process_df['Time'].apply(lambda x: np.floor(x / 3600)) % 24
                process_df['is_night'] = process_df['Hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
                process_df = process_df.drop('Time', axis=1)
            
            # Scale
            process_df['Amount'] = scaler.transform(process_df[['Amount']])
            
            # Predict
            probs = model.predict_proba(process_df.drop('Class', axis=1, errors='ignore'))[:, 1]
            preds = (probs >= threshold).astype(int)
            
            df_upload['Fraud_Probability'] = probs
            df_upload['Prediction'] = preds
            
            st.subheader("Results")
            st.write(df_upload[['Time', 'Amount', 'Fraud_Probability', 'Prediction']].head())
            
            fraud_count = df_upload['Prediction'].sum()
            st.warning(f"Detected {fraud_count} fraudulent transactions out of {len(df_upload)}.")