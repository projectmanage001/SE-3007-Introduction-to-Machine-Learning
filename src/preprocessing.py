import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import RobustScaler

def load_and_clean_data(filepath):
    
    # Load the dataset
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
    
    df = pd.read_csv(filepath)
    # Removing duplicates is a good practice we decided to implement
    df.drop_duplicates(inplace=True)
    return df

def perform_eda_and_save_plots(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # We find the class distribution counts
    class_counts = df['Class'].value_counts()
    print("\n--- Class Distribution (Raw Counts) ---")
    print(class_counts)
    
    # Saving the results as txt file
    with open(f"{output_dir}/class_counts.txt", "w") as f:
        f.write("Class Distribution (0=Normal, 1=Fraud):\n")
        f.write(class_counts.to_string())
    

    # 1. Class Distribution Pie Chart
    plt.figure(figsize=(6, 6))
    plt.pie(class_counts, labels=['Normal', 'Fraud'], autopct='%1.1f%%', colors=['#1f77b4', '#d62728'])
    plt.title('Class Distribution: Highly Imbalanced')
    plt.savefig(f"{output_dir}/class_distribution.png")
    plt.close()

    # 2. Amount Distribution (Log Scale because amounts vary wildly)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Amount'], bins=50, kde=True)
    plt.yscale('log')
    plt.title('Transaction Amount Distribution (Log Scale)')
    plt.savefig(f"{output_dir}/amount_distribution.png")
    plt.close()

    # 3. Correlation Matrix
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 20})
    plt.title('Correlation Matrix of Features')
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    # 4. PCA Scatter Plot
    plt.figure(figsize=(10, 6))
    sample_df = df.sample(min(5000, len(df)), random_state=42)# We created a ssample for better visualization than using the full dataset
    sns.scatterplot(x='V1', y='V2', hue='Class', data=sample_df, palette=['#1f77b4', '#d62728'], alpha=0.6)
    plt.title('PCA Scatter Plot: V1 vs V2 (Fraud Separation)')
    plt.savefig(f"{output_dir}/pca_scatter_v1_v2.png")
    plt.close()
    
    print(f"EDA plots and class counts saved to {output_dir}")

def feature_engineering(df):
    df = df.copy()
    
    # Converting seconds to hours (24-hour format) -> Because time of day might influence fraud likelihood
    # 3600 seconds = 1 hour
    df['Hour'] = df['Time'].apply(lambda x: np.floor(x / 3600)) % 24
    
    # Feature: Is it a night transaction? (e.g., between 00:00 and 06:00)
    # We suspect fraud might happen more during these hours.
    df['is_night'] = df['Hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
    
    # We drop the original 'Time' column as we extracted necessary info
    df = df.drop('Time', axis=1)
    
    return df

def scale_data(df):
    scaler = RobustScaler()
    
    # We only scale Amount and Time-derived features if needed. 
    # V1-V28 are already PCA transformed, but scaling them again doesn't hurt logic.
    # To keep it consistent, we scale 'Amount' specifically.
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    return df, scaler