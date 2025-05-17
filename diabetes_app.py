import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import traceback

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

# =================== UTILS ===================
def load_default_data():
    try:
        filepath = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            st.error("Default dataset not found. Please upload your own CSV file.")
            return None
    except Exception as e:
        st.error(f"Error loading default data: {str(e)}")
        return None

def clean_data(df):
    if df is None:
        return None
    try:
        df = df.dropna()
        df = df[(df['BMI'] >= 12) & (df['BMI'] <= 60)]
        for col in ['HighBP', 'HighChol', 'Diabetes_binary']:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df
    except Exception as e:
        st.error(f"Error cleaning data: {str(e)}")
        return None

# =================== MAIN APP ===================
def main():
    st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
    st.title("🩺 Diabetes Risk Predictor")

    try:
        # Load data
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = load_default_data()
        
        if df is None or df.empty:
            st.error("No data available. Please upload a valid CSV file.")
            return

        st.write("Data loaded successfully!")
        st.write(f"Dataset shape: {df.shape}")
        
    except Exception as e:
        st.error(f"Error initializing app: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()
