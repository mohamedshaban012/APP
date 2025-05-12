import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector

# ======================================
# 1. DATA CLEANING SECTION
# ======================================
def clean_data(df):
    """Handles missing values, outliers, and data inconsistencies."""
    if df.isnull().sum().any():
        st.warning(f"Missing values detected: \n{df.isnull().sum()}")
        df = df.dropna()
    df = df[(df['BMI'] >= 12) & (df['BMI'] <= 60)]
    binary_cols = ['HighBP', 'HighChol', 'Diabetes_binary']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

# ======================================
# 2. EDA (EXPLORATORY DATA ANALYSIS)
# ======================================
def perform_eda(df):
    st.subheader("Diabetes Distribution")
    fig, ax = plt.subplots()
    df['Diabetes_binary'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xticklabels(['No Diabetes', 'Diabetes'], rotation=0)
    st.pyplot(fig)

    st.subheader("Feature Correlation with Diabetes")
    corr = df.corr()[['Diabetes_binary']].sort_values('Diabetes_binary', ascending=False)
    st.dataframe(corr.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(df.corr(), ax=ax, cmap='coolwarm', annot=True, fmt='.1f')
    st.pyplot(fig)

    st.subheader("Feature Distributions by Diabetes Status")
    for feat in ['BMI', 'Age', 'GenHlth']:
        fig, ax = plt.subplots()
        sns.boxplot(x='Diabetes_binary', y=feat, data=df, ax=ax)
        ax.set_title(f"{feat} Distribution")
        st.pyplot(fig)

# ======================================
# 3. DIMENSIONALITY REDUCTION (placeholder)
# ======================================
# Your PCA or feature selector logic goes here
# Make sure to define `X_reduced`, `y`, and `reducer` before model training

# ======================================
# 4. SVM MODEL PIPELINE
# ======================================
def train_svm(X_train, y_train, C=1.0, gamma='scale'):
    """Creates and trains an optimized SVM pipeline with tunable hyperparameters"""
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
    )
    with st.spinner(f'Training SVM (C={C}, gamma={gamma})...'):
        pipeline.fit(X_train, y_train)
    return pipeline

# ======================================
# MAIN APP
# ======================================
st.header("3. Model Training")

# Dummy data placeholder – in production, replace with cleaned & reduced data
# df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
# df = clean_data(df)
# X = df.drop('Diabetes_binary', axis=1)
# y = df['Diabetes_binary']
# X_reduced = X  # Example
# reducer = None

# You must replace the following dummy line with your actual dimensionality reduction output
st.warning("⚠️ Reminder: Make sure `X_reduced`, `y`, and `reducer` are defined from prior steps.")

# Example placeholders
X_reduced = None
y = None
reducer = None

# Hyperparameter tuning sliders
st.markdown("### Tune SVM Hyperparameters")
C_value = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0, 0.1)
gamma_option = st.selectbox("Kernel Coefficient (gamma)", options=['scale', 'auto'])

if st.button("Train SVM Model"):
    if X_reduced is None or y is None:
        st.error("Please run preprocessing and dimensionality reduction first.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

        model = train_svm(X_train, y_train, C=C_value, gamma=gamma_option)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/diabetes_model.joblib")
        joblib.dump(reducer, "models/feature_reducer.joblib")

        st.success("Model trained successfully!")
        st.subheader("Evaluation Metrics")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.metric("Accuracy", f"{acc:.2%}")
        st.metric("Precision", f"{prec:.2%}")
        st.metric("Recall", f"{rec:.2%}")
        st.metric("F1 Score", f"{f1:.2%}")

        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        labels = ['No Diabetes', 'Diabetes']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix\n[TP, FP, FN, TN]')
        st.pyplot(fig)
