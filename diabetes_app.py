import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

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
@st.cache_data

def load_default_data():
    return pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

@st.cache_data

def clean_data(df):
    df = df.dropna()
    df = df[(df['BMI'] >= 12) & (df['BMI'] <= 60)]
    for col in ['HighBP', 'HighChol', 'Diabetes_binary']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

# =================== EDA ===================
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
    sns.heatmap(df.corr(), ax=ax, cmap='coolwarm', annot=False)
    st.pyplot(fig)

    st.subheader("Feature Distributions by Diabetes Status")
    for feat in ['BMI', 'Age', 'GenHlth']:
        fig, ax = plt.subplots()
        sns.boxplot(x='Diabetes_binary', y=feat, data=df, ax=ax)
        ax.set_title(f"{feat} Distribution")
        st.pyplot(fig)

# =================== DIMENSIONALITY REDUCTION ===================
def perform_dimensionality_reduction(X, y):
    st.markdown("#### Dimensionality Reduction")

    pca = PCA().fit(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.xlabel('Components')
    plt.ylabel('Explained Variance')
    st.pyplot(fig)

    method = st.radio("Select feature selection method:", ["PCA", "Forward Selection", "Backward Selection", "Feature Importance"])
    
    if method == "PCA":
        n_components = st.slider("Number of PCA components", 1, min(20, X.shape[1]), 5)
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(StandardScaler().fit_transform(X))
        return X_transformed, pca

    elif method in ["Forward Selection", "Backward Selection"]:
        direction = 'forward' if method == "Forward Selection" else 'backward'
        n_features = st.slider("Number of features to select", 1, min(20, X.shape[1]), 5)
        sfs = SequentialFeatureSelector(RandomForestClassifier(n_estimators=50),
                                        n_features_to_select=n_features,
                                        direction=direction, cv=5)
        sfs.fit(X, y)
        return X[X.columns[sfs.get_support()]], sfs

    else:
        model = RandomForestClassifier()
        model.fit(X, y)
        importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
        top_features = importance.sort_values("Importance", ascending=False).head(5)['Feature']
        fig, ax = plt.subplots()
        sns.barplot(data=importance.sort_values("Importance", ascending=False).head(10), x="Importance", y="Feature", ax=ax)
        st.pyplot(fig)
        return X[top_features], model

# =================== MODEL TRAINING ===================
def train_svm(X_train, y_train):
    pipeline = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(LinearSVC(dual=False, max_iter=2000), cv=5)
    )
    pipeline.fit(X_train, y_train)
    return pipeline

# =================== MAIN APP ===================
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
st.title("🩺 Diabetes Risk Predictor")

with st.sidebar:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.markdown("---")
    st.header("Navigation")
    st.markdown("- 📊 EDA\n- 🔍 Feature Selection\n- 🧠 Train Model\n- 📈 Predict Risk")

# Load and clean data
df = pd.read_csv(uploaded_file) if uploaded_file else load_default_data()
df_clean = clean_data(df)
X = df_clean.drop('Diabetes_binary', axis=1)
y = df_clean['Diabetes_binary']

# Tabs for each section
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🔍 Feature Selection", "🧠 Train Model", "📈 Predict Risk"])

with tab1:
    st.header("Exploratory Data Analysis")
    perform_eda(df_clean)

with tab2:
    st.header("Feature Selection & Dimensionality Reduction")
    X_reduced, reducer = perform_dimensionality_reduction(X, y)

with tab3:
    st.header("Train Model")
    if st.button("Train SVM Model"):
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
        model = train_svm(X_train, y_train)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")
        joblib.dump(reducer, "models/reducer.joblib")

        acc = accuracy_score(y_test, model.predict(X_test))
        st.metric("Accuracy", f"{acc:.2%}")
        st.text(classification_report(y_test, model.predict(X_test)))

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

with tab4:
    st.header("Predict Risk")
    if os.path.exists("models/model.joblib"):
        model = joblib.load("models/model.joblib")
        reducer = joblib.load("models/reducer.joblib")

        st.subheader("Enter Values for Prediction")
        input_data = {}
        with st.form("prediction_form"):
            cols = st.columns(2)
            for i, feat in enumerate(X.columns[:5]):
                with cols[i % 2]:
                    input_data[feat] = st.slider(feat, float(X[feat].min()), float(X[feat].max()), float(X[feat].median()))
            submitted = st.form_submit_button("Predict Diabetes Risk")

        if submitted:
            for feat in X.columns:
                if feat not in input_data:
                    input_data[feat] = float(X[feat].median())
            input_df = pd.DataFrame([input_data])
            if isinstance(reducer, PCA):
                scaled_input = StandardScaler().fit_transform(input_df)
                input_transformed = reducer.transform(scaled_input)
            elif hasattr(reducer, 'get_support'):
                input_transformed = input_df[X.columns[reducer.get_support()]]
            else:
                input_transformed = input_df[reducer.feature_names_in_]

            risk = model.predict_proba(input_transformed)[0][1]
            st.metric("Risk Score", f"{risk:.1%}")
            if risk > 0.7:
                st.error("High Risk - Consult a doctor")
            elif risk > 0.4:
                st.warning("Moderate Risk - Monitor health")
            else:
                st.success("Low Risk - Keep healthy habits")
