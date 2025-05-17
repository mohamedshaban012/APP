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
    filepath = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        st.error("Default dataset not found. Please upload your own CSV file.")
        return None

def clean_data(df):
    if df is None:
        return None
    df = df.dropna()
    df = df[(df['BMI'] >= 12) & (df['BMI'] <= 60)]
    for col in ['HighBP', 'HighChol', 'Diabetes_binary']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

# =================== EDA ===================
def perform_eda(df):
    if df is None:
        return
        
    st.subheader("Diabetes Distribution")
    fig, ax = plt.subplots()
    df['Diabetes_binary'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xticklabels(['No Diabetes', 'Diabetes'], rotation=0)
    st.pyplot(fig)
    plt.close()

    st.subheader("Feature Correlation with Diabetes")
    corr = df.corr(numeric_only=True)[['Diabetes_binary']].sort_values('Diabetes_binary', ascending=False)
    st.dataframe(corr.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(df.corr(numeric_only=True), ax=ax, cmap='coolwarm', annot=False)
    st.pyplot(fig)
    plt.close()

    st.subheader("Feature Distributions by Diabetes Status")
    for feat in ['BMI', 'Age', 'GenHlth']:
        if feat in df.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x='Diabetes_binary', y=feat, data=df, ax=ax)
            ax.set_title(f"{feat} Distribution")
            st.pyplot(fig)
            plt.close()

# =================== DIMENSIONALITY REDUCTION ===================
def perform_dimensionality_reduction(X, y):
    if X is None or y is None:
        return None, None
        
    st.markdown("#### Dimensionality Reduction")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA().fit(X_scaled)
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.axhline(y=0.95, color='r', linestyle='--')
    ax.set_xlabel('Components')
    ax.set_ylabel('Explained Variance')
    st.pyplot(fig)
    plt.close()

    method = st.radio("Select feature selection method:", ["PCA", "Forward Selection", "Backward Selection", "Feature Importance"])

    if method == "PCA":
        n_components = st.slider("Number of PCA components", 1, min(20, X.shape[1]), 5)
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X_scaled)
        return X_transformed, pca

    elif method in ["Forward Selection", "Backward Selection"]:
        direction = 'forward' if method == "Forward Selection" else 'backward'
        n_features = st.slider("Number of features to select", 1, min(20, X.shape[1]), 5)
        sfs = SequentialFeatureSelector(RandomForestClassifier(n_estimators=50),
                                      n_features_to_select=n_features,
                                      direction=direction, cv=5)
        sfs.fit(X, y)
        selected_features = X.columns[sfs.get_support()]
        return X[selected_features], sfs

    else:  # Feature Importance
        model = RandomForestClassifier()
        model.fit(X, y)
        importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
        importance = importance.sort_values("Importance", ascending=False)
        top_features = importance.head(5)['Feature'].values
        
        fig, ax = plt.subplots()
        sns.barplot(data=importance.head(10), x="Importance", y="Feature", ax=ax)
        st.pyplot(fig)
        plt.close()
        
        return X[top_features], top_features.tolist()

# =================== MODEL TRAINING ===================
def train_svm(X_train, y_train):
    if X_train is None or y_train is None:
        return None
        
    pipeline = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(LinearSVC(dual=False, max_iter=2000), cv=5)
    )
    pipeline.fit(X_train, y_train)
    return pipeline

# =================== MAIN APP ===================
def main():
    st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
    st.title("🩺 Diabetes Risk Predictor")

    with st.sidebar:
        st.header("Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        st.markdown("---")
        st.header("Navigation")
        st.markdown("- 📊 EDA\n- 🔍 Feature Selection\n- 🧠 Train Model\n- 📈 Predict Risk")

    # Load data
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return
    else:
        df = load_default_data()
        if df is None:
            return

    df_clean = clean_data(df)
    if df_clean is None or df_clean.empty:
        st.error("Dataset is empty or invalid after cleaning.")
        return

    if 'Diabetes_binary' not in df_clean.columns:
        st.error("Target column 'Diabetes_binary' not found in dataset.")
        return

    X = df_clean.drop('Diabetes_binary', axis=1)
    y = df_clean['Diabetes_binary']

    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🔍 Feature Selection", "🧠 Train Model", "📈 Predict Risk"])

    with tab1:
        st.header("Exploratory Data Analysis")
        perform_eda(df_clean)

    with tab2:
        st.header("Feature Selection & Dimensionality Reduction")
        X_reduced, reducer = perform_dimensionality_reduction(X, y)

    with tab3:
        st.header("Train Model")
        if 'X_reduced' in locals() and X_reduced is not None:
            if st.button("Train SVM Model"):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_reduced, y, test_size=0.2, random_state=42
                )
                model = train_svm(X_train, y_train)
                
                if model is not None:
                    os.makedirs("models", exist_ok=True)
                    joblib.dump((model, reducer), "models/model.joblib")

                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{acc:.2%}")
                    st.text(classification_report(y_test, y_pred))

                    fig, ax = plt.subplots()
                    sns.heatmap(confusion_matrix(y_test, y_pred), 
                                annot=True, fmt='d', cmap='Blues', ax=ax)
                    st.pyplot(fig)
                    plt.close()
        else:
            st.warning("Please perform feature selection first.")

    with tab4:
        st.header("Predict Risk")
        if os.path.exists("models/model.joblib"):
            try:
                model, reducer = joblib.load("models/model.joblib")
                
                st.subheader("Enter Values for Prediction")
                input_data = {}
                
                with st.form("prediction_form"):
                    cols = st.columns(2)
                    features_to_show = X.columns[:min(10, len(X.columns))]  # Show first 10 features max
                    
                    for i, feat in enumerate(features_to_show):
                        with cols[i % 2]:
                            input_data[feat] = st.slider(
                                feat,
                                float(X[feat].min()),
                                float(X[feat].max()),
                                float(X[feat].median())
                            )
                    submitted = st.form_submit_button("Predict Diabetes Risk")

                if submitted:
                    # Fill missing features with median values
                    for feat in X.columns:
                        if feat not in input_data:
                            input_data[feat] = float(X[feat].median())
                    
                    input_df = pd.DataFrame([input_data])
                    
                    try:
                        if isinstance(reducer, PCA):
                            scaled_input = StandardScaler().fit_transform(input_df)
                            input_transformed = reducer.transform(scaled_input)
                        elif hasattr(reducer, 'get_support'):  # SequentialFeatureSelector
                            input_transformed = input_df[X.columns[reducer.get_support()]]
                        elif isinstance(reducer, list):  # Feature Importance list
                            input_transformed = input_df[reducer]
                        else:
                            st.error("Unknown reducer type")
                            return
                            
                        if hasattr(model, 'predict_proba'):
                            risk = model.predict_proba(input_transformed)[0][1]
                        else:
                            risk = model.predict(input_transformed)[0]
                            
                        st.metric("Risk Score", f"{risk:.1%}")
                        if risk > 0.7:
                            st.error("High Risk - Consult a doctor")
                        elif risk > 0.4:
                            st.warning("Moderate Risk - Monitor health")
                        else:
                            st.success("Low Risk - Keep healthy habits")
                            
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
        else:
            st.warning("Please train a model first.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("🚨 The app crashed with the following error:")
        st.code(traceback.format_exc())
