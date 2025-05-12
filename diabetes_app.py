import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

# ======================================
# 1. DATA CLEANING SECTION
# ======================================
def clean_data(df):
    """Handles missing values, outliers, and data inconsistencies."""
    # Check for missing values
    if df.isnull().sum().any():
        st.warning(f"Missing values detected: \n{df.isnull().sum()}")
        df = df.dropna()
    
    # Handle outliers (example for BMI)
    df = df[(df['BMI'] >= 12) & (df['BMI'] <= 60)]  # Clinical BMI range
    
    # Convert binary features to int
    binary_cols = ['HighBP', 'HighChol', 'Diabetes_binary']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    return df

# ======================================
# 2. EDA (EXPLORATORY DATA ANALYSIS)
# ======================================
def perform_eda(df):
    """Generates visualizations and statistical summaries"""
    
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
    num_features = ['BMI', 'Age', 'GenHlth']
    for feat in num_features:
        fig, ax = plt.subplots()
        sns.boxplot(x='Diabetes_binary', y=feat, data=df, ax=ax)
        ax.set_title(f"{feat} Distribution")
        st.pyplot(fig)

# ======================================
# 3. DIMENSIONALITY REDUCTION
# ======================================
def perform_dimensionality_reduction(X, y):
    """Performs PCA and feature selection"""
    
    st.subheader("Dimensionality Reduction")
    
    # PCA Analysis
    st.markdown("#### Principal Component Analysis (PCA)")
    pca = PCA().fit(StandardScaler().fit_transform(X))
    
    # Plot explained variance
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.axhline(y=0.95, color='r', linestyle='--')
    st.pyplot(fig)
    
    n_components = st.slider("Select number of PCA components", 
                            min_value=1, 
                            max_value=min(20, X.shape[1]), 
                            value=5)
    
    # Feature Selection Methods
    st.markdown("#### Feature Selection Methods")
    method = st.radio("Select feature selection method:",
                     ["PCA", "Forward Selection", "Backward Selection", "Feature Importance"])
    
    if method == "PCA":
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(StandardScaler().fit_transform(X))
        st.info(f"Selected {n_components} principal components explaining {np.sum(pca.explained_variance_ratio_):.1%} of variance")
        return X_transformed, pca
    
    elif method in ["Forward Selection", "Backward Selection"]:
        direction = 'forward' if method == "Forward Selection" else 'backward'
        n_features = st.slider(f"Number of features to select ({method})", 
                              min_value=1, 
                              max_value=min(20, X.shape[1]), 
                              value=5)
        
        # Use a simpler estimator for feature selection
        sfs = SequentialFeatureSelector(
            RandomForestClassifier(n_estimators=50, random_state=42),
            n_features_to_select=n_features,
            direction=direction,
            cv=5
        )
        
        with st.spinner(f'Running {method}... This may take a while'):
            sfs.fit(X, y)
        
        selected_features = X.columns[sfs.get_support()]
        st.success(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")
        return X[selected_features], sfs
    
    else:  # Feature Importance
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        n_features = st.slider("Number of important features to select", 
                              min_value=1, 
                              max_value=min(20, X.shape[1]), 
                              value=5)
        
        selected_features = importance['Feature'].head(n_features)
        
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x='Importance', y='Feature', data=importance.head(n_features), palette='viridis')
        ax.set_title(f"Top {n_features} Most Important Features")
        st.pyplot(fig)
        
        return X[selected_features], model

# ======================================
# 4. SVM MODEL PIPELINE
# ======================================
def train_svm(X_train, y_train):
    """Creates and trains an SVM pipeline"""
    pipeline = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            LinearSVC(dual=False, random_state=42, max_iter=2000),
            cv=5
        )
    )
    
    with st.spinner('Training model...'):
        pipeline.fit(X_train, y_train)
    return pipeline

# ======================================
# MAIN APP
# ======================================
st.set_page_config(layout="wide")
st.title("🩺 Advanced Diabetes Risk Predictor")

# Add documentation
with st.expander("How to use this app"):
    st.markdown("""
    1. Explore the EDA section to understand the data
    2. Choose dimensionality reduction method
    3. Train the model
    4. Make predictions using the interactive input sliders
    """)

# Load default dataset
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
df_clean = clean_data(df)

# EDA Section
st.header("1. Exploratory Data Analysis")
perform_eda(df_clean)

# Feature Selection and Dimensionality Reduction
st.header("2. Dimensionality Reduction & Feature Selection")
X = df_clean.drop('Diabetes_binary', axis=1)
y = df_clean['Diabetes_binary']

X_reduced, reducer = perform_dimensionality_reduction(X, y)

# Model Training
st.header("3. Model Training")
if st.button("Train SVM Model"):
    if isinstance(X_reduced, np.ndarray):  # PCA was used
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, test_size=0.2, random_state=42)
    else:  # Feature selection was used
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, test_size=0.2, random_state=42)
    
    model = train_svm(X_train, y_train)
    
    # Save model and reducer
    MODEL_PATH = "models/diabetes_model.joblib"
    REDUCER_PATH = "models/feature_reducer.joblib"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(reducer, REDUCER_PATH)
    
    # Evaluation
    st.success("Model trained successfully!")
    st.subheader("Evaluation Metrics")
    acc = accuracy_score(y_test, model.predict(X_test))
    st.metric("Accuracy", f"{acc:.2%}")
    
    st.write("Classification Report:")
    st.text(classification_report(y_test, model.predict(X_test)))
    
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), 
                annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

# Prediction Interface
if os.path.exists("models/diabetes_model.joblib"):
    st.header("4. Risk Prediction")
    model = joblib.load("models/diabetes_model.joblib")
    reducer = joblib.load("models/feature_reducer.joblib")
    
    # Get the most important features
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    else:
        # For linear models
        try:
            coefs = model.named_steps['calibratedclassifiercv'].estimators_[0].coef_[0]
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': coefs
            }).sort_values('Coefficient', key=abs, ascending=False)
        except:
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Correlation': X.corrwith(y)
            }).sort_values('Correlation', key=abs, ascending=False)
    
    top_features = importance.head(5)['Feature'].tolist()
    
    st.subheader("Key Risk Factors")
    st.write("Adjust these top 5 most influential factors to see how they affect diabetes risk:")
    
    # Create input widgets for top features
    inputs = {}
    cols = st.columns(2)
    for i, feat in enumerate(top_features):
        with cols[i%2]:
            if X[feat].nunique() <= 5:  # Categorical
                inputs[feat] = st.selectbox(feat, sorted(X[feat].unique()))
            else:  # Numerical
                inputs[feat] = st.slider(
                    feat,
                    float(X[feat].min()),
                    float(X[feat].max()),
                    float(X[feat].median())
                )
    
    # Fill remaining features with median values
    for feat in X.columns:
        if feat not in inputs:
            inputs[feat] = float(X[feat].median())
    
    if st.button("Predict Diabetes Risk"):
        input_df = pd.DataFrame([inputs], columns=X.columns)
        
        # Apply the same transformation used in training
        if isinstance(reducer, PCA):
            scaled_input = StandardScaler().fit_transform(input_df)
            transformed_input = reducer.transform(scaled_input)
        elif hasattr(reducer, 'get_support'):  # Feature selector
            transformed_input = input_df[X.columns[reducer.get_support()]]
        else:  # Feature importance
            transformed_input = input_df[reducer.feature_names_in_]
        
        # Get probability
        proba = model.predict_proba(transformed_input)[0][1]
        
        # Display results
        st.metric("Diabetes Risk Probability", f"{proba:.1%}")
        
        # Risk interpretation
        if proba > 0.7:
            st.error("High Risk - Consult a doctor immediately")
            st.markdown("""
            **Recommendations:**
            - Schedule a doctor's appointment
            - Monitor blood sugar levels regularly
            - Consider lifestyle changes
            """)
        elif proba > 0.4:
            st.warning("Moderate Risk - Monitor your health")
            st.markdown("""
            **Recommendations:**
            - Increase physical activity
            - Improve diet (reduce sugar intake)
            - Consider regular checkups
            """)
        else:
            st.success("Low Risk - Maintain healthy habits")
            st.markdown("""
            **Recommendations:**
            - Continue balanced diet
            - Regular exercise
            - Annual health checkups
            """)
        
        # Show feature contributions
        st.subheader("How Each Factor Affects Your Risk")
        fig, ax = plt.subplots(figsize=(10,4))
        
        if 'Coefficient' in importance.columns:
            # For linear models
            top_factors = importance.head(5).copy()
            top_factors['Effect'] = top_factors['Coefficient'].apply(
                lambda x: "Increases Risk" if x > 0 else "Decreases Risk")
            sns.barplot(x='Coefficient', y='Feature', data=top_factors, 
                       palette=['red' if x > 0 else 'green' for x in top_factors['Coefficient']])
            ax.set_title("Top Factors Affecting Your Diabetes Risk")
        else:
            # For other models
            sns.barplot(x='Importance' if 'Importance' in importance.columns else 'Correlation', 
                       y='Feature', 
                       data=importance.head(5), 
                       palette='viridis')
            ax.set_title("Most Influential Risk Factors")
        
        st.pyplot(fig)
