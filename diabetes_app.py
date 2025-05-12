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
from sklearn.inspection import permutation_importance

# ======================================
# 1. DATA LOADING AND CLEANING
# ======================================
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    url = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    return pd.read_csv(url)

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
    fig, ax = plt.subplots(figsize=(8,4))
    counts = df['Diabetes_binary'].value_counts()
    bars = ax.bar(counts.index, counts.values, color=['#1f77b4', '#ff7f0e'])
    ax.set_xticks([0,1])
    ax.set_xticklabels(['No Diabetes', 'Diabetes'], rotation=0)
    ax.set_ylabel('Count')
    
    # Add percentage labels
    total = len(df)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height/total:.1%}',
                ha='center', va='bottom')
    
    st.pyplot(fig)
    
    st.subheader("Feature Correlation with Diabetes")
    corr = df.corr()[['Diabetes_binary']].sort_values('Diabetes_binary', ascending=False)
    st.dataframe(corr.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))
    
    st.subheader("Correlation Heatmap (Top 15 Features)")
    top_features = corr.index[1:16]  # Exclude Diabetes_binary itself
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(df[top_features].corr(), ax=ax, cmap='coolwarm', annot=True, fmt='.1f')
    st.pyplot(fig)
    
    st.subheader("Feature Distributions by Diabetes Status")
    num_features = ['BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth']
    for feat in num_features:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.boxplot(x='Diabetes_binary', y=feat, data=df, ax=ax, palette=['#1f77b4', '#ff7f0e'])
        ax.set_title(f"{feat} Distribution")
        ax.set_xticklabels(['No Diabetes', 'Diabetes'])
        st.pyplot(fig)

# ======================================
# 3. CONFUSION MATRIX VISUALIZATION
# ======================================
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False):
    """Plot a confusion matrix with improved visualization"""
    cm = confusion_matrix(y_true, y_pred)
    title = 'Confusion Matrix'
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized ' + title
    
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', ax=ax,
                xticklabels=classes, yticklabels=classes)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    # Adjust layout to prevent cutoff
    plt.tight_layout()
    return fig

# ======================================
# 4. DIMENSIONALITY REDUCTION
# ======================================
def perform_dimensionality_reduction(X_train, y_train, X_test=None):
    """Performs PCA and feature selection with proper train/test separation"""
    
    st.subheader("Dimensionality Reduction")
    
    # PCA Analysis
    st.markdown("#### Principal Component Analysis (PCA)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    pca = PCA().fit(X_train_scaled)
    
    # Plot explained variance
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.legend()
    st.pyplot(fig)
    
    n_components = st.slider("Select number of PCA components", 
                            min_value=1, 
                            max_value=min(20, X_train.shape[1]), 
                            value=5)
    
    # Feature Selection Methods
    st.markdown("#### Feature Selection Methods")
    method = st.radio("Select feature selection method:",
                     ["PCA", "Forward Selection", "Backward Selection", "Feature Importance"])
    
    if method == "PCA":
        pca = PCA(n_components=n_components)
        X_train_transformed = pca.fit_transform(X_train_scaled)
        if X_test is not None:
            X_test_transformed = pca.transform(scaler.transform(X_test))
        else:
            X_test_transformed = None
            
        st.info(f"Selected {n_components} principal components explaining {np.sum(pca.explained_variance_ratio_):.1%} of variance")
        return X_train_transformed, X_test_transformed, pca
    
    elif method in ["Forward Selection", "Backward Selection"]:
        direction = 'forward' if method == "Forward Selection" else 'backward'
        n_features = st.slider(f"Number of features to select ({method})", 
                              min_value=1, 
                              max_value=min(20, X_train.shape[1]), 
                              value=5)
        
        # Use a simpler estimator for feature selection
        sfs = SequentialFeatureSelector(
            RandomForestClassifier(n_estimators=50, random_state=42),
            n_features_to_select=n_features,
            direction=direction,
            cv=5
        )
        
        with st.spinner(f'Running {method}... This may take a while'):
            sfs.fit(X_train, y_train)
        
        selected_features = X_train.columns[sfs.get_support()]
        st.success(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")
        
        X_train_transformed = X_train[selected_features]
        X_test_transformed = X_test[selected_features] if X_test is not None else None
        
        return X_train_transformed, X_test_transformed, sfs
    
    else:  # Feature Importance
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        n_features = st.slider("Number of important features to select", 
                              min_value=1, 
                              max_value=min(20, X_train.shape[1]), 
                              value=5)
        
        selected_features = importance['Feature'].head(n_features).values
        
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x='Importance', y='Feature', data=importance.head(n_features), palette='viridis')
        ax.set_title(f"Top {n_features} Most Important Features")
        st.pyplot(fig)
        
        X_train_transformed = X_train[selected_features]
        X_test_transformed = X_test[selected_features] if X_test is not None else None
        
        return X_train_transformed, X_test_transformed, model

# ======================================
# 5. MODEL TRAINING AND EVALUATION
# ======================================
def train_model(X_train, y_train, model_type='SVM'):
    """Creates and trains a model pipeline"""
    
    if model_type == 'SVM':
        pipeline = make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                LinearSVC(dual=False, random_state=42, max_iter=2000),
                cv=5
            )
        )
    elif model_type == 'Random Forest':
        pipeline = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    with st.spinner(f'Training {model_type} model...'):
        pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with comprehensive metrics"""
    
    st.subheader("Model Evaluation")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Precision", f"{report['1']['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{report['1']['recall']:.2%}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm_tab1, cm_tab2 = st.tabs(["Counts", "Normalized"])
    
    with cm_tab1:
        fig = plot_confusion_matrix(y_test, y_pred, classes=['No Diabetes', 'Diabetes'])
        st.pyplot(fig)
    
    with cm_tab2:
        fig = plot_confusion_matrix(y_test, y_pred, classes=['No Diabetes', 'Diabetes'], normalize=True)
        st.pyplot(fig)
    
    # Classification Report
    st.subheader("Detailed Classification Report")
    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))
    
    # ROC Curve if probabilities are available
    if y_proba is not None:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)

# ======================================
# 6. MAIN APP
# ======================================
def main():
    st.set_page_config(layout="wide", page_title="Diabetes Risk Predictor", page_icon="🩺")
    st.title("🩺 Advanced Diabetes Risk Predictor")
    
    # Add documentation
    with st.expander("How to use this app"):
        st.markdown("""
        1. **Explore the Data**: Understand the dataset in the EDA section
        2. **Feature Selection**: Choose dimensionality reduction method
        3. **Model Training**: Train and evaluate the model
        4. **Predictions**: Make predictions using interactive sliders
        """)
    
    # Load and clean data
    df = load_data()
    df_clean = clean_data(df)
    
    # Split data early to avoid leakage
    X = df_clean.drop('Diabetes_binary', axis=1)
    y = df_clean['Diabetes_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Feature Selection", "Model Training", "Prediction"])
    
    with tab1:
        st.header("1. Exploratory Data Analysis")
        perform_eda(df_clean)
    
    with tab2:
        st.header("2. Dimensionality Reduction & Feature Selection")
        X_train_reduced, X_test_reduced, reducer = perform_dimensionality_reduction(X_train, y_train, X_test)
    
    with tab3:
        st.header("3. Model Training")
        
        model_type = st.radio("Select Model Type", 
                            ["SVM", "Random Forest"],
                            horizontal=True)
        
        if st.button("Train Model"):
            model = train_model(X_train_reduced, y_train, model_type)
            
            # Save model and reducer
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/diabetes_model.joblib")
            joblib.dump(reducer, "models/feature_reducer.joblib")
            joblib.dump(X_train.columns, "models/feature_names.joblib")  # Save original feature names
            
            st.success("Model trained successfully!")
            evaluate_model(model, X_test_reduced, y_test)
    
    with tab4:
        st.header("4. Risk Prediction")
        
        # Check if model exists
        if not all(os.path.exists(f"models/{f}") for f in ["diabetes_model.joblib", "feature_reducer.joblib"]):
            st.warning("Please train a model first in the 'Model Training' tab")
            st.stop()
        
        # Load artifacts
        model = joblib.load("models/diabetes_model.joblib")
        reducer = joblib.load("models/feature_reducer.joblib")
        feature_names = joblib.load("models/feature_names.joblib")
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            # Tree-based model
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
        elif hasattr(model, 'coef_'):
            # Linear model
            coefs = model.named_steps['calibratedclassifiercv'].estimators_[0].coef_[0] if hasattr(model, 'named_steps') else model.coef_[0]
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefs
            }).sort_values('Coefficient', key=abs, ascending=False)
        else:
            # Fallback to correlation
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Correlation': df_clean[feature_names].corrwith(y)
            }).sort_values('Correlation', key=abs, ascending=False)
        
        top_features = importance.head(5)['Feature'].tolist()
        
        # Create input widgets
        st.subheader("Adjust Risk Factors")
        inputs = {}
        cols = st.columns(2)
        
        # Create sliders for top features
        for i, feat in enumerate(top_features):
            with cols[i%2]:
                if df_clean[feat].nunique() <= 5:  # Categorical feature
                    unique_vals = sorted(df_clean[feat].unique())
                    inputs[feat] = st.selectbox(
                        label=feat,
                        options=unique_vals,
                        index=len(unique_vals)//2,
                        help=f"Select value for {feat}"
                    )
                else:  # Numerical feature
                    inputs[feat] = st.slider(
                        label=feat,
                        min_value=float(df_clean[feat].min()),
                        max_value=float(df_clean[feat].max()),
                        value=float(df_clean[feat].median()),
                        step=0.1 if df_clean[feat].dtype == float else 1.0,
                        help=f"Range: {df_clean[feat].min()} to {df_clean[feat].max()}"
                    )
        
        # Fill remaining features with median values
        for feat in feature_names:
            if feat not in inputs:
                if df_clean[feat].nunique() <= 5:
                    inputs[feat] = df_clean[feat].mode()[0]
                else:
                    inputs[feat] = df_clean[feat].median()
        
        if st.button("Predict Diabetes Risk"):
            # Create input dataframe
            input_df = pd.DataFrame([inputs], columns=feature_names)
            
            # Transform input
            try:
                if isinstance(reducer, PCA):
                    scaled_input = StandardScaler().fit_transform(input_df)
                    transformed_input = reducer.transform(scaled_input)
                elif hasattr(reducer, 'transform'):
                    transformed_input = reducer.transform(input_df)
                else:  # Feature selection
                    transformed_input = input_df[importance.head(len(reducer.feature_importances_))['Feature']]
                
                # Get prediction
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(transformed_input)[0][1]
                else:
                    proba = model.decision_function(transformed_input)[0]
                    proba = 1 / (1 + np.exp(-proba))  # Sigmoid transform
                
                # Display results
                st.subheader("Prediction Results")
                col1, col2 = st.columns([1,2])
                
                with col1:
                    # Gauge visualization
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.set_xlim(0,1)
                    ax.set_ylim(0,1)
                    ax.axis('off')
                    
                    # Draw gauge
                    theta = 180 * proba
                    ax.fill_between([0,1], [0,0], [1,1], color='lightgray')
                    ax.fill_between([0, proba], [0,0], [1,1], 
                                  color='red' if proba > 0.5 else 'orange' if proba > 0.3 else 'green')
                    
                    # Add text
                    ax.text(0.5, 0.7, f"{proba:.1%} Risk", 
                           ha='center', va='center', fontsize=20)
                    ax.text(0.5, 0.3, "High Risk" if proba > 0.5 else "Medium Risk" if proba > 0.3 else "Low Risk",
                           ha='center', va='center', fontsize=16)
                    
                    st.pyplot(fig)
                
                with col2:
                    # Risk interpretation
                    if proba > 0.7:
                        st.error("**High Risk** - Consult a doctor immediately")
                        st.markdown("""
                        **Recommendations:**
                        - Schedule a doctor's appointment
                        - Monitor blood sugar levels regularly
                        - Consider lifestyle changes
                        """)
                    elif proba > 0.4:
                        st.warning("**Moderate Risk** - Monitor your health")
                        st.markdown("""
                        **Recommendations:**
                        - Increase physical activity
                        - Improve diet (reduce sugar intake)
                        - Consider regular checkups
                        """)
                    else:
                        st.success("**Low Risk** - Maintain healthy habits")
                        st.markdown("""
                        **Recommendations:**
                        - Continue balanced diet
                        - Regular exercise
                        - Annual health checkups
                        """)
                
                # Show feature contributions
                st.subheader("How Each Factor Affects Your Risk")
                
                if 'Coefficient' in importance.columns:
                    # For linear models
                    coef_df = importance.head(5).copy()
                    coef_df['Absolute Effect'] = coef_df['Coefficient'].abs()
                    coef_df['Direction'] = coef_df['Coefficient'].apply(lambda x: "Increases Risk" if x > 0 else "Decreases Risk")
                    
                    fig, ax = plt.subplots(figsize=(10,5))
                    sns.barplot(x='Coefficient', y='Feature', data=coef_df,
                               palette=['red' if x > 0 else 'green' for x in coef_df['Coefficient']])
                    ax.set_title("Feature Effects on Diabetes Risk")
                    ax.set_xlabel("Coefficient Magnitude (Positive = Increases Risk)")
                    st.pyplot(fig)
                    
                    # Show how user's values compare to average
                    st.markdown("**Your values compared to population averages:**")
                    comp_df = pd.DataFrame({
                        'Feature': coef_df['Feature'],
                        'Your Value': [inputs[f] for f in coef_df['Feature']],
                        'Population Average': [df_clean[f].mean() for f in coef_df['Feature']],
                        'Effect': coef_df['Direction']
                    })
                    st.dataframe(comp_df.style.applymap(
                        lambda x: 'color: red' if "Increases" in str(x) else 'color: green',
                        subset=['Effect']
                    ))
                
                else:
                    # For other models
                    fig, ax = plt.subplots(figsize=(10,5))
                    sns.barplot(x='Importance' if 'Importance' in importance.columns else 'Correlation',
                              y='Feature',
                              data=importance.head(5),
                              palette='viridis')
                    ax.set_title("Most Influential Risk Factors")
                    st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
