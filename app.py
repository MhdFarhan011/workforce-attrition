import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page configuration
st.set_page_config(page_title="HR Attrition Risk Predictor Dashboard", layout="wide")

st.title("🚀 HR Candidate Job Change & Attrition Predictor")
st.write("An end-to-end intelligent platform providing overview metrics, risk predictions, and model driver analysis using pre-loaded test data.")

# 1. Load Model, Training Data, and Default Test Data
@st.cache_resource
def load_assets():
    model = joblib.load('best_xgb_model.pkl')
    train_df = pd.read_csv('aug_train.csv')
    if 'enrollee_id' in train_df.columns: train_df = train_df.drop(columns=['enrollee_id'])
    if 'target' in train_df.columns: train_df = train_df.drop(columns=['target'])
    
    # Check if test.csv is already available locally
    test_df = None
    if os.path.exists('aug_test.csv'):
        test_df = pd.read_csv('aug_test.csv')
    return model, train_df, test_df

model, train_df, test_df = load_assets()

# 2. Handle Default vs Uploaded Data
if test_df is not None:
    st.success("✅ Pre-loaded test dataset (`aug_test.csv`) detected automatically!")
else:
    st.warning("⚠️ `aug_test.csv` not found in the directory. Please upload your test dataset below:")
    uploaded_file = st.file_uploader("Upload CSV file for scoring", type=["csv"])
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)

if test_df is not None:
    inference_df = test_df.copy()
    if 'enrollee_id' in inference_df.columns:
        inference_df = inference_df.drop(columns=['enrollee_id'])
    if 'target' in inference_df.columns:
        inference_df = inference_df.drop(columns=['target'], errors='ignore')
        
    # Align columns using the training dataset format
    combined_df = pd.concat([train_df, inference_df], axis=0, ignore_index=True)
    encoded_combined = pd.get_dummies(combined_df)
    X_test_encoded = encoded_combined.iloc[len(train_df):]
    
    # Predict
    test_predictions = model.predict(X_test_encoded)
    test_probabilities = model.predict_proba(X_test_encoded)[:, 1]
    
    # Assign Risk Tiers
    def assign_risk_tier(prob):
        if prob >= 0.25:
            return 'High Risk'
        elif prob >= 0.15:
            return 'Medium Risk'
        else:
            return 'Low Risk'
            
    output_results = test_df.copy()
    output_results['predicted_target'] = test_predictions
    output_results['attrition_probability'] = test_probabilities
    output_results['risk_tier'] = output_results['attrition_probability'].apply(assign_risk_tier)
    
    # --- SECTION 1: OVERVIEW METRICS ---
    st.write("---")
    st.write("### 📊 Overview & Executive Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("High Risk Candidates", len(output_results[output_results['risk_tier'] == 'High Risk']))
    col2.metric("Medium Risk Candidates", len(output_results[output_results['risk_tier'] == 'Medium Risk']))
    col3.metric("Low Risk Candidates", len(output_results[output_results['risk_tier'] == 'Low Risk']))
    
    # --- SECTION 2: PREDICTION VISUALIZATIONS ---
    st.write("---")
    st.write("### 📈 Prediction Visualizations")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Candidate Risk Tier Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=output_results, x='risk_tier', order=['Low Risk', 'Medium Risk', 'High Risk'], palette='coolwarm', ax=ax)
        ax.set_title("Count by Risk Tier", fontweight='bold')
        ax.set_xlabel("Risk Tier")
        ax.set_ylabel("Number of Candidates")
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
        st.pyplot(fig)
        
    with col_chart2:
        st.subheader("Attrition Probability Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(output_results['attrition_probability'], bins=20, kde=True, color='teal', ax=ax2)
        ax2.set_title("Probability Histogram", fontweight='bold')
        ax2.set_xlabel("Attrition Probability Score")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    # --- SECTION 3: DRIVER ANALYSIS ---
    st.write("---")
    st.write("### 🔍 Model Driver Analysis (Feature Importance)")
    st.write("Understanding what primary factors influence the model's decision-making process across features.")
    
    feature_names = X_test_encoded.columns
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1], color='coral')
    ax3.set_title("Top 10 Features Driving Attrition", fontweight='bold', fontsize=12)
    ax3.set_xlabel("Relative Importance Score")
    ax3.set_ylabel("Feature Name")
    st.pyplot(fig3)

    # --- SECTION 4: FULL DATA TABLE & DOWNLOAD ---
    st.write("---")
    st.write("### 🗂️ Detailed Scored Predictions Table")
    st.dataframe(output_results)
    
    csv = output_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Scored Results as CSV",
        data=csv,
        file_name='scored_candidates_app.csv',
        mime='text/csv',
    )
