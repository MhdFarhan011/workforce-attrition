import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page configuration
st.set_page_config(page_title="HR Attrition Risk Predictor Dashboard", layout="wide")

st.title("🚀 HR Candidate Job Change & Attrition Predictor")
st.write("An end-to-end intelligent platform with executive overview, driver analysis, batch insights, and live single-candidate prediction inputs.")

# 1. Load Model, Training Data, and Default Test Data
@st.cache_resource
def load_assets():
    model = joblib.load('best_xgb_model.pkl')
    train_df = pd.read_csv('aug_train.csv')
    if 'enrollee_id' in train_df.columns: train_df = train_df.drop(columns=['enrollee_id'])
    if 'target' in train_df.columns: train_df = train_df.drop(columns=['target'])
    
    test_df = None
    if os.path.exists('aug_test.csv'):
        test_df = pd.read_csv('aug_test.csv')
    return model, train_df, test_df

model, train_df, test_df = load_assets()

# Create Tabs for Navigation
tab1, tab2 = st.tabs(["📊 Batch Dataset Dashboard & Drivers", "👤 Interactive Single Candidate Prediction"])

# ==========================================
# TAB 1: BATCH DASHBOARD & DRIVER ANALYSIS
# ==========================================
with tab1:
    st.subheader("Pre-loaded Dataset Insights & Overview")
    
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
        
        # --- Overview Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("High Risk Candidates", len(output_results[output_results['risk_tier'] == 'High Risk']))
        col2.metric("Medium Risk Candidates", len(output_results[output_results['risk_tier'] == 'Medium Risk']))
        col3.metric("Low Risk Candidates", len(output_results[output_results['risk_tier'] == 'Low Risk']))
        
        # --- Visualizations ---
        st.write("---")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**Candidate Risk Tier Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=output_results, x='risk_tier', order=['Low Risk', 'Medium Risk', 'High Risk'], palette='coolwarm', ax=ax)
            ax.set_xlabel("Risk Tier")
            ax.set_ylabel("Number of Candidates")
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
            st.pyplot(fig)
            
        with col_chart2:
            st.markdown("**Attrition Probability Histogram**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(output_results['attrition_probability'], bins=20, kde=True, color='teal', ax=ax2)
            ax2.set_xlabel("Attrition Probability Score")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)

        # --- Driver Analysis ---
        st.write("---")
        st.markdown("**Model Driver Analysis (Top Attrition Features)**")
        feature_names = X_test_encoded.columns
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
        
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1], color='coral')
        ax3.set_xlabel("Relative Importance Score")
        ax3.set_ylabel("Feature Name")
        st.pyplot(fig3)

        # --- Detailed Table & Download ---
        st.write("---")
        st.markdown("**Full Scored Predictions Table**")
        st.dataframe(output_results)
        
        csv = output_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Scored Results as CSV",
            data=csv,
            file_name='scored_candidates_app.csv',
            mime='text/csv',
        )
    else:
        st.warning("`test.csv` not found in repository.")

# ==========================================
# TAB 2: INTERACTIVE SINGLE CANDIDATE INPUT
# ==========================================
with tab2:
    st.subheader("Test Individual Candidate Profiles")
    st.write("Input specific candidate metrics below to generate a real-time attrition prediction and actionable retention insights.")

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            input_city = st.selectbox("City Code", [c for c in train_df['city'].unique()]) if 'city' in train_df.columns else "city_103"
            input_cdi = st.slider("City Development Index (CDI)", 0.40, 1.00, 0.920, 0.001)
            input_exp_rel = st.selectbox("Relevant Experience", ["Has relevent experience", "No relevent experience"])
            input_ed_level = st.selectbox("Education Level", train_df['education_level'].dropna().unique() if 'education_level' in train_df.columns else ["Graduate"])
            
        with col_b:
            input_company_type = st.selectbox("Company Type", train_df['company_type'].dropna().unique() if 'company_type' in train_df.columns else ["Pvt Ltd"])
            input_last_job = st.selectbox("Last New Job", train_df['last_new_job'].dropna().unique() if 'last_new_job' in train_df.columns else ["1"])
            input_training_hours = st.slider("Training Hours", 1, 350, 25)
            input_experience = st.selectbox("Total Experience (Years)", train_df['experience'].dropna().unique() if 'experience' in train_df.columns else ["5"])

        submit_button = st.form_submit_button(label="Predict Candidate Risk")

    if submit_button:
        # Construct single row dataframe
        single_df = pd.DataFrame([{
            'city': input_city,
            'city_development_index': input_cdi,
            'relevent_experience': input_exp_rel,
            'education_level': input_ed_level,
            'company_type': input_company_type,
            'last_new_job': input_last_job,
            'training_hours': input_training_hours,
            'experience': input_experience
        }])
        
        # Align with training features
        combined_single = pd.concat([train_df.drop(columns=[col for col in train_df.columns if col not in single_df.columns], errors='ignore'), single_df], axis=0, ignore_index=True)
        # To ensure full matching structure with train columns:
        dummy_combined = pd.concat([train_df, single_df], axis=0, ignore_index=True)
        encoded_single_combined = pd.get_dummies(dummy_combined)
        X_single_encoded = encoded_single_combined.iloc[len(train_df):]
        
        # Predict probability
        prob = model.predict_proba(X_single_encoded)[:, 1][0]
        
        # Determine tier & generate tailored insights
        if prob >= 0.25:
            tier = "High Risk 🔴"
            insight = "This candidate has a high probability of looking for a job change. **Actionable HR Advice:** Schedule a retention check-in, review compensation competitiveness, and discuss clear internal career pathways immediately."
        elif prob >= 0.15:
            tier = "Medium Risk 🟡"
            insight = "This candidate shows moderate flight risk. **Actionable HR Advice:** Ensure regular engagement, monitor project satisfaction, and offer continuous upskilling options."
        else:
            tier = "Low Risk 🟢"
            insight = "This candidate is stable and unlikely to leave in the near term. **Actionable HR Advice:** Maintain standard engagement and standard performance tracking."

        st.write("---")
        st.subheader("🎯 Prediction Results & Strategic Insights")
        
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Calculated Attrition Probability", f"{prob*100:.1f}%")
        res_col2.metric("Assigned Risk Tier", tier)
        
        st.info(insight)
