#!/usr/bin/env python
"""
Loan Approval Prediction UI
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

from src.mlproject.pipeline.prediction_pipeline import PredictionPipeline

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-approved {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .prediction-rejected {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize prediction pipeline
@st.cache_resource
def load_prediction_pipeline():
    try:
        pipeline = PredictionPipeline()
        return pipeline
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🏦 Loan Approval Prediction System</h1>
            <p>AI-Powered Decision Support for Personal Loan Applications</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    pipeline = load_prediction_pipeline()
    
    if pipeline is None:
        st.error("⚠️ Model not found. Please train the model first.")
        st.info("Run `python main.py` to train the model before using this UI.")
        return
    
    # Get feature info
    feature_info = pipeline.get_feature_info()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        if pipeline.metadata:
            st.metric("Best Model", pipeline.metadata.get('best_model_name', 'Unknown'))
            st.metric("F1 Score", f"{pipeline.metadata.get('best_model_f1_score', 0):.4f}")
            st.metric("Training Date", pipeline.metadata.get('training_date', 'Unknown')[:10])
        
        st.markdown("---")
        st.markdown("## 📝 Instructions")
        st.info("""
        1. Fill in the customer information
        2. Click "Predict" button
        3. View the prediction result
        4. Adjust values to explore different scenarios
        """)
        
        st.markdown("---")
        st.markdown("## 🎯 Quick Actions")
        if st.button("🔄 Reset to Default Values"):
            st.session_state.reset = True
            st.rerun()
        
        if st.button("📖 Load Sample Customer"):
            st.session_state.sample = True
            st.rerun()
    
    # Main content - Two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📋 Customer Information")
        
        # Create input fields
        with st.form(key="prediction_form"):
            # Personal Information
            st.markdown("### 👤 Personal Information")
            pers_col1, pers_col2, pers_col3 = st.columns(3)
            
            with pers_col1:
                age = st.number_input(
                    "Age (years)",
                    min_value=18,
                    max_value=100,
                    value=35,
                    help="Customer's age in years"
                )
            
            with pers_col2:
                income = st.number_input(
                    "Annual Income ($)",
                    min_value=0,
                    max_value=500000,
                    value=75000,
                    step=5000,
                    help="Annual income in USD"
                )
            
            with pers_col3:
                experience = st.number_input(
                    "Experience (years)",
                    min_value=0,
                    max_value=50,
                    value=5,
                    help="Years of professional experience"
                )
            
            # Family and Education
            st.markdown("### 👨‍👩‍👧‍👦 Family & Education")
            fam_col1, fam_col2 = st.columns(2)
            
            with fam_col1:
                family_size = st.number_input(
                    "Family Size",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Number of family members"
                )
            
            with fam_col2:
                education = st.selectbox(
                    "Education Level",
                    options=[1, 2, 3],
                    format_func=lambda x: {1: "Undergraduate", 2: "Graduate", 3: "Advanced"}[x],
                    help="Highest education level completed"
                )
            
            # Financial Information
            st.markdown("### 💰 Financial Information")
            fin_col1, fin_col2 = st.columns(2)
            
            with fin_col1:
                mortgage = st.number_input(
                    "Mortgage Value ($)",
                    min_value=0,
                    max_value=500000,
                    value=0,
                    step=10000,
                    help="Current mortgage value in USD"
                )
            
            with fin_col2:
                securities = st.number_input(
                    "Securities Amount ($)",
                    min_value=0,
                    max_value=200000,
                    value=0,
                    step=5000,
                    help="Amount in securities account"
                )
            
            # Account Information
            st.markdown("### 💳 Account Information")
            acc_col1, acc_col2, acc_col3 = st.columns(3)
            
            with acc_col1:
                securities_account = st.selectbox(
                    "Has Securities Account?",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Does the customer have a securities account?"
                )
            
            with acc_col2:
                cd_account = st.selectbox(
                    "Has CD Account?",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Does the customer have a certificate of deposit account?"
                )
            
            with acc_col3:
                cd_amount = st.number_input(
                    "CD Amount ($)",
                    min_value=0,
                    max_value=100000,
                    value=0,
                    step=5000,
                    help="Amount in certificate of deposit"
                )
            
            # Banking Behavior
            st.markdown("### 💻 Banking Behavior")
            bank_col1, bank_col2 = st.columns(2)
            
            with bank_col1:
                online_banking = st.selectbox(
                    "Uses Online Banking?",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Does the customer use online banking services?"
                )
            
            with bank_col2:
                credit_card = st.selectbox(
                    "Has Credit Card?",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Does the customer have a credit card with the bank?"
                )
            
            # Personal Loan Status
            personal_loan = st.selectbox(
                "Existing Personal Loan?",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Does the customer already have a personal loan?"
            )
            
            # Submit button
            st.markdown("---")
            submitted = st.form_submit_button(
                "🔮 Predict Loan Approval", 
                use_container_width=True,
                type="primary"
            )
    
    with col2:
        st.markdown("## 📊 Prediction Result")
        
        if submitted:
            # Prepare features
            features = [
                age, income, experience, family_size, education,
                mortgage, securities_account, cd_account, online_banking,
                credit_card, personal_loan, securities, cd_amount
            ]
            
            # Make prediction
            with st.spinner("Analyzing customer data..."):
                result = pipeline.predict(features)
            
            # Display result
            if result['prediction'] == 1:
                st.markdown(f"""
                    <div class="prediction-approved">
                        <h2>✅ LOAN APPROVED!</h2>
                        <p>The customer is likely to accept the personal loan.</p>
                        <p><strong>Confidence:</strong> {result.get('confidence', 'N/A')}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-rejected">
                        <h2>❌ LOAN REJECTED</h2>
                        <p>The customer is unlikely to accept the personal loan.</p>
                        <p><strong>Confidence:</strong> {result.get('confidence', 'N/A')}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Display additional info
            st.markdown("### 📈 Prediction Details")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Decision", result['class'])
            with col_m2:
                st.metric("Confidence", result.get('confidence', 'N/A'))
            
            if 'model_used' in result:
                st.info(f"🤖 Model: {result['model_used']}")
            
            # Risk meter
            if result.get('probability'):
                prob = result['probability']
                st.markdown("### ⚡ Risk Assessment")
                risk_level = "Low" if prob > 0.7 else "Medium" if prob > 0.4 else "High"
                risk_color = "#28a745" if prob > 0.7 else "#ffc107" if prob > 0.4 else "#dc3545"
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>Approval Probability: {prob*100:.1f}%</span>
                        <span style="color: {risk_color}; font-weight: bold;">{risk_level} Risk</span>
                    </div>
                    <div style="background-color: #e9ecef; border-radius: 10px; overflow: hidden;">
                        <div style="width: {prob*100}%; background-color: {risk_color}; height: 20px; transition: width 0.3s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.info("👈 Fill in the customer information and click 'Predict' to see the result.")
            
            # Feature importance placeholder
            st.markdown("### 🎯 Key Factors for Loan Approval")
            st.markdown("""
            <div class="info-box">
                <ul>
                    <li>💰 <strong>Income</strong> - Higher income increases approval chances</li>
                    <li>🏠 <strong>Mortgage</strong> - Lower mortgage debt is favorable</li>
                    <li>🎓 <strong>Education</strong> - Higher education level helps</li>
                    <li>💳 <strong>Banking Relationship</strong> - Existing accounts improve odds</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Additional analytics section
    st.markdown("---")
    st.markdown("## 📈 Decision Analytics")
    
    ana_col1, ana_col2, ana_col3 = st.columns(3)
    
    with ana_col1:
        st.markdown("### 💡 Tips for Better Approval")
        st.success("""
        - Maintain a good income-to-mortgage ratio
        - Build relationship with the bank (credit card, online banking)
        - Higher education level is favorable
        """)
    
    with ana_col2:
        st.markdown("### ⚠️ Risk Factors")
        st.warning("""
        - High mortgage relative to income
        - No banking relationship
        - Low income or experience
        """)
    
    with ana_col3:
        st.markdown("### 📊 Model Performance")
        if pipeline.metadata:
            st.info(f"""
            - Model: {pipeline.metadata.get('best_model_name', 'N/A')}
            - F1 Score: {pipeline.metadata.get('best_model_f1_score', 0):.3f}
            - Trained on historical loan data
            """)
        else:
            st.info("Model performance metrics not available")

if __name__ == "__main__":
    main()