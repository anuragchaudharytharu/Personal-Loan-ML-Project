import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from src.mlproject.pipeline.prediction_pipeline import PredictionPipeline

st.set_page_config(page_title="Loan Predictor", page_icon="🏦")

@st.cache_resource
def load_model():
    try:
        return PredictionPipeline()
    except:
        return None

pipeline = load_model()

if not pipeline:
    st.warning("Run 'python main.py' first")
    st.stop()

st.title("🏦 Loan Approval")

with st.form("f"):
    c1, c2 = st.columns(2)
    
    with c1:
        age = st.number_input("Age", 18, 100, 35)
        income = st.number_input("Income", 0, 500000, 75000)
        exp = st.number_input("Experience", 0, 50, 5)
        family = st.number_input("Family", 1, 10, 2)
        edu = st.selectbox("Education", [1, 2, 3], format_func=lambda x: {1:"Grad",2:"PG",3:"Advanced"}[x])
    
    with c2:
        mortgage = st.number_input("Mortgage", 0, 500000, 0)
        securities = st.number_input("Securities Amt", 0, 200000, 0)
        cd = st.number_input("CD Amt", 0, 100000, 0)
        sec_acc = st.checkbox("Securities Acc")
        cd_acc = st.checkbox("CD Acc")
        online = st.checkbox("Online")
        credit = st.checkbox("Credit Card")
    
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    features = [age, income, exp, family, edu, mortgage, 
                int(sec_acc), int(cd_acc), int(online), int(credit), 
                0, securities, cd]
    
    result = pipeline.predict(features)
    
    if result['prediction'] == 1:
        st.success(f"✅ APPROVED ({result.get('confidence', 'N/A')})")
    else:
        st.error(f"❌ REJECTED ({result.get('confidence', 'N/A')})")