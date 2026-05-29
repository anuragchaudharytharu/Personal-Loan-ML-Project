#!/usr/bin/env python
"""
Simple Loan Approval Prediction UI with Gradio
Run with: python app_gradio.py
"""

import gradio as gr
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.mlproject.pipeline.prediction_pipeline import PredictionPipeline

# Initialize prediction pipeline
pipeline = PredictionPipeline()

def predict_loan(
    age, income, experience, family_size, education,
    mortgage, securities_account, cd_account, online_banking,
    credit_card, personal_loan, securities, cd_amount
):
    """Make prediction based on user inputs"""
    
    features = [
        age, income, experience, family_size, education,
        mortgage, securities_account, cd_account, online_banking,
        credit_card, personal_loan, securities, cd_amount
    ]
    
    result = pipeline.predict(features)
    
    if result['prediction'] == 1:
        decision = "✅ APPROVED"
        confidence_color = "green"
    else:
        decision = "❌ REJECTED"
        confidence_color = "red"
    
    confidence = result.get('confidence', 'N/A')
    probability = result.get('probability', 0)
    
    return f"""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: {confidence_color};">{decision}</h1>
        <h3>Confidence: {confidence}</h3>
        <p>Approval Probability: {probability*100:.1f}%</p>
        <p>Model: {result.get('model_used', 'Unknown')}</p>
    </div>
    """

# Create Gradio interface
iface = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Slider(18, 100, value=35, label="Age (years)"),
        gr.Slider(0, 500000, value=75000, label="Annual Income ($)"),
        gr.Slider(0, 50, value=5, label="Experience (years)"),
        gr.Slider(1, 10, value=2, label="Family Size"),
        gr.Dropdown([1, 2, 3], value=2, label="Education Level", info="1:Undergraduate, 2:Graduate, 3:Advanced"),
        gr.Slider(0, 500000, value=0, label="Mortgage Value ($)"),
        gr.Radio([0, 1], value=0, label="Has Securities Account?"),
        gr.Radio([0, 1], value=0, label="Has CD Account?"),
        gr.Radio([0, 1], value=1, label="Uses Online Banking?"),
        gr.Radio([0, 1], value=1, label="Has Credit Card?"),
        gr.Radio([0, 1], value=0, label="Existing Personal Loan?"),
        gr.Slider(0, 200000, value=0, label="Securities Amount ($)"),
        gr.Slider(0, 100000, value=0, label="CD Amount ($)"),
    ],
    outputs=gr.HTML(),
    title="🏦 Loan Approval Prediction System",
    description="Enter customer information to predict loan approval likelihood",
    theme="soft"
)

if __name__ == "__main__":
    iface.launch(share=True)