import os
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

from mlproject.logging import logger
from mlproject.exception import CustomException
from mlproject.utils import load_object


class PredictionPipeline:
    """Prediction pipeline for making predictions with UI"""
    
    def __init__(self, model_path="artifacts/model_trainer/model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model and metadata"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.model = load_object(self.model_path)
            logger.info(f"Model loaded from: {self.model_path}")
            
            # Load metadata
            metadata_path = self.model_path.parent / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadata loaded from: {metadata_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise CustomException(e, sys)
    
    def predict(self, features):
        """
        Make prediction for a single sample
        
        Args:
            features: List of feature values
            
        Returns:
            dict: Prediction result
        """
        try:
            # Convert to numpy array
            input_array = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(input_array)[0]
            
            # Get probability
            probability = None
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(input_array)[0]
                prob_score = probability[1] if len(probability) > 1 else probability[0]
            else:
                prob_score = None
            
            result = {
                'prediction': int(prediction),
                'class': 'Approved' if prediction == 1 else 'Rejected',
                'probability': float(prob_score) if prob_score is not None else None,
                'confidence': f"{float(prob_score) * 100:.2f}%" if prob_score is not None else "N/A"
            }
            
            # Add model info if available
            if self.metadata:
                result['model_used'] = self.metadata.get('best_model_name', 'Unknown')
                result['model_score'] = self.metadata.get('best_model_f1_score', None)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise CustomException(e, sys)
    
    def predict_batch(self, features_list):
        """Make predictions for multiple samples"""
        try:
            input_array = np.array(features_list)
            predictions = self.model.predict(input_array)
            
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(input_array)
                probabilities = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            
            results = []
            for i, pred in enumerate(predictions):
                result = {
                    'prediction': int(pred),
                    'class': 'Approved' if pred == 1 else 'Rejected'
                }
                if probabilities is not None:
                    result['probability'] = float(probabilities[i])
                    result['confidence'] = f"{float(probabilities[i]) * 100:.2f}%"
                results.append(result)
            
            return results
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_feature_info(self):
        """Get information about expected features"""
        return {
            'feature_names': [
                'Age', 'Income', 'Experience', 'Family Size', 'Education',
                'Mortgage', 'Securities Account', 'CD Account', 'Online Banking',
                'Credit Card', 'Personal Loan', 'Securities', 'Certificate of Deposit'
            ],
            'feature_descriptions': {
                'Age': 'Age of the customer in years',
                'Income': 'Annual income in USD',
                'Experience': 'Years of professional experience',
                'Family Size': 'Number of family members',
                'Education': 'Education level (1: Undergraduate, 2: Graduate, 3: Advanced)',
                'Mortgage': 'Mortgage value in USD',
                'Securities Account': 'Does the customer have a securities account? (0: No, 1: Yes)',
                'CD Account': 'Does the customer have a certificate of deposit account? (0: No, 1: Yes)',
                'Online Banking': 'Does the customer use online banking? (0: No, 1: Yes)',
                'Credit Card': 'Does the customer have a credit card? (0: No, 1: Yes)',
                'Personal Loan': 'Already have a personal loan? (0: No, 1: Yes)',
                'Securities': 'Amount in securities in USD',
                'Certificate of Deposit': 'Amount in CD in USD'
            },
            'default_values': [35, 75000, 5, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0]
        }