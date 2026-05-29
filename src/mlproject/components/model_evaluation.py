import os
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

from mlproject.logging import logger
from mlproject.exception import CustomException
from mlproject.utils import save_object, load_object


class ModelEvaluation:
    def __init__(self, config, params):
        """
        config: ModelEvaluationConfig
        params: params.yaml dict
        """
        self.config = config
        self.params = params
    
    def load_data(self):
        """Load test data"""
        try:
            test_arr = np.load(self.config.test_data_path)
            
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]
            
            logger.info(f"Test data loaded: {X_test.shape}, {y_test.shape}")
            return X_test, y_test
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_model(self):
        """Load the trained model from model_trainer directory"""
        try:
            # Convert to Path if it's a string
            model_path = self.config.model_path
            if isinstance(model_path, str):
                model_path = Path(model_path)
            
            model = load_object(model_path)
            logger.info(f"Model loaded from: {model_path}")
            return model
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_metadata(self):
        """Load model metadata from model_trainer directory"""
        try:
            model_path = self.config.model_path
            if isinstance(model_path, str):
                model_path = Path(model_path)
            
            metadata_path = str(model_path).replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Metadata loaded from: {metadata_path}")
                return metadata
            else:
                logger.warning("No metadata file found")
                return None
        except Exception as e:
            logger.warning(f"Could not load metadata: {str(e)}")
            return None
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate all evaluation metrics"""
        try:
            metrics = {
                'f1_score': float(f1_score(y_true, y_pred)),
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred)),
                'recall': float(recall_score(y_true, y_pred)),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
            
            logger.info(f"Calculated metrics: F1={metrics['f1_score']:.4f}, "
                       f"Accuracy={metrics['accuracy']:.4f}, "
                       f"Precision={metrics['precision']:.4f}, "
                       f"Recall={metrics['recall']:.4f}")
            
            return metrics
        except Exception as e:
            raise CustomException(e, sys)
    
    def compare_with_baseline(self, metrics, baseline_metrics=None):
        """Compare current model performance with baseline"""
        try:
            if baseline_metrics is None:
                # Load baseline from config if available
                baseline_metrics = {
                    'f1_score': getattr(self.config, 'baseline_f1_score', 0.8),
                    'accuracy': getattr(self.config, 'baseline_accuracy', 0.8)
                }
            
            comparison = {
                'f1_score_improvement': metrics['f1_score'] - baseline_metrics['f1_score'],
                'accuracy_improvement': metrics['accuracy'] - baseline_metrics['accuracy'],
                'passes_baseline': metrics['f1_score'] >= baseline_metrics['f1_score']
            }
            
            logger.info(f"Comparison with baseline (F1={baseline_metrics['f1_score']:.4f}): "
                       f"Improvement={comparison['f1_score_improvement']:.4f}, "
                       f"Passes baseline={comparison['passes_baseline']}")
            
            return comparison
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_evaluation_results(self, metrics, comparison, metadata=None):
        """Save evaluation results to model_evaluation directory"""
        try:
            # Create evaluation directory if it doesn't exist
            eval_dir = Path(self.config.root_dir)
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON in model_evaluation directory
            evaluation_path = eval_dir / 'evaluation_results.json'
            with open(evaluation_path, 'w') as f:
                json.dump({
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'model_path': str(self.config.model_path),
                    'metrics': metrics,
                    'comparison_with_baseline': comparison,
                    'model_metadata': metadata
                }, f, indent=4)
            logger.info(f"Evaluation results saved to: {evaluation_path}")
            
            # Save metrics as CSV
            metrics_df = pd.DataFrame([
                {'metric': 'f1_score', 'value': metrics['f1_score']},
                {'metric': 'accuracy', 'value': metrics['accuracy']},
                {'metric': 'precision', 'value': metrics['precision']},
                {'metric': 'recall', 'value': metrics['recall']}
            ])
            
            metrics_csv_path = eval_dir / 'metrics.csv'
            metrics_df.to_csv(metrics_csv_path, index=False)
            logger.info(f"Metrics saved to CSV: {metrics_csv_path}")
            
            # Save confusion matrix as CSV
            cm_df = pd.DataFrame(metrics['confusion_matrix'])
            cm_csv_path = eval_dir / 'confusion_matrix.csv'
            cm_df.to_csv(cm_csv_path, index=False)
            logger.info(f"Confusion matrix saved to: {cm_csv_path}")
            
            # Save classification report as CSV
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            report_csv_path = eval_dir / 'classification_report.csv'
            report_df.to_csv(report_csv_path)
            logger.info(f"Classification report saved to: {report_csv_path}")
            
            return evaluation_path
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_evaluation(self):
        """Main method to evaluate the trained model"""
        try:
            logger.info("="*50)
            logger.info("Starting Model Evaluation")
            logger.info("="*50)
            
            # Load test data
            X_test, y_test = self.load_data()
            
            # Load trained model
            model = self.load_model()
            
            # Load metadata if available
            metadata = self.load_metadata()
            
            # Make predictions
            y_pred = model.predict(X_test)
            logger.info(f"Predictions made for {len(y_pred)} samples")
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # Compare with baseline
            comparison = self.compare_with_baseline(metrics)
            
            # Save evaluation results
            evaluation_path = self.save_evaluation_results(metrics, comparison, metadata)
            
            # Log summary
            logger.info("="*50)
            logger.info("Model Evaluation Summary:")
            logger.info(f"  - F1 Score: {metrics['f1_score']:.4f}")
            logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  - Precision: {metrics['precision']:.4f}")
            logger.info(f"  - Recall: {metrics['recall']:.4f}")
            logger.info(f"  - Passes Baseline: {comparison['passes_baseline']}")
            
            if metadata:
                logger.info(f"  - Best Model from Training: {metadata.get('best_model_name', 'Unknown')}")
                logger.info(f"  - Training F1 Score: {metadata.get('best_model_f1_score', 0):.4f}")
                logger.info(f"  - Test F1 Score: {metrics['f1_score']:.4f}")
                logger.info(f"  - Performance Gap: {metrics['f1_score'] - metadata.get('best_model_f1_score', 0):.4f}")
            
            logger.info("="*50)
            logger.info(f"✅ Model Evaluation Completed Successfully")
            logger.info(f"📁 Results saved to: {self.config.root_dir}")
            logger.info("="*50)
            
            return metrics, evaluation_path
            
        except Exception as e:
            raise CustomException(e, sys)