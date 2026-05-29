from mlproject.config import ConfigurationManager
from mlproject.components.model_evaluation import ModelEvaluation
from mlproject.logging import logger
from mlproject.exception import CustomException

import sys


class ModelEvaluationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):
        try:
            logger.info("="*50)
            logger.info("Starting Model Evaluation Pipeline")
            logger.info("="*50)

            # Get configuration
            config_manager = ConfigurationManager()
            model_evaluation_config = config_manager.get_model_evaluation_config()
            params = config_manager.params

            logger.info(f"Model evaluation config loaded")
            logger.info(f"Model path: {model_evaluation_config.model_path}")
            logger.info(f"Test data path: {model_evaluation_config.test_data_path}")
            logger.info(f"Evaluation output directory: {model_evaluation_config.root_dir}")

            # Initialize model evaluation
            model_evaluation = ModelEvaluation(
                config=model_evaluation_config,
                params=params
            )

            # Run model evaluation
            metrics, evaluation_path = model_evaluation.initiate_model_evaluation()

            logger.info("="*50)
            logger.info(f"✅ Model Evaluation Pipeline Completed Successfully")
            logger.info(f"📊 Final F1 Score: {metrics['f1_score']:.4f}")
            logger.info(f"📊 Final Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"📁 Results saved to: {evaluation_path}")
            logger.info("="*50)

            return metrics, evaluation_path

        except Exception as e:
            logger.error(f"Model Evaluation Pipeline failed: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = ModelEvaluationTrainingPipeline()
    metrics, results_path = pipeline.main()
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS:")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*50}")