from mlproject.exception import CustomException
import sys
from mlproject.logging import logger
from mlproject.pipeline import (
    DataIngestionTrainingPipeline,
    DataValidationTrainingPipeline,
    DataTransformationTrainingPipeline,
    ModelTrainerTrainingPipeline,
    ModelEvaluationTrainingPipeline
)


def run_pipeline(stage_name, obj, *args):
    try:
        logger.info("="*50)
        logger.info(">>>>>>> %s STARTED <<<<<<<", stage_name)
        logger.info("="*50)

        result = obj.main(*args)

        logger.info("="*50)
        logger.info(">>>>>>> %s COMPLETED <<<<<<<", stage_name)
        logger.info("="*50)

        return result

    except Exception as e:
        logger.error(f"Pipeline failed at {stage_name}")
        raise CustomException(e, sys)


if __name__ == "__main__":

    try:
        # ---------------- INGESTION ----------------
        train_path, test_path = run_pipeline(
            "Data Ingestion Stage",
            DataIngestionTrainingPipeline()
        )
        logger.info(f"Train path: {train_path}")
        logger.info(f"Test path: {test_path}")

        # ---------------- VALIDATION ----------------
        validation_status = run_pipeline(
            "Data Validation Stage",
            DataValidationTrainingPipeline(),
            train_path,
            test_path
        )

        # stop pipeline if validation fails
        if not validation_status:
            logger.error("Data Validation Failed. Stopping pipeline.")
            raise Exception("Pipeline stopped due to validation failure")
        
        logger.info(f"Validation Status: {validation_status}")

        # ---------------- TRANSFORMATION ----------------
        train_arr, test_arr = run_pipeline(
            "Data Transformation Stage",
            DataTransformationTrainingPipeline(),
            train_path,
            test_path
        )

        logger.info("Data Transformation Completed Successfully")
        logger.info(f"Train array shape: {train_arr.shape if hasattr(train_arr, 'shape') else len(train_arr)}")
        logger.info(f"Test array shape: {test_arr.shape if hasattr(test_arr, 'shape') else len(test_arr)}")

        # ---------------- MODEL TRAINER ----------------
        best_model_name, best_score = run_pipeline(
            "Model Trainer Stage",
            ModelTrainerTrainingPipeline()
        )

        # ---------------- MODEL EVALUATION ----------------
        metrics, evaluation_results = run_pipeline(
            "Model Evaluation Stage",
            ModelEvaluationTrainingPipeline()
        )

        # ---------------- FINAL SUMMARY ----------------
        logger.info("="*50)
        logger.info("FULL PIPELINE EXECUTED SUCCESSFULLY")
        logger.info("="*50)
        logger.info(f"Best Model from Training: {best_model_name}")
        logger.info(f"Training F1 Score: {best_score:.4f}")
        logger.info(f"Test F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        
        # Calculate performance gap
        performance_gap = best_score - metrics['f1_score']
        if performance_gap > 0:
            logger.info(f"Performance Gap (Train-Test): {performance_gap:.4f}")
        else:
            logger.info(f"Performance Gap (Train-Test): {performance_gap:.4f}")
        
        logger.info("="*50)

        # ---------------- VERIFY SAVED MODEL ----------------
        try:
            from mlproject.utils import load_object
            import json
            from pathlib import Path
            
            # Load the saved model
            model_path = Path("artifacts/model_trainer/model.pkl")
            if model_path.exists():
                model = load_object(model_path)
                logger.info(f"Model loaded successfully from {model_path}")
                
                # Load metadata
                metadata_path = Path("artifacts/model_trainer/model_metadata.json")
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    logger.info("Model metadata loaded:")
                    logger.info(f"Model: {metadata['best_model_name']}")
                    logger.info(f"Training F1 Score: {metadata['best_model_f1_score']:.4f}")
                    
                    # Log all models' F1 scores from training
                    logger.info("All Models Performance (Training):")
                    for model_name, score in metadata['all_models_f1_scores'].items():
                        logger.info(f"   - {model_name}: {score:.4f}")
            else:
                logger.warning(f"Model file not found at {model_path}")
            
            # Load evaluation results
            eval_path = Path("artifacts/model_trainer/model_evaluation.json")
            if eval_path.exists():
                with open(eval_path, 'r') as f:
                    eval_results = json.load(f)
                logger.info("Evaluation Results Loaded:")
                logger.info(f"Test F1 Score: {eval_results['metrics']['f1_score']:.4f}")
                logger.info(f"Test Accuracy: {eval_results['metrics']['accuracy']:.4f}")
                logger.info(f"Passes Baseline: {eval_results['comparison_with_baseline']['passes_baseline']}")
                
        except Exception as e:
            logger.warning(f"Could not verify saved artifacts: {str(e)}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)