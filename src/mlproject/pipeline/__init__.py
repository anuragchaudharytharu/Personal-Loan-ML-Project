from .stage_01_data_ingestion import (
    DataIngestionTrainingPipeline
)

from .stage_02_data_validation import (
    DataValidationTrainingPipeline
)

from .stage_03_data_transformation import (
    DataTransformationTrainingPipeline
)

from .stage_04_model_trainer import ModelTrainerTrainingPipeline

from.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

__all__ = [
    "DataIngestionTrainingPipeline",
    "DataValidationTrainingPipeline",
    "DataTransformationTrainingPipeline",
    "ModelTrainerTrainingPipeline",
    "ModelEvaluationTrainingPipeline"
]