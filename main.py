from mlproject.exception import CustomException
import sys
from mlproject.logging import logger
from mlproject.pipeline import (
    DataIngestionTrainingPipeline,
    DataValidationTrainingPipeline,
    DataTransformationTrainingPipeline
)


def run_pipeline(stage_name, obj, *args):
    try:
        logger.info(">>>>>>> %s STARTED <<<<<<<", stage_name)

        result = obj.main(*args)

        logger.info(">>>>>>> %s COMPLETED <<<<<<<", stage_name)

        return result

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":

    # ---------------- INGESTION ----------------
    train_path, test_path = run_pipeline(
        "Data Ingestion Stage",
        DataIngestionTrainingPipeline()
    )

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

    # ---------------- TRANSFORMATION ----------------
    train_arr, test_arr = run_pipeline(
        "Data Transformation Stage",
        DataTransformationTrainingPipeline(),
        train_path,
        test_path
    )

    logger.info("Data Transformation Completed Successfully")

    logger.info(f"Train array shape: {train_arr.shape}")
    logger.info(f"Test array shape: {test_arr.shape}")

    logger.info("Pipeline executed successfully")