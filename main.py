from mlproject.exception import CustomException
import sys
from mlproject.logging import logger
from mlproject.pipeline import (
    DataIngestionTrainingPipeline,
    DataValidationTrainingPipeline
)


def run_pipeline(stage_name, obj):
    try:
        logger.info(">>>>>>> %s started <<<<<<<", stage_name)
        obj.main()
        logger.info(">>>>>>> %s completed <<<<<<<", stage_name)
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":

    run_pipeline(
        "Data Ingestion Stage",
        DataIngestionTrainingPipeline()
    )

    run_pipeline(
        "Data Validation Stage",
        DataValidationTrainingPipeline()
    )