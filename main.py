from mlproject.exception import CustomException
import sys
from mlproject.logging import logger
from mlproject.pipeline import (
    DataIngestionTrainingPipeline,
    DataValidationTrainingPipeline
)


STAGE_NAME = "Data Ingestion Stage"
if __name__ == "__main__":
    try:
        logger.info(
            ">>>>>>> stage %s started <<<<<<<",
            STAGE_NAME
        )

        data_ingestion_obj = DataIngestionTrainingPipeline()

        data_ingestion_obj.main()

        logger.info(
            ">>>>>>> stage %s completed <<<<<<<",
            STAGE_NAME
        )

    except Exception as e:
        raise CustomException(e, sys)
    
STAGE_NAME = "Data Validation Stage"
if __name__ == "__main__":
    try:
        logger.info(
            ">>>>>>> stage %s started <<<<<<<",
            STAGE_NAME
        )

        data_validation_obj = DataValidationTrainingPipeline()

        data_validation_obj.main()

        logger.info(
            ">>>>>>> stage %s completed <<<<<<<",
            STAGE_NAME
        )

    except Exception as e:
        raise CustomException(e, sys)