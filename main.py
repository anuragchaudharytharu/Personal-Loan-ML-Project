from mlproject.exception import CustomException
import sys
from mlproject.logging import logger
from mlproject.pipeline import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
if __name__ == "__main__":
    try:
        logger.info(
            ">>>>>>> stage %s started <<<<<<<",
            STAGE_NAME
        )

        obj = DataIngestionTrainingPipeline()

        obj.main()

        logger.info(
            ">>>>>>> stage %s completed <<<<<<<",
            STAGE_NAME
        )

    except Exception as e:
        raise CustomException(e, sys)