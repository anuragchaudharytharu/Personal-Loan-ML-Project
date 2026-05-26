from mlproject.config import ConfigurationManager
from mlproject.components import DataValidation
from mlproject.logging import logger
from mlproject.exception import CustomException
import sys


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self, train_path, test_path):

        try:
            logger.info("Data Validation Pipeline started")

            # load config (only settings, not data)
            config = ConfigurationManager()
            validation_config = config.get_data_validation_config()

            # component
            validation = DataValidation(validation_config)

            # run validation using ingestion outputs
            status = validation.validate(train_path, test_path)

            logger.info("Data Validation Pipeline completed")

            return status

        except Exception as e:
            raise CustomException(e, sys)