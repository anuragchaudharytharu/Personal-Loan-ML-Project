from mlproject.config import ConfigurationManager
from mlproject.components import ModelTrainer

from mlproject.logging import logger
from mlproject.exception import CustomException

import sys


class ModelTrainerTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        try:
            logger.info("Starting Model Trainer Pipeline")

            config_manager = ConfigurationManager()

            model_trainer_config = (
                config_manager.get_model_trainer_config()
            )

            params = config_manager.params

            model_trainer = ModelTrainer(
                config=model_trainer_config,
                params=params
            )

            best_model_name, best_score = (
                model_trainer.initiate_model_training()
            )

            logger.info(
                f"Best Model: {best_model_name} | Score: {best_score}"
            )

            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)