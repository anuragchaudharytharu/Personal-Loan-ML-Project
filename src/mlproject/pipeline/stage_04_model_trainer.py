# from mlproject.config import ConfigurationManager
# from mlproject.components import ModelTrainer

# from mlproject.logging import logger
# from mlproject.exception import CustomException

# import sys


# class ModelTrainerTrainingPipeline:

#     def __init__(self):
#         pass

#     def main(self):

#         try:
#             logger.info("Starting Model Trainer Pipeline")

#             config_manager = ConfigurationManager()

#             model_trainer_config = (
#                 config_manager.get_model_trainer_config()
#             )

#             params = config_manager.params

#             model_trainer = ModelTrainer(
#                 config=model_trainer_config,
#                 params=params
#             )

#             best_model_name, best_score = (
#                 model_trainer.initiate_model_training()
#             )

#             logger.info(
#                 f"Best Model: {best_model_name} | Score: {best_score}"
#             )

#             return best_model_name, best_score

#         except Exception as e:
#             raise CustomException(e, sys)

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
            logger.info("="*50)
            logger.info("Starting Model Trainer Pipeline")
            logger.info("="*50)

            # Get configuration
            config_manager = ConfigurationManager()
            model_trainer_config = config_manager.get_model_trainer_config()
            params = config_manager.params

            # Initialize model trainer
            model_trainer = ModelTrainer(
                config=model_trainer_config,
                params=params
            )

            # Run model training
            best_model_name, best_score = model_trainer.initiate_model_training()

            logger.info("="*50)
            logger.info(f"✅ Model Trainer Pipeline Completed Successfully")
            logger.info(f"🏆 Best Model: {best_model_name}")
            logger.info(f"📊 Best F1 Score: {best_score:.4f}")
            logger.info("="*50)

            return best_model_name, best_score

        except Exception as e:
            logger.error(f"Model Trainer Pipeline failed: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = ModelTrainerTrainingPipeline()
    result = pipeline.main()
    print(f"Pipeline Result: {result}")