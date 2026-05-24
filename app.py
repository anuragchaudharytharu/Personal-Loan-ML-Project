from mlproject.logger import logging
from mlproject.exception import CustomException

from mlproject.components.data_ingestion import DataIngestion
from mlproject.components.data_transformation import DataTransformation
from mlproject.components.model_trainer import ModelTrainer

import sys


if __name__ == "__main__":
    logging.info("Pipeline execution started")
    print("Running ML Pipeline")

    try:
        # -----------------------------
        # DATA INGESTION
        # -----------------------------
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # -----------------------------
        # DATA TRANSFORMATION
        # -----------------------------
        data_transformation = DataTransformation()
        
        X_train, X_test, y_train, y_test, preprocessor = \
            data_transformation.initiate_data_transformation(
                train_data_path,
                test_data_path,
            )

        # -----------------------------
        # MODEL TRAINING
        # -----------------------------
        model_trainer = ModelTrainer()

        best_score, best_model_name = model_trainer.initiate_model_trainer(
            X_train,
            y_train,
            X_test,
            y_test,
            preprocessor
        )

        logging.info(f"Best Model: {best_model_name}")
        logging.info(f"Best Score: {best_score}")

        print("Best Model:", best_model_name)
        print("Best F1 Score:", best_score)

    except Exception as e:
        logging.error("Error occurred in pipeline")
        raise CustomException(e, sys)

# from mlproject.logger import logging
# from mlproject.exception import CustomException
# from mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig
# from mlproject.components.data_transformation import DataTransformationConfig, DataTransformation
# from mlproject.components.model_trainer import ModelTrainer, ModelTrainerConfig

# import sys

# if __name__ == "__main__":
#     logging.info("the execution has started")
#     print("Running app")

#     try:
#         data_ingestion_config = DataIngestionConfig()
#         data_ingestion = DataIngestion()
#         train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
#         data_transformation_config = DataTransformationConfig()
#         data_transformation = DataTransformation()
#         train_arr, test_arr,_,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
#         # Model Training
#         model_trainer = ModelTrainer()
#         model_trainer.initiate_model_trainer(train_arr, test_arr)
        
#     except Exception as e:
#         logging.info('Custom Exception')
#         raise CustomException(e, sys)