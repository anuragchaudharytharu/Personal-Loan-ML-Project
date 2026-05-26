import pandas as pd
import os
import sys
from mlproject.entity import DataValidationConfig
from mlproject.logging import logger
from mlproject.exception import CustomException

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        
    # ---------------- VALIDATION ----------------
    def validate(self, train_path, test_path) -> bool:

        try:
            logger.info("Data validation started")

            # read datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # schema columns
            schema_cols = set(self.config.all_schema.keys())

            # dataset columns
            train_cols = set(train_df.columns)
            test_cols = set(test_df.columns)

            # ---------------- CHECKS ----------------
            validation_status = (
                train_cols == schema_cols and
                test_cols == schema_cols
            )

            # create directory
            os.makedirs(self.config.root_dir, exist_ok=True)

            # save status
            with open(self.config.status_file, "w") as f:
                f.write(f"Validation status: {validation_status}")

            if validation_status:
                logger.info("Data validation passed")
            else:
                logger.warning("Data validation failed")

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)