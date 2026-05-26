import os
import sys
import pandas as pd
import urllib.request as request
from pathlib import Path

from mlproject.logging import logger
from mlproject.utils import get_size
from mlproject.entity import DataIngestionConfig
from mlproject.exception import CustomException

from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    # ---------------- DOWNLOAD ----------------
    def download_file(self):
        try:
            if not os.path.exists(self.config.local_data_file):

                filename, headers = request.urlretrieve(
                    url=self.config.source_URL,
                    filename=self.config.local_data_file
                )

                logger.info("File downloaded successfully: %s", filename)

            else:
                logger.info(
                    "File already exists | size: %s",
                    get_size(Path(self.config.local_data_file))
                )

        except Exception as e:
            raise CustomException(e, sys)
            
    
    # ---------------- SPLIT ----------------
    def split_data(self):
        try:
            df = pd.read_csv(self.config.local_data_file)

            os.makedirs(self.config.root_dir, exist_ok=True)

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)

            logger.info("Train-test split completed")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        
        
    # ---------------- MAIN PIPELINE ----------------
    def initiate_data_ingestion(self):
        try:
            logger.info("Data ingestion started")

            self.download_file()
            train_path, test_path = self.split_data()

            logger.info("Data ingestion completed successfully")

            return train_path, test_path

        except Exception as e:
            raise CustomException(e, sys)