import os
import urllib.request as request
from pathlib import Path

from mlproject.logging import logger
from mlproject.utils import get_size
from mlproject.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )

            logger.info(
                "%s downloaded successfully!",
                filename
            )

        else:
            logger.info(
                "File already exists with size: %s",
                get_size(Path(self.config.local_data_file))
            )

        logger.info("Data ingestion step completed successfully")