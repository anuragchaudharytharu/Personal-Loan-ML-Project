from mlproject.config import ConfigurationManager
from mlproject.components import DataIngestion


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = DataIngestion(config = data_ingestion_config)

        train_path, test_path = data_ingestion.initiate_data_ingestion()

        return train_path, test_path