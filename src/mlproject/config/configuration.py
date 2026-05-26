from mlproject.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH
)

from mlproject.utils import (
    read_yaml,
    create_directories
)

from mlproject.entity import (
    DataValidationConfig,
    DataIngestionConfig
)
from pathlib import Path

class ConfigurationManager:

    def __init__(
        self,
        config_file_path=CONFIG_FILE_PATH,
        schema_file_path=SCHEMA_FILE_PATH,
        params_file_path=PARAMS_FILE_PATH
    ):

        self.config = read_yaml(config_file_path)
        self.schema = read_yaml(schema_file_path)
        self.params = read_yaml(params_file_path)

        create_directories(
            [Path(self.config.artifacts_root)]
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        create_directories(
            [Path(config.root_dir)]
        )

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file)
        )        
    
    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories(
            [Path(config.root_dir)]
        )

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE = config.STATUS_FILE,
            downloaded_data_dir = config.downloaded_data_dir,
            all_schema = schema,
        )
