from mlproject.config import ConfigurationManager
from mlproject.components import DataValidation


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        data_validation_config = (
            config.get_data_validation_config()
        )

        data_validation = DataValidation(
            config=data_validation_config
        )

        data_validation.validate_all_columns()