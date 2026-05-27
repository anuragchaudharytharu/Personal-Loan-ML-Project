from mlproject.config import ConfigurationManager
from mlproject.components import DataTransformation


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self, train_path, test_path):
        config = ConfigurationManager()

        transformation_config = config.get_data_transformation_config()

        transformation = DataTransformation(config=transformation_config)

        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_path,
            test_path
        )

        return train_arr, test_arr