import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlproject.exception import CustomException
from mlproject.logging import logger
from mlproject.utils import save_object
from mlproject.entity import DataTransformationConfig
from mlproject.utils import (
    save_object,
    read_yaml
)


# ----------------------------
# TRANSFORMATION CLASS
# ----------------------------
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

        # Load schema once
        SCHEMA_FILE_PATH = Path("schema.yaml")
        self.schema = read_yaml(SCHEMA_FILE_PATH)

        self.target_column = str(self.schema["target_column"]).strip()

        self.drop_columns = list(self.schema["drop_columns"])
        
        self.categorical_columns = list(self.schema["categorical_columns"])
        
        self.numerical_columns = list(self.schema["numerical_columns"])

    # ----------------------------
    # PREPROCESSOR
    # ----------------------------
    def get_preprocessor(self):
        try:
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logger.info("Created numerical and categorical pipelines")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, self.numerical_columns),
                ("cat_pipeline", cat_pipeline, self.categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # ----------------------------
    # MAIN TRANSFORMATION
    # ----------------------------
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Loaded train and test data")

            # ----------------------------
            # DROP COLUMNS
            # ----------------------------
            train_df.drop(columns=self.drop_columns, inplace=True)
            test_df.drop(columns=self.drop_columns, inplace=True)

            # ----------------------------
            # TYPECAST CATEGORICAL
            # ----------------------------
            for col in self.categorical_columns:
                train_df[col] = train_df[col].astype(str)
                test_df[col] = test_df[col].astype(str)

            # ----------------------------
            # SPLIT FEATURES & TARGET
            # ----------------------------
            target_col = self.target_column

            if target_col not in train_df.columns:
                raise Exception(f"Target column '{target_col}' not found. Available columns: {train_df.columns}")
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[self.target_column]
            
            
            if target_col not in test_df.columns:
                raise Exception(f"Target column '{target_col}' not found. Available columns: {test_df.columns}")            
            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]

            # ----------------------------
            # PREPROCESSING
            # ----------------------------
            preprocessor = self.get_preprocessor()

            logger.info("Fitting preprocessor on training data")

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # ----------------------------
            # COMBINE
            # ----------------------------
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            # ----------------------------
            # SAVE PREPROCESSOR
            # ----------------------------
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logger.info("Preprocessor saved successfully")

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)