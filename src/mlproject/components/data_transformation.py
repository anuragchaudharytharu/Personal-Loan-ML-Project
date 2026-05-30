import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json

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
        
        # Get outlier columns from schema
        self.outlier_columns = list(self.schema.get("outlier_columns", []))

    # ----------------------------
    # OUTLIER DETECTION & REMOVAL
    # ----------------------------
    def remove_outliers(self, df, outlier_columns, method="iqr", threshold=1.5):
        """
        Remove outliers from specified columns
        
        Parameters:
        - df: DataFrame
        - outlier_columns: list of column names to check for outliers
        - method: "iqr" or "zscore"
        - threshold: IQR multiplier (default 1.5) or Z-score threshold (default 3)
        
        Returns:
        - DataFrame with outliers removed
        - Dictionary with outlier statistics
        """
        try:
            original_shape = df.shape
            outlier_stats = {}
            rows_to_keep = pd.Series([True] * len(df))
            
            logger.info(f"Checking outliers in columns: {outlier_columns}")
            
            if method == "iqr":
                # IQR method
                for col in outlier_columns:
                    if col in df.columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        
                        # Identify rows to keep (not outliers)
                        col_rows_to_keep = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                        outliers_removed = (~col_rows_to_keep).sum()
                        
                        # Update rows to keep (intersection)
                        rows_to_keep = rows_to_keep & col_rows_to_keep
                        
                        outlier_stats[col] = {
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound),
                            'outliers_removed': int(outliers_removed),
                            'outlier_percentage': float(outliers_removed / len(df) * 100)
                        }
                        
                        logger.info(f"  Column '{col}': Removed {outliers_removed} outliers ({outliers_removed/len(df)*100:.2f}%)")
                        logger.info(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            elif method == "zscore":
                # Z-score method
                from scipy import stats
                
                for col in outlier_columns:
                    if col in df.columns:
                        z_scores = np.abs(stats.zscore(df[col].dropna()))
                        col_rows_to_keep = pd.Series(True, index=df.index)
                        col_rows_to_keep[df[col].dropna().index[z_scores > threshold]] = False
                        outliers_removed = (~col_rows_to_keep).sum()
                        
                        # Update rows to keep
                        rows_to_keep = rows_to_keep & col_rows_to_keep
                        
                        outlier_stats[col] = {
                            'threshold': threshold,
                            'outliers_removed': int(outliers_removed),
                            'outlier_percentage': float(outliers_removed / len(df) * 100)
                        }
                        
                        logger.info(f"  Column '{col}': Removed {outliers_removed} outliers ({outliers_removed/len(df)*100:.2f}%)")
            
            # Apply filtering
            df_clean = df[rows_to_keep].copy()
            final_shape = df_clean.shape
            total_removed = original_shape[0] - final_shape[0]
            
            logger.info("="*50)
            logger.info("OUTLIER REMOVAL SUMMARY:")
            logger.info(f"  Original rows: {original_shape[0]}")
            logger.info(f"  Total outliers removed: {total_removed} ({total_removed/original_shape[0]*100:.2f}%)")
            logger.info(f"  Final rows: {final_shape[0]}")
            logger.info("="*50)
            
            return df_clean, outlier_stats
            
        except Exception as e:
            logger.warning(f"Error in outlier removal: {str(e)}")
            logger.info("Continuing without outlier removal")
            return df, {}

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
            logger.info(f"Original train shape: {train_df.shape}")
            logger.info(f"Original test shape: {test_df.shape}")

            # ----------------------------
            # DROP COLUMNS
            # ----------------------------
            train_df.drop(columns=self.drop_columns, inplace=True)
            test_df.drop(columns=self.drop_columns, inplace=True)
            
            logger.info(f"Dropped columns: {self.drop_columns}")
            logger.info(f"Train shape after dropping: {train_df.shape}")
            logger.info(f"Test shape after dropping: {test_df.shape}")

            # ----------------------------
            # TYPECAST CATEGORICAL
            # ----------------------------
            for col in self.categorical_columns:
                if col in train_df.columns:
                    train_df[col] = train_df[col].astype(str)
                    test_df[col] = test_df[col].astype(str)
                    logger.info(f"Converted column '{col}' to string type")

            # ----------------------------
            # REMOVE OUTLIERS FROM TRAINING DATA
            # ----------------------------
            if self.outlier_columns:
                logger.info("="*50)
                logger.info("STARTING OUTLIER REMOVAL")
                logger.info(f"Outlier columns: {self.outlier_columns}")
                logger.info("="*50)
                
                # Get outlier columns that exist in dataframe
                outlier_cols_exist = [col for col in self.outlier_columns if col in train_df.columns]
                
                if outlier_cols_exist:
                    # Get outlier method and threshold from config (with defaults)
                    outlier_method = getattr(self.config, 'outlier_method', 'iqr')
                    outlier_threshold = getattr(self.config, 'outlier_threshold', 1.5)
                    
                    # Remove outliers from training data
                    train_df, outlier_stats = self.remove_outliers(
                        train_df, 
                        outlier_cols_exist,
                        method=outlier_method,
                        threshold=outlier_threshold
                    )
                    
                    # Save outlier statistics
                    if outlier_stats:
                        outlier_stats_path = os.path.join(self.config.root_dir, "outlier_stats.json")
                        with open(outlier_stats_path, 'w') as f:
                            json.dump(outlier_stats, f, indent=4)
                        logger.info(f"Outlier statistics saved to: {outlier_stats_path}")
                else:
                    logger.warning(f"No outlier columns found in dataframe. Available columns: {train_df.columns.tolist()}")
            else:
                logger.info("No outlier columns specified in schema. Skipping outlier removal.")

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

            logger.info(f"Features shape - Train: {X_train.shape}, Test: {X_test.shape}")
            logger.info(f"Target shape - Train: {y_train.shape}, Test: {y_test.shape}")
            logger.info(f"Target distribution - Train:\n{y_train.value_counts()}")
            logger.info(f"Target distribution - Test:\n{y_test.value_counts()}")

            # ----------------------------
            # PREPROCESSING
            # ----------------------------
            preprocessor = self.get_preprocessor()

            logger.info("Fitting preprocessor on training data")

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            logger.info(f"Transformed features shape - Train: {X_train_arr.shape}, Test: {X_test_arr.shape}")

            # ----------------------------
            # COMBINE
            # ----------------------------
            # Handle sparse matrix if needed
            if hasattr(X_train_arr, 'toarray'):
                X_train_arr = X_train_arr.toarray()
                X_test_arr = X_test_arr.toarray()
            
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            # ----------------------------
            # SAVE TRANSFORMED ARRAYS
            # ----------------------------
            train_arr_path = os.path.join(
                self.config.root_dir,
                "train.npy"
            )

            test_arr_path = os.path.join(
                self.config.root_dir,
                "test.npy"
            )

            np.save(train_arr_path, train_arr)
            np.save(test_arr_path, test_arr)

            logger.info(f"Transformed train array saved to: {train_arr_path}")
            logger.info(f"Transformed test array saved to: {test_arr_path}")
            
            # ----------------------------
            # SAVE PREPROCESSOR
            # ----------------------------
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logger.info(f"Preprocessor saved to: {self.config.preprocessor_obj_file_path}")

            return (
                train_arr_path,
                test_arr_path,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)