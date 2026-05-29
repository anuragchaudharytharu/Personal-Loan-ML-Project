# import os
# import sys
# import numpy as np
# import pandas as pd
# import importlib
# import json
# from datetime import datetime
# from pathlib import Path

# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import f1_score
# from sklearn.utils.class_weight import compute_class_weight

# from mlproject.logging import logger
# from mlproject.exception import CustomException
# from mlproject.utils import save_object

# # COMPONENT CLASS
# class ModelTrainer:

#     def __init__(self, config, params):
#         """
#         config: ModelTrainerConfig
#         params: params.yaml dict
#         """

#         self.config = config
#         self.params = params
        
#     # DATA LOADING
#     def load_data(self):

#         train_arr = np.load(self.config.train_data_path)
#         test_arr = np.load(self.config.test_data_path)

#         X_train = train_arr[:, :-1]
#         y_train = train_arr[:, -1]

#         X_test = test_arr[:, :-1]
#         y_test = test_arr[:, -1]

#         return X_train, X_test, y_train, y_test
    
#     # CLASS IMBALANCE HANDLEER (XGBOOST HELPER)
#     def compute_scale_pos_weight(self, y_train):

#         classes = np.unique(y_train)

#         weights = compute_class_weight(
#             class_weight="balanced",
#             classes=classes,
#             y=y_train
#         )

#         weight_dict = dict(zip(classes, weights))

#         return weight_dict[1] / weight_dict[0]
    
#     def _get_model_class(self, model_name):
#         """Dynamically import model class based on model name"""
        
#         # Mapping of model names to their module paths
#         model_modules = {
#             "LogisticRegression": ("sklearn.linear_model", "LogisticRegression"),
#             "DecisionTree": ("sklearn.tree", "DecisionTreeClassifier"),
#             "RandomForest": ("sklearn.ensemble", "RandomForestClassifier"),
#             "GradientBoosting": ("sklearn.ensemble", "GradientBoostingClassifier"),
#             "ExtraTrees": ("sklearn.ensemble", "ExtraTreesClassifier"),
#             "KNN": ("sklearn.neighbors", "KNeighborsClassifier"),
#             "SVM": ("sklearn.svm", "SVC"),
#             "XGBoost": ("xgboost", "XGBClassifier"),
#         }
        
#         if model_name not in model_modules:
#             raise ValueError(f"Unsupported model: {model_name}")
        
#         module_path, class_name = model_modules[model_name]
#         module = importlib.import_module(module_path)
#         model_class = getattr(module, class_name)
        
#         return model_class
    
#     # MODEL FACTORY (maps YAML -> sklearn models)
#     def get_model_and_params(self, model_name, model_data, y_train):
#         # Dynamically import the model class based on model_name
#         model_class = self._get_model_class(model_name)
        
#         params = model_data["params"]
        
#         # Convert ConfigBox to regular dict if needed
#         if hasattr(params, 'items'):
#             param_grid = dict(params)
#         else:
#             param_grid = params
        
#         # Clean parameter names (remove 'model__' prefix)
#         if isinstance(param_grid, dict):
#             param_grid = {key.replace('model__', ''): value for key, value in param_grid.items()}
            
#             # Remove any non-parameter keys
#             param_grid.pop('eval_metric', None)
#             param_grid.pop('scale_pos_weight', None)
            
#             # Ensure all values are lists
#             for key, value in param_grid.items():
#                 if not isinstance(value, (list, np.ndarray)):
#                     param_grid[key] = [value]
                
#                 # Handle None/null values
#                 if isinstance(param_grid[key], list):
#                     param_grid[key] = [None if x == 'null' or x is None else x for x in param_grid[key]]
        
#         # special handling for xgboost
#         if model_name == "XGBoost":
#             from xgboost import XGBClassifier
            
#             scale_pos_weight = self.compute_scale_pos_weight(y_train)
            
#             model = XGBClassifier(
#                 eval_metric="logloss",
#                 scale_pos_weight=scale_pos_weight
#             )
#         else:
#             model = model_class()
        
#         logger.info(f"Created model: {model_name} with class: {model_class.__name__}")
#         logger.info(f"Parameter grid: {param_grid}")
        
#         return model, param_grid
    
#     # MAIN COMPONENT LOGIC (TRAINING ONLY)
#     def initiate_model_training(self):

#         try:
#             logger.info("Starting model training component")

#             X_train, X_test, y_train, y_test = self.load_data()

#             models_config = self.params["models"]

#             best_model = None
#             best_score = -1
#             best_model_name = None
#             best_params = None
#             best_cv_results = None
            
#             # Store all model results
#             all_models_results = {}

#             for model_name, model_data in models_config.items():

#                 logger.info(f"Training model: {model_name}")

#                 model, param_grid = self.get_model_and_params(
#                     model_name,
#                     model_data,
#                     y_train
#                 )

#                 grid_search = GridSearchCV(
#                     estimator=model,
#                     param_grid=param_grid,
#                     cv=self.config.cv,
#                     scoring=self.config.scoring,
#                     n_jobs=self.config.n_jobs,
#                     verbose=1
#                 )

#                 grid_search.fit(X_train, y_train)

#                 y_pred = grid_search.best_estimator_.predict(X_test)

#                 score = f1_score(y_test, y_pred)

#                 logger.info(f"{model_name} Best Params: {grid_search.best_params_}")
#                 logger.info(f"{model_name} F1 Score: {score}")
                
#                 # Store results for this model
#                 all_models_results[model_name] = {
#                     'best_score': float(score),
#                     'best_params': grid_search.best_params_,
#                     'cv_best_score': float(grid_search.best_score_),
#                     'cv_results': grid_search.cv_results_
#                 }

#                 if score > best_score:
#                     best_score = score
#                     best_model = grid_search.best_estimator_
#                     best_model_name = model_name
#                     best_params = grid_search.best_params_
#                     best_cv_results = grid_search.cv_results_

#             logger.info(f"Best Model: {best_model_name} | Score: {best_score}")

#             # Save the best model - keep as Path
#             save_object(
#                 file_path=self.config.model_path,
#                 obj=best_model
#             )
            
#             # Create Path objects for metadata files
#             model_path_str = str(self.config.model_path)
#             metadata_path = Path(model_path_str.replace('.pkl', '_metadata.json'))
#             cv_results_path = Path(model_path_str.replace('.pkl', '_cv_results.pkl'))
#             all_results_path = Path(model_path_str.replace('.pkl', '_all_results.json'))
            
#             # Prepare metadata
#             metadata = {
#                 'best_model_name': best_model_name,
#                 'best_model_score': float(best_score),
#                 'best_model_params': best_params,
#                 'training_date': datetime.now().isoformat(),
#                 'cv_folds': self.config.cv,
#                 'scoring_metric': self.config.scoring,
#                 'model_path': model_path_str,
#                 'all_models_summary': {
#                     model_name: {
#                         'best_score': results['best_score'],
#                         'best_params': results['best_params'],
#                         'cv_best_score': results['cv_best_score']
#                     }
#                     for model_name, results in all_models_results.items()
#                 }
#             }
            
#             # Save metadata as JSON
#             with open(metadata_path, 'w') as f:
#                 json.dump(metadata, f, indent=4)
            
#             # Optionally save all CV results for the best model (can be large)
#             if best_cv_results:
#                 save_object(
#                     file_path=cv_results_path,
#                     obj=best_cv_results
#                 )
#                 logger.info(f"CV results saved to: {cv_results_path}")
            
#             # Save all models results (without the full CV results to save space)
#             simplified_results = {
#                 model_name: {
#                     'best_score': results['best_score'],
#                     'best_params': results['best_params'],
#                     'cv_best_score': results['cv_best_score']
#                 }
#                 for model_name, results in all_models_results.items()
#             }
            
#             with open(all_results_path, 'w') as f:
#                 json.dump(simplified_results, f, indent=4)
            
#             logger.info(f"Best model saved to: {self.config.model_path}")
#             logger.info(f"Metadata saved to: {metadata_path}")
#             logger.info(f"All results saved to: {all_results_path}")

#             return best_model_name, best_score

#         except Exception as e:
#             raise CustomException(e, sys)


import os
import sys
import numpy as np
import pandas as pd
import importlib
import json
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from mlproject.logging import logger
from mlproject.exception import CustomException
from mlproject.utils import save_object

# COMPONENT CLASS
class ModelTrainer:

    def __init__(self, config, params):
        """
        config: ModelTrainerConfig
        params: params.yaml dict
        """

        self.config = config
        self.params = params
        
    # DATA LOADING
    def load_data(self):

        train_arr = np.load(self.config.train_data_path)
        test_arr = np.load(self.config.test_data_path)

        X_train = train_arr[:, :-1]
        y_train = train_arr[:, -1]

        X_test = test_arr[:, :-1]
        y_test = test_arr[:, -1]

        return X_train, X_test, y_train, y_test
    
    # CLASS IMBALANCE HANDLEER (XGBOOST HELPER)
    def compute_scale_pos_weight(self, y_train):

        classes = np.unique(y_train)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )

        weight_dict = dict(zip(classes, weights))

        return weight_dict[1] / weight_dict[0]
    
    def _get_model_class(self, model_name):
        """Dynamically import model class based on model name"""
        
        # Mapping of model names to their module paths
        model_modules = {
            "LogisticRegression": ("sklearn.linear_model", "LogisticRegression"),
            "DecisionTree": ("sklearn.tree", "DecisionTreeClassifier"),
            "RandomForest": ("sklearn.ensemble", "RandomForestClassifier"),
            "GradientBoosting": ("sklearn.ensemble", "GradientBoostingClassifier"),
            "ExtraTrees": ("sklearn.ensemble", "ExtraTreesClassifier"),
            "KNN": ("sklearn.neighbors", "KNeighborsClassifier"),
            "SVM": ("sklearn.svm", "SVC"),
            "XGBoost": ("xgboost", "XGBClassifier"),
        }
        
        if model_name not in model_modules:
            raise ValueError(f"Unsupported model: {model_name}")
        
        module_path, class_name = model_modules[model_name]
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        return model_class
    
    # MODEL FACTORY (maps YAML -> sklearn models)
    def get_model_and_params(self, model_name, model_data, y_train):
        # Dynamically import the model class based on model_name
        model_class = self._get_model_class(model_name)
        
        params = model_data["params"]
        
        # Convert ConfigBox to regular dict if needed
        if hasattr(params, 'items'):
            param_grid = dict(params)
        else:
            param_grid = params
        
        # Clean parameter names (remove 'model__' prefix)
        if isinstance(param_grid, dict):
            param_grid = {key.replace('model__', ''): value for key, value in param_grid.items()}
            
            # Remove any non-parameter keys
            param_grid.pop('eval_metric', None)
            param_grid.pop('scale_pos_weight', None)
            
            # Ensure all values are lists
            for key, value in param_grid.items():
                if not isinstance(value, (list, np.ndarray)):
                    param_grid[key] = [value]
                
                # Handle None/null values
                if isinstance(param_grid[key], list):
                    param_grid[key] = [None if x == 'null' or x is None else x for x in param_grid[key]]
        
        # special handling for xgboost
        if model_name == "XGBoost":
            from xgboost import XGBClassifier
            
            scale_pos_weight = self.compute_scale_pos_weight(y_train)
            
            model = XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight
            )
        else:
            model = model_class()
        
        logger.info(f"Created model: {model_name} with class: {model_class.__name__}")
        logger.info(f"Parameter grid: {param_grid}")
        
        return model, param_grid
    
    # MAIN COMPONENT LOGIC (TRAINING ONLY)
    def initiate_model_training(self):

        try:
            logger.info("Starting model training component")

            X_train, X_test, y_train, y_test = self.load_data()

            models_config = self.params["models"]

            best_model = None
            best_score = -1
            best_model_name = None
            best_params = None
            best_cv_results = None
            
            # Store all model results
            all_models_results = {}

            for model_name, model_data in models_config.items():

                logger.info(f"Training model: {model_name}")

                model, param_grid = self.get_model_and_params(
                    model_name,
                    model_data,
                    y_train
                )

                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=self.config.cv,
                    scoring=self.config.scoring,
                    n_jobs=self.config.n_jobs,
                    verbose=1
                )

                grid_search.fit(X_train, y_train)

                y_pred = grid_search.best_estimator_.predict(X_test)

                score = f1_score(y_test, y_pred)

                logger.info(f"{model_name} Best Params: {grid_search.best_params_}")
                logger.info(f"{model_name} F1 Score: {score}")
                
                # Store results for this model
                all_models_results[model_name] = {
                    'f1_score': float(score),
                    'best_params': grid_search.best_params_,
                    'cv_best_score': float(grid_search.best_score_),
                    'cv_results': grid_search.cv_results_
                }

                if score > best_score:
                    best_score = score
                    best_model = grid_search.best_estimator_
                    best_model_name = model_name
                    best_params = grid_search.best_params_
                    best_cv_results = grid_search.cv_results_

            logger.info(f"Best Model: {best_model_name} | Score: {best_score}")

            # Save the best model - keep as Path
            save_object(
                file_path=self.config.model_path,
                obj=best_model
            )
            
            # Create Path objects for metadata files
            model_path_str = str(self.config.model_path)
            metadata_path = Path(model_path_str.replace('.pkl', '_metadata.json'))
            cv_results_path = Path(model_path_str.replace('.pkl', '_cv_results.pkl'))
            all_results_path = Path(model_path_str.replace('.pkl', '_all_results.json'))
            
            # Prepare metadata with F1 scores
            metadata = {
                'best_model_name': best_model_name,
                'best_model_f1_score': float(best_score),
                'best_model_params': best_params,
                'training_date': datetime.now().isoformat(),
                'cv_folds': self.config.cv,
                'scoring_metric': self.config.scoring,
                'model_path': model_path_str,
                'all_models_f1_scores': {
                    model_name: results['f1_score']
                    for model_name, results in all_models_results.items()
                },
                'all_models_summary': {
                    model_name: {
                        'f1_score': results['f1_score'],
                        'best_params': results['best_params'],
                        'cv_best_score': results['cv_best_score']
                    }
                    for model_name, results in all_models_results.items()
                }
            }
            
            # Save metadata as JSON
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Optionally save all CV results for the best model (can be large)
            if best_cv_results:
                save_object(
                    file_path=cv_results_path,
                    obj=best_cv_results
                )
                logger.info(f"CV results saved to: {cv_results_path}")
            
            # Save all models results with F1 scores (without the full CV results to save space)
            simplified_results = {
                model_name: {
                    'f1_score': results['f1_score'],
                    'best_params': results['best_params'],
                    'cv_best_score': results['cv_best_score']
                }
                for model_name, results in all_models_results.items()
            }
            
            with open(all_results_path, 'w') as f:
                json.dump(simplified_results, f, indent=4)
            
            logger.info(f"Best model saved to: {self.config.model_path}")
            logger.info(f"Metadata saved to: {metadata_path}")
            logger.info(f"All results saved to: {all_results_path}")
            
            # Log all F1 scores
            logger.info("All Models F1 Scores:")
            for model_name, results in all_models_results.items():
                logger.info(f"  {model_name}: {results['f1_score']:.4f}")

            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)