import sys
import yaml
import json
import joblib
from pathlib import Path
from typing import Any

from box import ConfigBox
from ensure import ensure_annotations

from mlproject.logging import logger
from mlproject.exception import CustomException

@ensure_annotations
def read_yaml(path_to_yaml: Path):

    try:
        if not path_to_yaml.exists():
            raise FileNotFoundError(f"File not found: {path_to_yaml}")

        with open(path_to_yaml, encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)

        if content is None:
            raise ValueError("YAML file is empty or invalid")

        logger.info(f"YAML loaded successfully: {path_to_yaml}")

        return ConfigBox(content)

    except Exception as e:
        raise CustomException(e, sys)
    
    
@ensure_annotations
def create_directories(path_to_directories, verbose: bool = True):

    try:
        for path in path_to_directories:
            path.mkdir(parents=True, exist_ok=True)

            if verbose:
                logger.info(f"Created directory at: {path}")

    except Exception as e:
        raise CustomException(e, sys)
    
    
@ensure_annotations
def save_json(path: Path, data):

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        logger.info(f"JSON saved at: {path}")

    except Exception as e:
        raise CustomException(e, sys)
    
    
@ensure_annotations
def load_json(path: Path):

    try:
        with open(path, encoding="utf-8") as f:
            content = json.load(f)

        logger.info(f"JSON loaded from: {path}")

        return ConfigBox(content)

    except Exception as e:
        raise CustomException(e, sys)
    
    
@ensure_annotations
def save_bin(data, path: Path):

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, path)

        logger.info(f"Binary saved at: {path}")

    except Exception as e:
        raise CustomException(e, sys)
    
    
@ensure_annotations
def load_bin(path: Path):

    try:
        data = joblib.load(path)

        logger.info(f"Binary loaded from: {path}")

        return data

    except Exception as e:
        raise CustomException(e, sys)
    
    
@ensure_annotations
def get_size(path: Path):

    try:
        size_kb = round(path.stat().st_size / 1024)
        return f"~ {size_kb} KB"

    except Exception as e:
        raise CustomException(e, sys)