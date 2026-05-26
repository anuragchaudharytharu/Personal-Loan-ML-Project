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
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
        reads yaml file and returns

        Args:
            path_to_yaml (str): path like input

        Raises:
            ValueError: if yaml file is empty
            e: empty file

        Returns:
            ConfigBox: ConfigBox type
    """

    try:

        if not path_to_yaml.exists():
            raise FileNotFoundError(f"File not found: {path_to_yaml}")

        with open(path_to_yaml, encoding="utf-8") as yaml_file:

            content = yaml.safe_load(yaml_file)

            if content is None or len(content) == 0:
                raise ValueError("YAML file is empty or invalid")

            logger.info(
                "YAML file loaded successfully: %s",
                path_to_yaml
            )

            return ConfigBox(content)

    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def create_directories(
    path_to_directories: list[Path],
    verbose: bool = True
):
    """
        create list of directories

        Args:
            path_to_directories (list): list of path of directories
            ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """

    try:

        for path in path_to_directories:

            path.mkdir(parents=True, exist_ok=True)

            if verbose:
                logger.info(
                    "Created directory at: %s",
                    path
                )

    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def save_json(path: Path, data: dict):
    """
        save json data

        Args:
            path (Path): path to json file
            data (dict): data to be saved in json file
    """

    try:

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        logger.info(
            "JSON file saved successfully at: %s",
            path
        )

    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
        load json files data

        Args:
            path (Path): path to json file

        Returns:
            ConfigBox: data as class attributes instead of dict
    """

    try:

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, encoding="utf-8") as f:
            content = json.load(f)

        logger.info(
            "JSON file loaded successfully from: %s",
            path
        )

        return ConfigBox(content)

    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """
        save binary file

        Args:
            data (Any): data to be saved as binary
            path (Path): path to binary file
    """

    try:

        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(value=data, filename=path)

        logger.info(
            "Binary file saved successfully at: %s",
            path
        )

    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
        load binary data

        Args:
            path (Path): path to binary file

        Returns:
            Any: object stored in the file
    """

    try:

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        data = joblib.load(path)

        logger.info(
            "Binary file loaded successfully from: %s",
            path
        )

        return data

    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def get_size(path: Path) -> str:
    """
        get size in KB

        Args:
            path (Path): path of the file

        Returns:
            str: size in KB
    """

    try:

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        size_in_kb = round(path.stat().st_size / 1024)

        return f"~ {size_in_kb} KB"

    except Exception as e:
        raise CustomException(e, sys)