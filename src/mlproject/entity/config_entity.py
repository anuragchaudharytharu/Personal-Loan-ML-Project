from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    train_data_path: Path
    test_data_path: Path
    
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    all_schema: dict
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    preprocessor_obj_file_path: Path
    

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_path: Path
    params_path: Path

    train_data_path: Path
    test_data_path: Path

    target_column: str

    scoring: str
    cv: int
    n_jobs: int