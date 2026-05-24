import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name = "mlproject"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitering.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "agenda.txt"
]


for file_path in list_of_files:
    file_path = Path(file_path)
    file_directory, file_name = os.path.split(file_path)
    
    if file_directory != "":
        os.makedirs(file_directory, exist_ok=True)
        logging.info(f"Creating firectory: {file_directory} for the file {file_name}")   
         
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path == 0)):
        with open(file_path, 'w') as f:
            pass
        logging.info(f"Creating empty file: {file_path}")   
         
    else:
        logging.info(f"{file_name} already exists.")