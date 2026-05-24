import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

project_name = "mlproject"


list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html",
    "agenda.txt"
]


for filepath in list_of_files:
    
    # create file path for the list of files. Path() is used to make different os read the file like for windows, linux etc 
    file_path = Path(filepath)
    
    # store file directory and file name by spliting the file path
    file_dir, file_name = os.path.split(file_path)

    if file_dir:
        # make directory automatically if it doesnot exist
        os.makedirs(file_dir, exist_ok=True)
        logging.info(
            f"Creating directory: {file_dir} for the file: {file_name}"
        )
        
    # create file if not exists OR size of file is empty
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        # simply creates an EMPTY file.
        with open(file_path, "w") as f:
            pass
        logging.info(f"Creating empty file: {file_path}")   
    
    else:
        logging.info(f"{file_name} already exists")
        
