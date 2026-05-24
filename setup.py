from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

REPO_NAME = "Personal-Loan-ML-Project"
AUTHOR_USER_NAME = "anuragchaudharytharu"
SRC_REPO = "mlProject"
AUTHOR_EMAIL = "anuragchaudharytharu371@gmail.com"


def get_requirements(file_path:str) -> List[str]:
    # this function will return the list of requirements    
    requirements = []
    
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    install_requires=get_requirements('requirements.txt'),
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)