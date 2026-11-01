from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    Reads requirements.txt and returns list of dependencies.
    """
    requirements = []
    with open(file_path, "r", encoding="utf-8") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "").strip() for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return [req for req in requirements if req and not req.startswith("#")]


setup(
    name="sumarg_ml_engine",
    version="0.0.1",
    author="PARRAS",
    author_email="",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)