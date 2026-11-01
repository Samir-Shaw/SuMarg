import os
import sys
import pickle
import yaml
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ml_engine.src.exception import CustomException
from ml_engine.src.logger import logging


def save_object(file_path: str, obj: object) -> None:
    """
    Saves a python object as a pickle file.
    Creates destination directory automatically if missing.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object successfully saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    """
    Loads a python object from a pickle file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logging.info(f"Object successfully loaded from: {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)


def read_yaml(file_path: Path) -> dict:
    """
    Reads a YAML file safely and returns content as dictionary.
    """
    try:
        with open(file_path, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"YAML file loaded successfully from: {file_path}")
            return content
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict) -> dict:
    """
    Trains each model from dictionary and returns comprehensive evaluation metrics.
    """
    try:
        report = {}
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            r2 = r2_score(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))

            report[model_name] = {
                "model_obj": model,
                "r2_score": float(r2),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4)
            }
            logging.info(f"Trained {model_name} -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        return report
    except Exception as e:
        raise CustomException(e, sys)