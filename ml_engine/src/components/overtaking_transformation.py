import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from ml_engine.src.exception import CustomException
from ml_engine.src.logger import logging
from ml_engine.src.utils import save_object
from traci_core.config import ROOT_DIR, DATASET_PROCESSED_DIR


@dataclass
class OvertakingTransformationConfig:
    preprocessor_obj_file_path: Path = ROOT_DIR / "ml_engine" / "artifacts" / "overtaking_preprocessor.pkl"
    status_file_path: Path = ROOT_DIR / "ml_engine" / "artifacts" / "overtaking_status.txt"
    train_data_path: Path = DATASET_PROCESSED_DIR / "overtake_train.csv"
    test_data_path: Path = DATASET_PROCESSED_DIR / "overtake_test.csv"


class OvertakingDataTransformation:
    def __init__(self):
        self.data_transformation_config = OvertakingTransformationConfig()

    def get_data_transformer_object(self) -> Pipeline:
        try:
            pipeline = Pipeline(
                steps=[
                    ("scaler", RobustScaler())
                ]
            )
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def check_validation_status(self) -> bool:
        try:
            status_file = self.data_transformation_config.status_file_path
            if not status_file.exists():
                raise FileNotFoundError(f"Overtaking validation status not found at: {status_file}")

            with open(status_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                status = content.split(":")[-1].strip()

            if status == "True":
                logging.info("Overtaking Data Validation check PASSED. Transformation started.")
                return True
            else:
                logging.error(f"Overtaking Data Validation check FAILED. Status: {status}")
                return False

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        try:
            logging.info("Starting Overtaking Data Transformation component.")

            is_valid = self.check_validation_status()
            if not is_valid:
                raise CustomException("Overtaking data validation failed! Aborting transformation.", sys)

            train_df = pd.read_csv(self.data_transformation_config.train_data_path)
            test_df = pd.read_csv(self.data_transformation_config.test_data_path)

            target_column_name = "decision"
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name].astype(int)

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name].astype(int)

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying RobustScaler preprocessing on overtaking train & test datasets.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=str(self.data_transformation_config.preprocessor_obj_file_path),
                obj=preprocessing_obj
            )
            logging.info(f"Overtaking preprocessor saved at: {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr, str(self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformation = OvertakingDataTransformation()
    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation()
    print("Overtaking Data Transformation Successful.")
    print(f"Train Array Shape: {train_arr.shape}")
    print(f"Test Array Shape:  {test_arr.shape}")
    print(f"Preprocessor Saved: {preprocessor_path}")