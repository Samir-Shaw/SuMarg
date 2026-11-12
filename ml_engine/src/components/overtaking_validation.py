import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import yaml

from ml_engine.src.exception import CustomException
from ml_engine.src.logger import logging
from traci_core.config import ROOT_DIR, DATASET_PROCESSED_DIR


@dataclass
class OvertakingValidationConfig:
    schema_file_path: Path = ROOT_DIR / "ml_engine" / "config" / "overtaking_schema.yaml"
    status_file_path: Path = ROOT_DIR / "ml_engine" / "artifacts" / "overtaking_status.txt"
    train_data_path: Path = DATASET_PROCESSED_DIR / "overtake_train.csv"
    test_data_path: Path = DATASET_PROCESSED_DIR / "overtake_test.csv"


class OvertakingDataValidation:
    def __init__(self):
        self.validation_config = OvertakingValidationConfig()

    def _read_schema(self) -> dict:
        try:
            with open(self.validation_config.schema_file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_all_columns(self) -> bool:
        logging.info("Overtaking Data Validation component started.")
        try:
            validation_status = True
            schema = self._read_schema()
            expected_columns = list(schema.get("COLUMNS", {}).keys())

            if not self.validation_config.train_data_path.exists() or not self.validation_config.test_data_path.exists():
                raise FileNotFoundError("Processed overtaking train/test files not found.")

            train_df = pd.read_csv(self.validation_config.train_data_path)
            test_df = pd.read_csv(self.validation_config.test_data_path)

            train_cols = list(train_df.columns)
            test_cols = list(test_df.columns)

            logging.info(f"Expected overtaking schema columns ({len(expected_columns)}): {expected_columns}")

            if len(train_cols) != len(expected_columns) or len(test_cols) != len(expected_columns):
                validation_status = False
                logging.error(
                    f"Column count mismatch! Expected {len(expected_columns)}, "
                    f"got train={len(train_cols)}, test={len(test_cols)}"
                )

            for col in expected_columns:
                if col not in train_cols or col not in test_cols:
                    validation_status = False
                    logging.error(f"Column [{col}] missing in overtaking datasets.")

            # Null values check
            if train_df.isnull().sum().sum() > 0 or test_df.isnull().sum().sum() > 0:
                validation_status = False
                logging.error("Null values detected in overtaking datasets.")

            # Valid target classes check (only 0, 1, 2 allowed)
            if not set(train_df["decision"].unique()).issubset({0, 1, 2}) or not set(test_df["decision"].unique()).issubset({0, 1, 2}):
                validation_status = False
                logging.error("Invalid class labels found in 'decision' column.")

            self.validation_config.status_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.validation_config.status_file_path, "w", encoding="utf-8") as f:
                f.write(f"Validation status: {validation_status}\n")

            logging.info(
                f"Overtaking Data Validation status: {validation_status}. "
                f"Written to: {self.validation_config.status_file_path}"
            )
            return validation_status

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    validator = OvertakingDataValidation()
    status = validator.validate_all_columns()
    print(f"\n[OVERTAKING VALIDATION RESULT]: {'SUCCESS (True)' if status else 'FAILED (False)'}")