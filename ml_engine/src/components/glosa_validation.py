import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import yaml

from ml_engine.src.exception import CustomException
from ml_engine.src.logger import logging
from traci_core.config import ROOT_DIR, DATASET_PROCESSED_DIR


@dataclass
class GlosaValidationConfig:
    schema_file_path: Path = ROOT_DIR / "ml_engine" / "config" / "glosa_schema.yaml"
    status_file_path: Path = ROOT_DIR / "ml_engine" / "artifacts" / "glosa_status.txt"
    train_data_path: Path = DATASET_PROCESSED_DIR / "glosa_train.csv"
    test_data_path: Path = DATASET_PROCESSED_DIR / "glosa_test.csv"


class GlosaDataValidation:
    def __init__(self):
        self.validation_config = GlosaValidationConfig()

    def _read_schema(self) -> dict:
        try:
            with open(self.validation_config.schema_file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_all_columns(self) -> bool:
        logging.info("GLOSA Data Validation component started.")
        try:
            validation_status = True
            schema = self._read_schema()
            expected_columns = list(schema.get("COLUMNS", {}).keys())

            if not self.validation_config.train_data_path.exists() or not self.validation_config.test_data_path.exists():
                raise FileNotFoundError("Processed GLOSA train/test files not found.")

            train_df = pd.read_csv(self.validation_config.train_data_path)
            test_df = pd.read_csv(self.validation_config.test_data_path)

            train_cols = list(train_df.columns)
            test_cols = list(test_df.columns)

            logging.info(f"Expected GLOSA schema columns ({len(expected_columns)}): {expected_columns}")

            if len(train_cols) != len(expected_columns) or len(test_cols) != len(expected_columns):
                validation_status = False
                logging.error(
                    f"Column count mismatch! Expected {len(expected_columns)}, "
                    f"got train={len(train_cols)}, test={len(test_cols)}"
                )

            for col in expected_columns:
                if col not in train_cols or col not in test_cols:
                    validation_status = False
                    logging.error(f"Column [{col}] missing in GLOSA datasets.")

            # Null values check
            if train_df.isnull().sum().sum() > 0 or test_df.isnull().sum().sum() > 0:
                validation_status = False
                logging.error("Null values detected in GLOSA datasets.")

            # Valid signal phase check (only 0=RED, 1=YELLOW, 2=GREEN allowed)
            if not set(train_df["signal_phase"].unique()).issubset({0, 1, 2}) or not set(test_df["signal_phase"].unique()).issubset({0, 1, 2}):
                validation_status = False
                logging.error("Invalid phase values found in 'signal_phase' column.")

            # Physical bounds check on target speed
            if (train_df["optimal_glide_speed"] < 0.0).any() or (test_df["optimal_glide_speed"] < 0.0).any():
                validation_status = False
                logging.error("Negative values found in 'optimal_glide_speed' column.")

            self.validation_config.status_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.validation_config.status_file_path, "w", encoding="utf-8") as f:
                f.write(f"Validation status: {validation_status}\n")

            logging.info(
                f"GLOSA Data Validation status: {validation_status}. "
                f"Written to: {self.validation_config.status_file_path}"
            )
            return validation_status

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    validator = GlosaDataValidation()
    status = validator.validate_all_columns()
    print(f"\n[GLOSA VALIDATION RESULT]: {'SUCCESS (True)' if status else 'FAILED (False)'}")