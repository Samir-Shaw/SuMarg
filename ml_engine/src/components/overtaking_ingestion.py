import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_engine.src.exception import CustomException
from ml_engine.src.logger import logging
from traci_core.config import ROOT_DIR, DATASET_PROCESSED_DIR


@dataclass
class OvertakingIngestionConfig:
    raw_data_path: Path = ROOT_DIR / "ml_engine" / "data" / "raw" / "overtaking_data.csv"
    processed_dir: Path = DATASET_PROCESSED_DIR
    train_data_path: Path = DATASET_PROCESSED_DIR / "overtake_train.csv"
    test_data_path: Path = DATASET_PROCESSED_DIR / "overtake_test.csv"
    test_size: float = 0.2
    random_state: int = 42


class OvertakingDataIngestion:
    def __init__(self):
        self.ingestion_config = OvertakingIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Overtaking Data Ingestion component started.")
        try:
            if not self.ingestion_config.raw_data_path.exists():
                raise FileNotFoundError(
                    f"Raw overtaking dataset not found at: {self.ingestion_config.raw_data_path}"
                )

            logging.info(f"Reading raw overtaking data from: {self.ingestion_config.raw_data_path}")
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Raw overtaking data shape: {df.shape}")

            self.ingestion_config.processed_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Stratified splitting overtaking data: test_size = {self.ingestion_config.test_size}")
            train_df, test_df = train_test_split(
                df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state,
                stratify=df["decision"]
            )

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Overtake train data saved: {self.ingestion_config.train_data_path} ({len(train_df)} rows)")
            logging.info(f"Overtake test data saved: {self.ingestion_config.test_data_path} ({len(test_df)} rows)")
            logging.info("Overtaking Data Ingestion completed successfully.")

            return str(self.ingestion_config.train_data_path), str(self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = OvertakingDataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"\n[SUCCESS] Overtake Train file: {train_path}")
    print(f"[SUCCESS] Overtake Test file:  {test_path}")