import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_engine.src.exception import CustomException
from ml_engine.src.logger import logging
from traci_core.config import ROOT_DIR, DATASET_PROCESSED_DIR


@dataclass
class GlosaIngestionConfig:
    raw_data_path: Path = ROOT_DIR / "ml_engine" / "data" / "raw" / "glosa_data.csv"
    processed_dir: Path = DATASET_PROCESSED_DIR
    train_data_path: Path = DATASET_PROCESSED_DIR / "glosa_train.csv"
    test_data_path: Path = DATASET_PROCESSED_DIR / "glosa_test.csv"
    test_size: float = 0.2
    random_state: int = 42


class GlosaDataIngestion:
    def __init__(self):
        self.ingestion_config = GlosaIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("GLOSA Data Ingestion component started.")
        try:
            if not self.ingestion_config.raw_data_path.exists():
                raise FileNotFoundError(
                    f"Raw GLOSA dataset not found at: {self.ingestion_config.raw_data_path}"
                )

            logging.info(f"Reading raw GLOSA data from: {self.ingestion_config.raw_data_path}")
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Raw GLOSA data shape: {df.shape}")

            self.ingestion_config.processed_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Stratified splitting GLOSA data by signal_phase: test_size = {self.ingestion_config.test_size}")
            train_df, test_df = train_test_split(
                df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state,
                stratify=df["signal_phase"]
            )

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"GLOSA train data saved: {self.ingestion_config.train_data_path} ({len(train_df)} rows)")
            logging.info(f"GLOSA test data saved: {self.ingestion_config.test_data_path} ({len(test_df)} rows)")
            logging.info("GLOSA Data Ingestion completed successfully.")

            return str(self.ingestion_config.train_data_path), str(self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = GlosaDataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"\n[SUCCESS] GLOSA Train file: {train_path}")
    print(f"[SUCCESS] GLOSA Test file:  {test_path}")