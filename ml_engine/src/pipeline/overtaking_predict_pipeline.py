import sys
import pandas as pd

from ml_engine.src.exception import CustomException
from ml_engine.src.logger import logging
from ml_engine.src.utils import load_object
from traci_core.config import ROOT_DIR


class OvertakingPredictPipeline:
    def __init__(self):
        self.model_path = ROOT_DIR / "ml_engine" / "artifacts" / "overtaking_model.pkl"
        self.preprocessor_path = ROOT_DIR / "ml_engine" / "artifacts" / "overtaking_preprocessor.pkl"

        self.model = load_object(file_path=str(self.model_path))
        self.preprocessor = load_object(file_path=str(self.preprocessor_path))
        logging.info("OvertakingPredictPipeline initialized with saved artifacts.")

    def predict(self, features: pd.DataFrame) -> int:
        try:
            scaled_data = self.preprocessor.transform(features)
            prediction = self.model.predict(scaled_data)
            return int(prediction[0])
        except Exception as e:
            raise CustomException(e, sys)


class OvertakingCustomData:
    def __init__(
        self,
        ego_speed: float,
        lead_distance: float,
        lead_speed: float,
        left_lane_available: int,
        right_lane_available: int,
        left_gap: float,
        right_gap: float
    ):
        self.ego_speed = ego_speed
        self.lead_distance = lead_distance
        self.lead_speed = lead_speed
        self.left_lane_available = left_lane_available
        self.right_lane_available = right_lane_available
        self.left_gap = left_gap
        self.right_gap = right_gap

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "ego_speed": [float(self.ego_speed)],
                "lead_distance": [float(self.lead_distance)],
                "lead_speed": [float(self.lead_speed)],
                "left_lane_available": [int(self.left_lane_available)],
                "right_lane_available": [int(self.right_lane_available)],
                "left_gap": [float(self.left_gap)],
                "right_gap": [float(self.right_gap)],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data = OvertakingCustomData(
        ego_speed=12.5,
        lead_distance=6.2,
        lead_speed=4.0,
        left_lane_available=1,
        right_lane_available=0,
        left_gap=45.0,
        right_gap=0.0
    )
    pred_df = data.get_data_as_dataframe()
    pipeline = OvertakingPredictPipeline()
    decision = pipeline.predict(pred_df)

    decision_map = {0: "KEEP_LANE", 1: "OVERTAKE_LEFT", 2: "OVERTAKE_RIGHT"}
    print(f"Predicted Overtaking Decision: {decision} ({decision_map[decision]})")