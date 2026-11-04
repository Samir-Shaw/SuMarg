import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from ml_engine.src.exception import CustomException
from ml_engine.src.logger import logging
from ml_engine.src.utils import load_object
from traci_core.config import ROOT_DIR


class PredictPipeline:
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        try:
            if model_path is None:
                self.model_path = ROOT_DIR / "ml_engine" / "artifacts" / "model.pkl"
            else:
                self.model_path = Path(model_path)

            if preprocessor_path is None:
                self.preprocessor_path = ROOT_DIR / "ml_engine" / "artifacts" / "preprocessor.pkl"
            else:
                self.preprocessor_path = Path(preprocessor_path)

            self.model = load_object(str(self.model_path))
            self.preprocessor = load_object(str(self.preprocessor_path))
            logging.info("Prediction pipeline loaded model.pkl and preprocessor.pkl successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Input DataFrame par preprocessor transform apply karke target speed predict karta hai.
        """
        try:
            scaled_features = self.preprocessor.transform(features)
            predictions = self.model.predict(scaled_features)
            
            # Physics Guard: Negative speed nahi ho sakti
            return np.clip(predictions, 0.0, None)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_target_speed(
        self, 
        ego_speed: float, 
        leader_speed: float, 
        leader_distance: float, 
        relative_speed: float = None, 
        time_to_collision: float = None, 
        speed_limit: float = 13.89,
        v2v_emergency_flag: int = 0
    ) -> float:
        """
        Direct TraCI simulation controller call ke liye method, supporting V2V safety flag.
        """
        try:
            if relative_speed is None:
                relative_speed = ego_speed - leader_speed

            if time_to_collision is None:
                if relative_speed > 0.05 and leader_distance > 0:
                    time_to_collision = min(leader_distance / relative_speed, 99.0)
                else:
                    time_to_collision = 99.0

            input_dict = {
                "vehicle_speed": float(ego_speed),
                "leader_speed": float(leader_speed),
                "leader_distance": float(leader_distance),
                "relative_speed": float(relative_speed),
                "time_to_collision": float(time_to_collision),
                "speed_limit": float(speed_limit),
                "v2v_emergency_flag": int(v2v_emergency_flag) # <--- Naya V2V feature added
            }

            input_df = pd.DataFrame([input_dict])

            pred_speed = self.predict(input_df)[0]
            return float(pred_speed)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    predictor = PredictPipeline()

    # Test 1: Emergency Braking Zone with V2V Active
    emergency_speed = predictor.predict_target_speed(
        ego_speed=12.0,
        leader_speed=2.0,
        leader_distance=4.0,
        v2v_emergency_flag=1
    )
    print(f"[Test 1 - V2V Emergency Stop] Distance: 4m -> Target Speed: {emergency_speed:.2f} m/s")

    # Test 2: Safe Cruising
    cruise_speed = predictor.predict_target_speed(
        ego_speed=10.0,
        leader_speed=13.89,
        leader_distance=80.0,
        v2v_emergency_flag=0
    )
    print(f"[Test 2 - Safe Cruising]  Distance: 80m -> Target Speed: {cruise_speed:.2f} m/s")