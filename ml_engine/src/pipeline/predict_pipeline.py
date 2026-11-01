# Step 3: Load .pkl model and predict target_speed
import os
import joblib
import numpy as np

class RiskPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.model_path = os.path.join(base_dir, "artifacts", "models", "braking_lgbm_v1.pkl")
        else:
            self.model_path = model_path
            
        self.model = joblib.load(self.model_path)

    def predict_target_speed(self, ego_speed, front_speed, gap, rel_speed):
        features = np.array([[ego_speed, front_speed, gap, rel_speed]])
        return float(self.model.predict(features)[0])