import sys
import json
from pathlib import Path
from dataclasses import dataclass

from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from ml_engine.src.exception import CustomException
from ml_engine.src.logger import logging
from ml_engine.src.utils import save_object
from traci_core.config import ROOT_DIR


@dataclass
class GlosaTrainerConfig:
    trained_model_file_path: Path = ROOT_DIR / "ml_engine" / "artifacts" / "glosa_model.pkl"
    metrics_file_path: Path = ROOT_DIR / "ml_engine" / "artifacts" / "glosa_metrics.json"


class GlosaModelTrainer:
    def __init__(self):
        self.model_trainer_config = GlosaTrainerConfig()

    def _evaluate_regressors(self, X_train, y_train, X_test, y_test, models: dict) -> dict:
        report = {}
        for name, model in models.items():
            logging.info(f"Training GLOSA candidate regressor: {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = float(r2_score(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(root_mean_squared_error(y_test, y_pred))

            report[name] = {
                "r2_score": round(r2, 4),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "model_obj": model,
            }
            logging.info(f"[{name}] -> R2: {r2:.4f} | MAE: {mae:.4f} m/s | RMSE: {rmse:.4f} m/s")
        return report

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting transformed GLOSA train and test arrays into X and y.")
            X_train, y_train = train_array[:, :-1], train_array[:, -1].astype(float)
            X_test, y_test = test_array[:, :-1], test_array[:, -1].astype(float)

            models = {
                "Ridge Regression": Ridge(alpha=1.0, random_state=42),
                "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, random_state=42),
                "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.08, random_state=42, n_jobs=-1),
                "CatBoost": CatBoostRegressor(iterations=200, learning_rate=0.08, verbose=0, random_seed=42),
                "LightGBM": LGBMRegressor(n_estimators=150, learning_rate=0.08, random_state=42, verbose=-1, n_jobs=-1),
            }

            logging.info(f"Initiating evaluation across {len(models)} candidate regressors for GLOSA.")
            model_report = self._evaluate_regressors(X_train, y_train, X_test, y_test, models=models)

            best_model_name = max(model_report, key=lambda k: model_report[k]["r2_score"])
            best_model_details = model_report[best_model_name]
            best_model_obj = best_model_details["model_obj"]

            if best_model_details["r2_score"] < 0.85:
                raise CustomException("No best GLOSA model found with acceptable R2 score (> 0.85)", sys)

            logging.info(
                f"Best GLOSA Model Selected: {best_model_name} "
                f"with R2: {best_model_details['r2_score']:.4f} | MAE: {best_model_details['mae']:.4f} m/s"
            )

            save_object(
                file_path=str(self.model_trainer_config.trained_model_file_path),
                obj=best_model_obj
            )

            metrics_summary = {
                "best_model": best_model_name,
                "r2_score": best_model_details["r2_score"],
                "mae": best_model_details["mae"],
                "rmse": best_model_details["rmse"],
                "all_models": {
                    name: {k: v for k, v in info.items() if k != "model_obj"}
                    for name, info in model_report.items()
                }
            }

            with open(self.model_trainer_config.metrics_file_path, "w", encoding="utf-8") as f:
                json.dump(metrics_summary, f, indent=4)

            logging.info(f"Trained GLOSA model saved at: {self.model_trainer_config.trained_model_file_path}")
            logging.info(f"GLOSA metrics saved at: {self.model_trainer_config.metrics_file_path}")

            return best_model_name, best_model_details["r2_score"], best_model_details["mae"]

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from ml_engine.src.components.glosa_transformation import GlosaDataTransformation

    transformation = GlosaDataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation()

    trainer = GlosaModelTrainer()
    best_model, r2, mae = trainer.initiate_model_trainer(train_arr, test_arr)
    print("\n--- GLOSA Pipeline Execution Summary ---")
    print(f"Status:     Training Succeeded")
    print(f"Best Model: {best_model}")
    print(f"R2 Score:   {r2:.4f}")
    print(f"MAE:        {mae:.4f} m/s")