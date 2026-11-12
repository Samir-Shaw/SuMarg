import sys
import json
from pathlib import Path
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from ml_engine.src.exception import CustomException
from ml_engine.src.logger import logging
from ml_engine.src.utils import save_object, evaluate_classifiers
from traci_core.config import ROOT_DIR


@dataclass
class OvertakingTrainerConfig:
    trained_model_file_path: Path = ROOT_DIR / "ml_engine" / "artifacts" / "overtaking_model.pkl"
    metrics_file_path: Path = ROOT_DIR / "ml_engine" / "artifacts" / "overtaking_metrics.json"


class OvertakingModelTrainer:
    def __init__(self):
        self.model_trainer_config = OvertakingTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting transformed overtaking train and test arrays into X and y.")
            X_train, y_train = train_array[:, :-1], train_array[:, -1].astype(int)
            X_test, y_test = test_array[:, :-1], test_array[:, -1].astype(int)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
                "K-Neighbors Classifier": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
                "Decision Tree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(n_estimators=100, eval_metric="mlogloss", random_state=42, n_jobs=-1),
                "CatBoost": CatBoostClassifier(iterations=150, auto_class_weights="Balanced", verbose=0, random_seed=42),
                "LightGBM": LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1)
            }

            logging.info(f"Initiating evaluation across {len(models)} candidate classifiers via utils.evaluate_classifiers.")
            model_report = evaluate_classifiers(X_train, y_train, X_test, y_test, models=models)

            # Best model selection based on highest Macro F1 Score
            best_model_name = max(model_report, key=lambda k: model_report[k]["macro_f1"])
            best_model_details = model_report[best_model_name]
            best_model_obj = best_model_details["model_obj"]

            if best_model_details["macro_f1"] < 0.80:
                raise CustomException("No best overtaking model found with acceptable Macro F1 score (> 0.80)", sys)

            logging.info(
                f"Best Overtaking Model Selected: {best_model_name} "
                f"with Macro F1: {best_model_details['macro_f1']:.4f} | Accuracy: {best_model_details['accuracy']:.4f}"
            )

            # Save best model artifact (overtaking_model.pkl)
            save_object(
                file_path=str(self.model_trainer_config.trained_model_file_path),
                obj=best_model_obj
            )

            # Save evaluation metrics (overtaking_metrics.json)
            metrics_summary = {
                "best_model": best_model_name,
                "accuracy": best_model_details["accuracy"],
                "macro_f1": best_model_details["macro_f1"],
                "weighted_f1": best_model_details["weighted_f1"],
                "macro_precision": best_model_details["macro_precision"],
                "macro_recall": best_model_details["macro_recall"],
                "confusion_matrix": best_model_details["confusion_matrix"],
                "classification_report": best_model_details["classification_report"],
                "all_models": {
                    name: {
                        k: v for k, v in info.items()
                        if k not in ("model_obj", "classification_report", "confusion_matrix")
                    }
                    for name, info in model_report.items()
                }
            }

            with open(self.model_trainer_config.metrics_file_path, "w", encoding="utf-8") as f:
                json.dump(metrics_summary, f, indent=4)

            logging.info(f"Trained overtaking model saved at: {self.model_trainer_config.trained_model_file_path}")
            logging.info(f"Overtaking metrics saved at: {self.model_trainer_config.metrics_file_path}")

            return best_model_name, best_model_details["macro_f1"]

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from ml_engine.src.components.overtaking_transformation import OvertakingDataTransformation

    transformation = OvertakingDataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation()

    trainer = OvertakingModelTrainer()
    best_model, score = trainer.initiate_model_trainer(train_arr, test_arr)
    print("\n--- Overtaking Pipeline Execution Summary ---")
    print(f"Status:     Training Succeeded")
    print(f"Best Model: {best_model}")
    print(f"Macro F1:   {score:.4f}")