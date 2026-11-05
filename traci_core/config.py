from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
# simulation paths
SIM_CONFIG_PATH = ROOT_DIR / "simulation" / "osm.sumocfg"

# ml engine data paths
DATASET_RAW_PATH = ROOT_DIR / "ml_engine" / "data" / "raw" / "braking_data.csv"
DATASET_PROCESSED_DIR = ROOT_DIR / "ml_engine" / "data" / "processed"

# ml engine artifacts paths
ARTIFACTS_DIR = ROOT_DIR / "ml_engine" / "artifacts"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
STATUS_PATH = ARTIFACTS_DIR / "status.txt"

STEP_LENGTH = 0.1  

EGO_VEHICLE_ID = "ego_car"
MAX_SPEED = 22.0       
EMERGENCY_DECEL = 9.0  

DEFAULT_VIEW_ID = "View #0"
EGO_CAMERA_ZOOM = 1200