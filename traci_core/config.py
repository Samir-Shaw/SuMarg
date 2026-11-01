from pathlib import Path

# Automatically detect 'SuMarg/' project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Simulation Paths
SIM_CONFIG_PATH = ROOT_DIR / "simulation" / "osm.sumocfg"

# ML Engine Data Paths
DATASET_RAW_PATH = ROOT_DIR / "ml_engine" / "data" / "raw" / "braking_data.csv"
DATASET_PROCESSED_DIR = ROOT_DIR / "ml_engine" / "data" / "processed"

# ML Engine Artifacts Paths (Directly synced with DataTransformation & ModelTrainer)
ARTIFACTS_DIR = ROOT_DIR / "ml_engine" / "artifacts"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
STATUS_PATH = ARTIFACTS_DIR / "status.txt"

# Simulation Hyperparameters
STEP_LENGTH = 0.1  # 100ms lockstep simulation tick

# Ego Vehicle Constants
EGO_VEHICLE_ID = "ego_car"
MAX_SPEED = 22.0       # ~80 km/h
EMERGENCY_DECEL = 9.0  # m/s^2 emergency braking limit