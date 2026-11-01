from pathlib import Path

# Automatically detect 'SuMarg/' project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Simulation Paths
SIM_CONFIG_PATH = ROOT_DIR / "simulation" / "osm.sumocfg"

# ML Engine Paths
DATASET_RAW_PATH = ROOT_DIR / "ml_engine" / "data" / "raw" / "braking_data.csv"
DATASET_PROCESSED_DIR = ROOT_DIR / "ml_engine" / "data" / "processed"
BRAKE_MODEL_PATH = ROOT_DIR / "ml_engine" / "artifacts" / "models" / "braking_lgbm_v1.pkl"

# Simulation Hyperparameters
STEP_LENGTH = 0.1  # 100ms lockstep simulation tick

# Ego Vehicle Constants
EGO_VEHICLE_ID = "ego_car"
MAX_SPEED = 22.0   # ~80 km/h
EMERGENCY_DECEL = 9.0  # m/s^2 emergency braking limit