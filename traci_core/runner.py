import sys
from pathlib import Path

# Project root directory ko Python path mein add karna
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import traci
from traci_core.config import SIM_CONFIG_PATH, STEP_LENGTH, EGO_VEHICLE_ID, MAX_SPEED
from traci_core.controllers.longitudinal_controller import LongitudinalController
from traci_core.collectors.sensor_utils import SensorUtils
from ml_engine.src.pipeline.predict_pipeline import PredictPipeline
from ml_engine.src.logger import logging


def run():
    # 1. Pipeline aur Model Load Karein
    logging.info("Starting SuMarg CAV Control Loop...")
    predictor = PredictPipeline()

    # 2. SUMO GUI Simulation Start Karein
    sumo_cmd = [
        "sumo-gui",
        "-c", str(SIM_CONFIG_PATH),
        "--step-length", str(STEP_LENGTH),
        "--start", "true"
    ]
    traci.start(sumo_cmd)

    initialized = False

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # Ego car network par appear hone ka wait karein
            if EGO_VEHICLE_ID in traci.vehicle.getIDList():
                # One-time safe speed-mode initialization
                if not initialized:
                    LongitudinalController.initialize_vehicle(EGO_VEHICLE_ID)
                    initialized = True

                # Step A: Telemetry & Radar Data Fetch via SensorUtils
                telemetry = SensorUtils.get_vehicle_telemetry(EGO_VEHICLE_ID)
                if not telemetry:
                    continue

                ego_speed = telemetry["speed"]
                speed_limit = telemetry["speed_limit"]
                
                # Radar detection (leader vehicle)
                radar = SensorUtils.detect_leader(EGO_VEHICLE_ID, radar_range=100.0, current_speed=ego_speed)
                
                lead_speed = radar["leader_speed"]
                gap = radar["gap"]
                rel_speed = radar["relative_speed"]
                ttc = radar["ttc"]

                # Step B: ML Model Inference (Target Speed Prediction)
                target_speed = predictor.predict_target_speed(
                    ego_speed=ego_speed,
                    leader_speed=lead_speed,
                    leader_distance=gap,
                    relative_speed=rel_speed,
                    time_to_collision=ttc,
                    speed_limit=speed_limit
                )

                # Step C: Actuator Control (Smooth Braking / ACC / Stop)
                LongitudinalController.apply_control(EGO_VEHICLE_ID, target_speed)

    except traci.TraCIException as e:
        logging.error(f"TraCI Runtime Exception: {e}")
    finally:
        traci.close()
        logging.info("SUMO Simulation closed cleanly.")


if __name__ == "__main__":
    run()