import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import traci
from traci_core.config import SIM_CONFIG_PATH, STEP_LENGTH, EGO_VEHICLE_ID
from traci_core.controllers.longitudinal_controller import LongitudinalController
from traci_core.collectors.sensor_utils import SensorUtils
from traci_core.collectors.v2v_module import V2VModule
from traci_core.scenarios import EmergencyBrakingScenario
from ml_engine.src.pipeline.predict_pipeline import PredictPipeline
from ml_engine.src.logger import logging


def run():
    logging.info("Starting SuMarg Simulation with V2V Sensor Fusion...")
    predictor = PredictPipeline()
    v2v = V2VModule(comm_range_meters=150.0)
    
    scenario = EmergencyBrakingScenario(lead_id="lead_car", trigger_step=50, stop_duration=40, resume_speed=11.0)

    sumo_cmd = [
        "sumo-gui",
        "-c", str(SIM_CONFIG_PATH),
        "--step-length", str(STEP_LENGTH),
        "--start", "true",
        "--delay", "80"
    ]
    traci.start(sumo_cmd)

    initialized = False
    step_count = 0

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step_count += 1
            active_vehicles = traci.vehicle.getIDList()
            

            scenario.update(current_step=step_count, active_vehicles=active_vehicles, v2v_module=v2v)

            if EGO_VEHICLE_ID in active_vehicles:
                if not initialized:
                    LongitudinalController.initialize_vehicle(EGO_VEHICLE_ID)
                    traci.vehicle.setSpeedMode(EGO_VEHICLE_ID, 31)
                    traci.vehicle.setLaneChangeMode(EGO_VEHICLE_ID, 0)  #Automatic lane change disabled

                    try:
                        traci.gui.trackVehicle("View #0", EGO_VEHICLE_ID)
                        traci.gui.setZoom("View #0", 1200)
                    except traci.TraCIException as gui_err:
                        logging.warning(f"Camera lock: {gui_err}")
                        
                    initialized = True

                ego_pos = traci.vehicle.getPosition(EGO_VEHICLE_ID)
                telemetry = SensorUtils.get_vehicle_telemetry(EGO_VEHICLE_ID)
                if not telemetry:
                    v2v.step_cleanup()
                    continue

                ego_speed = telemetry["speed"]
                speed_limit = telemetry["speed_limit"]

                v2v_alerts = v2v.receive_for_vehicle(EGO_VEHICLE_ID, ego_pos)
                v2v_emergency = any(msg["event"] == "EMERGENCY_BRAKE" for msg in v2v_alerts)
                v2v_flag = 1 if v2v_emergency else 0

                if v2v_emergency:
                    logging.info("V2V ALERT DETECTED: Passing emergency flag=1 to ML model.")

                radar = SensorUtils.detect_leader(EGO_VEHICLE_ID, radar_range=100.0, current_speed=ego_speed)

                target_speed = predictor.predict_target_speed(
                    ego_speed=ego_speed,
                    leader_speed=radar["leader_speed"],
                    leader_distance=radar["gap"],
                    relative_speed=radar["relative_speed"],
                    time_to_collision=radar["ttc"],
                    speed_limit=speed_limit,
                    v2v_emergency_flag=v2v_flag  
                )

                LongitudinalController.apply_control(EGO_VEHICLE_ID, target_speed)

            else:
                if initialized:
                    logging.info("Ego vehicle completed route.")
                    break

            v2v.step_cleanup()

    except traci.TraCIException as e:
        logging.error(f"TraCI Runtime Exception: {e}")
    finally:
        traci.close()
        logging.info("Simulation closed cleanly.")


if __name__ == "__main__":
    run()