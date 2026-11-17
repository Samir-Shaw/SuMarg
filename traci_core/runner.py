import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import traci
from traci_core.config import SIM_CONFIG_PATH, STEP_LENGTH, EGO_VEHICLE_ID
from traci_core.controllers.longitudinal_controller import LongitudinalController
from traci_core.controllers.overtaking_controller import OvertakingController
from traci_core.collectors.sensor_utils import SensorUtils
from traci_core.collectors.v2v_module import V2VModule
from traci_core.scenarios.emergency_braking import EmergencyBrakingScenario
from traci_core.scenarios.smart_overtaking import SmartOvertakingScenario
from ml_engine.src.pipeline.predict_pipeline import PredictPipeline
from ml_engine.src.pipeline.overtaking_predict_pipeline import (
    OvertakingPredictPipeline,
    OvertakingCustomData
)
from ml_engine.src.logger import logging

MAX_STEPS = 450


def parse_args():
    parser = argparse.ArgumentParser(description="SuMarg CAV Simulation Runner")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["braking", "overtaking"],
        default="overtaking",
        help="Run 'braking' (Function A: Longitudinal ACC & Auto-Brake) or 'overtaking' (Function B: Smart Overtaking)"
    )
    return parser.parse_args()


def run(scenario_mode: str = "overtaking"):
    logging.info(f"Starting SuMarg Simulation with V2V Sensor Fusion [{scenario_mode.upper()} MODE]...")
    predictor = PredictPipeline()
    overtaking_predictor = OvertakingPredictPipeline() if scenario_mode == "overtaking" else None
    v2v = V2VModule(comm_range_meters=150.0)

    LEAD_VEHICLE_ID = "lead_car"
    BLOCKER_VEHICLE_ID = "blocker_car"

    if scenario_mode == "braking":
        scenario = EmergencyBrakingScenario(
            lead_id=LEAD_VEHICLE_ID,
            trigger_step=60,
            stop_duration=40,
            resume_speed=11.0
        )
    else:
        scenario = SmartOvertakingScenario(
            ego_id=EGO_VEHICLE_ID,
            lead_id=LEAD_VEHICLE_ID,
            blocker_id=BLOCKER_VEHICLE_ID,
            trigger_step=60,
            clear_left_step=115,
            stop_duration=35,
            resume_speed=7.5
        )

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
        while traci.simulation.getMinExpectedNumber() > 0 and step_count < MAX_STEPS:
            traci.simulationStep()
            step_count += 1
            active_vehicles = traci.vehicle.getIDList()

            # Front lead car lane lock: Taaki lead vehicle spontaneous lane switch na kare
            if LEAD_VEHICLE_ID in active_vehicles:
                traci.vehicle.setLaneChangeMode(LEAD_VEHICLE_ID, 0)

            # Scenario updates
            scenario.update(current_step=step_count, active_vehicles=active_vehicles, v2v_module=v2v)

            if EGO_VEHICLE_ID in active_vehicles:
                if not initialized:
                    LongitudinalController.initialize_vehicle(EGO_VEHICLE_ID)
                    OvertakingController.initialize_vehicle(EGO_VEHICLE_ID)
                    traci.vehicle.setLaneChangeMode(EGO_VEHICLE_ID, 0)

                    try:
                        traci.gui.trackVehicle("View #0", EGO_VEHICLE_ID)
                        traci.gui.setZoom("View #0", 1200)
                    except traci.TraCIException as gui_err:
                        logging.warning(f"Camera lock: {gui_err}")

                    initialized = True

                ego_pos = traci.vehicle.getPosition(EGO_VEHICLE_ID)
                ego_lane = traci.vehicle.getLaneIndex(EGO_VEHICLE_ID)
                ego_edge = traci.vehicle.getRoadID(EGO_VEHICLE_ID)

                telemetry = SensorUtils.get_vehicle_telemetry(EGO_VEHICLE_ID)
                if not telemetry:
                    v2v.step_cleanup()
                    continue

                ego_speed = telemetry["speed"]
                speed_limit = telemetry["speed_limit"]

                #Radar detection for same-lane leader
                radar = SensorUtils.detect_leader(EGO_VEHICLE_ID, radar_range=100.0, current_speed=ego_speed)

                #Same-lane V2V validation
                raw_v2v_alerts = v2v.receive_for_vehicle(EGO_VEHICLE_ID, ego_pos)
                v2v_emergency = False

                for msg in raw_v2v_alerts:
                    if msg.get("event") == "EMERGENCY_BRAKE":
                        sender_id = msg.get("sender_id", LEAD_VEHICLE_ID)
                        if sender_id in active_vehicles:
                            sender_lane = traci.vehicle.getLaneIndex(sender_id)
                            sender_edge = traci.vehicle.getRoadID(sender_id)

                            is_same_corridor = (sender_edge == ego_edge and sender_lane == ego_lane)
                            is_radar_confirmed = (radar.get("leader_id") == sender_id)

                            if is_same_corridor or is_radar_confirmed:
                                v2v_emergency = True
                                break

                v2v_flag = 1 if v2v_emergency else 0

                if v2v_emergency:
                    logging.info("V2V SAME-LANE ALERT DETECTED: Passing emergency flag=1 to ML model.")

                #1.Lateral Smart Overtaking Inference (Function B)
                overtake_triggered = False
                if scenario_mode == "overtaking" and overtaking_predictor is not None:
                    overtake_features = SensorUtils.get_overtaking_features_for_ml(
                        EGO_VEHICLE_ID, active_vehicles=active_vehicles, radar_range=100.0, safety_margin=18.0
                    )
                    if overtake_features is not None:
                        if v2v_emergency and radar.get("has_leader"):
                            overtake_features["lead_speed"] = 0.0

                        same_lane_leader = traci.vehicle.getLeader(EGO_VEHICLE_ID, 45.0)
                        if same_lane_leader is not None:
                            leader_id, leader_dist = same_lane_leader
                            actual_leader_speed = traci.vehicle.getSpeed(leader_id)
                            has_obstacle_ahead = (
                                leader_dist < 45.0
                                and (actual_leader_speed < (speed_limit - 2.0) or v2v_emergency)
                            )
                        else:
                            has_obstacle_ahead = False

                        if has_obstacle_ahead:
                            custom_data = OvertakingCustomData(
                                ego_speed=overtake_features["ego_speed"],
                                lead_distance=overtake_features["lead_distance"],
                                lead_speed=overtake_features["lead_speed"],
                                left_lane_available=overtake_features["left_lane_available"],
                                right_lane_available=overtake_features["right_lane_available"],
                                left_gap=overtake_features["left_gap"],
                                right_gap=overtake_features["right_gap"]
                            )
                            decision = overtaking_predictor.predict(custom_data.get_data_as_dataframe())
                        else:
                            decision = 0

                        overtake_triggered = OvertakingController.apply_decision(
                            ego_id=EGO_VEHICLE_ID,
                            decision=decision,
                            current_lane=overtake_features["current_lane"],
                            total_lanes=overtake_features["total_lanes"],
                            current_step=step_count
                        )

                #2.Longitudinal Speed / Emergency Braking Control (Function A)
                if overtake_triggered:
                    LongitudinalController.apply_control(EGO_VEHICLE_ID, target_speed=speed_limit, duration=1.0)
                else:
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
                    logging.info("Ego vehicle completed route. Terminating simulation.")
                    break

            v2v.step_cleanup()

    except traci.TraCIException as e:
        logging.error(f"TraCI Runtime Exception: {e}")
    finally:
        traci.close()
        logging.info("Simulation closed cleanly.")


if __name__ == "__main__":
    args = parse_args()
    run(scenario_mode=args.mode)