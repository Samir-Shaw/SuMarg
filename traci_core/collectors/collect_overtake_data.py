import csv
import sys
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import traci
from traci_core.config import ROOT_DIR, SIM_CONFIG_PATH, STEP_LENGTH, EGO_VEHICLE_ID
from traci_core.collectors.sensor_utils import SensorUtils

OVERTAKE_DATASET_RAW_PATH = ROOT_DIR / "ml_engine" / "data" / "raw" / "overtaking_data.csv"
TOTAL_SAMPLES = 50000


def maintain_lane_without_blocking_turns(veh_id: str, assigned_lane: int):
    edge_id = traci.vehicle.getRoadID(veh_id)
    if not edge_id or edge_id.startswith(":"):
        return

    current_lane = traci.vehicle.getLaneIndex(veh_id)
    lane_id = f"{edge_id}_{current_lane}"
    try:
        lane_len = traci.lane.getLength(lane_id)
        pos = traci.vehicle.getLanePosition(veh_id)

        if pos < (lane_len - 35.0):
            traci.vehicle.setLaneChangeMode(veh_id, 0)
            if current_lane != assigned_lane:
                traci.vehicle.changeLane(veh_id, assigned_lane, 5.0)
        else:
            traci.vehicle.setLaneChangeMode(veh_id, 1621)
    except traci.TraCIException:
        pass


def run_overtaking_data_generation(target_samples: int = TOTAL_SAMPLES):
    OVERTAKE_DATASET_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "ego_speed",
        "lead_distance",
        "lead_speed",
        "left_lane_available",
        "right_lane_available",
        "left_gap",
        "right_gap",
        "decision"
    ]

    sumo_cmd = [
        "sumo",
        "-c", str(SIM_CONFIG_PATH),
        "--step-length", str(STEP_LENGTH),
        "--no-step-log", "true",
        "--collision.action", "none"
    ]

    samples_collected = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    episode = 0

    print(f"Starting Balanced Headless SUMO Collection (Target: {target_samples} samples)...")

    with open(OVERTAKE_DATASET_RAW_PATH, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        while samples_collected < target_samples:
            episode += 1
            traci.start(sumo_cmd)
            step_count = 0
            bg_counter = 0
            obstacle_trigger = random.randint(35, 65)

            try:
                while traci.simulation.getMinExpectedNumber() > 0 and samples_collected < target_samples:
                    traci.simulationStep()
                    step_count += 1
                    active_vehicles = traci.vehicle.getIDList()

                    if step_count > 20 and EGO_VEHICLE_ID not in active_vehicles:
                        break

                    # Dynamically assign lane based on whichever overtake side needs more samples
                    # Lane 0 -> generates OVERTAKE_LEFT (1), Lane 1 -> generates OVERTAKE_RIGHT (2)
                    if step_count % 300 == 1:
                        assigned_lane = 0 if class_counts[1] <= class_counts[2] else 1

                    # A. Maintain ego lane safely
                    if EGO_VEHICLE_ID in active_vehicles:
                        maintain_lane_without_blocking_turns(EGO_VEHICLE_ID, assigned_lane)

                    # B. Keep lead_car in same lane & create realistic slow obstacle waves
                    if "lead_car" in active_vehicles:
                        maintain_lane_without_blocking_turns("lead_car", assigned_lane)

                        if step_count % obstacle_trigger == 0:
                            traci.vehicle.setSpeed("lead_car", random.uniform(1.0, 5.5))
                        elif step_count % obstacle_trigger == (obstacle_trigger // 2):
                            traci.vehicle.setSpeed("lead_car", random.uniform(10.0, 14.0))

                    # C. Dynamic Adjacent Traffic Injection
                    if step_count % 30 == 0 and EGO_VEHICLE_ID in active_vehicles:
                        bg_counter += 1
                        bg_id = f"dyn_ep{episode}_{bg_counter}"
                        adjacent_lane = 1 - assigned_lane
                        try:
                            traci.vehicle.add(
                                vehID=bg_id,
                                routeID="r_loop",
                                typeID="traffic_car",
                                depart="now",
                                departLane=str(adjacent_lane),
                                departPos="free",
                                departSpeed=str(random.uniform(6.0, 13.5))
                            )
                        except traci.TraCIException:
                            pass

                    # D. Telemetry & Natural Subsampling
                    if EGO_VEHICLE_ID in active_vehicles:
                        telemetry = SensorUtils.get_vehicle_telemetry(EGO_VEHICLE_ID)
                        if not telemetry:
                            continue

                        ego_speed = telemetry["speed"]
                        radar = SensorUtils.detect_leader(EGO_VEHICLE_ID, radar_range=100.0, current_speed=ego_speed)
                        lead_speed = radar["leader_speed"]
                        lead_gap = radar["gap"]

                        lateral = SensorUtils.scan_360_surroundings(
                            ego_id=EGO_VEHICLE_ID,
                            active_vehicles=active_vehicles,
                            safety_margin=18.0
                        )
                        if not lateral.get("valid"):
                            continue

                        critical_dist = 22.0 + random.gauss(0, 1.4)
                        safety_margin = 18.0 + random.gauss(0, 1.6)
                        slow_thresh = 7.5

                        is_front_blocked = (lead_gap <= critical_dist) and (lead_speed <= slow_thresh)
                        left_really_safe = lateral["has_left"] and (lateral["left_gap"] >= safety_margin)
                        right_really_safe = lateral["has_right"] and (lateral["right_gap"] >= safety_margin)

                        if not is_front_blocked:
                            decision = 0
                        elif left_really_safe:
                            decision = 1
                        elif right_really_safe:
                            decision = 2
                        else:
                            decision = 0

                        # Keep natural realistic proportions (~45% KEEP, ~27% LEFT, ~27% RIGHT)
                        if decision == 0 and random.random() > 0.35:
                            continue
                        if decision == 1 and (class_counts[1] - class_counts[2]) > 1500:
                            continue
                        if decision == 2 and (class_counts[2] - class_counts[1]) > 1500:
                            continue

                        # Realistic Automotive Sensor Jitter
                        noisy_ego_speed = max(0.0, ego_speed + random.gauss(0, 0.30))
                        noisy_lead_dist = max(0.5, min(100.0, lead_gap + random.gauss(0, 1.20)))
                        noisy_lead_speed = max(0.0, lead_speed + random.gauss(0, 0.50))
                        noisy_left_gap = max(0.0, min(100.0, lateral["left_gap"] + random.gauss(0, 1.50)))
                        noisy_right_gap = max(0.0, min(100.0, lateral["right_gap"] + random.gauss(0, 1.50)))

                        sensed_left_avail = 1 if (lateral["has_left"] and noisy_left_gap >= 18.0) else 0
                        sensed_right_avail = 1 if (lateral["has_right"] and noisy_right_gap >= 18.0) else 0

                        if random.random() < 0.025:
                            sensed_left_avail = 1 - sensed_left_avail
                        if random.random() < 0.025:
                            sensed_right_avail = 1 - sensed_right_avail

                        writer.writerow([
                            round(noisy_ego_speed, 2),
                            round(noisy_lead_dist, 2),
                            round(noisy_lead_speed, 2),
                            sensed_left_avail,
                            sensed_right_avail,
                            round(noisy_left_gap, 2),
                            round(noisy_right_gap, 2),
                            decision
                        ])

                        samples_collected += 1
                        class_counts[decision] += 1

                        if samples_collected % 5000 == 0:
                            print(f"Captured [{samples_collected}/{target_samples}] samples | Class counts: {class_counts}")
            finally:
                traci.close()

    print(f"\nDone! Saved {samples_collected} realistic SUMO samples to:\n{OVERTAKE_DATASET_RAW_PATH}")
    print(f"Final Decision Distribution (0=KEEP, 1=LEFT, 2=RIGHT): {class_counts}")


if __name__ == "__main__":
    run_overtaking_data_generation()