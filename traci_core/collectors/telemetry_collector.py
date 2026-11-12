import csv
import sys
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import traci
from traci_core.config import ROOT_DIR, SIM_CONFIG_PATH, STEP_LENGTH, EGO_VEHICLE_ID
from traci_core.collectors.sensor_utils import SensorUtils

OVERTAKE_DATASET_RAW_PATH = ROOT_DIR / "ml_engine" / "data" / "raw" / "overtaking_data.csv"
TOTAL_SAMPLES = 50000


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

    print(f"Starting Headless SUMO to capture {target_samples} calibrated overtaking telemetry samples...")

    with open(OVERTAKE_DATASET_RAW_PATH, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        while samples_collected < target_samples:
            episode += 1
            traci.start(sumo_cmd)

            step_count = 0
            bg_counter = 0
            obstacle_trigger = random.randint(40, 80)

            while traci.simulation.getMinExpectedNumber() > 0 and samples_collected < target_samples:
                traci.simulationStep()
                step_count += 1
                active_vehicles = traci.vehicle.getIDList()

                if step_count > 15 and EGO_VEHICLE_ID not in active_vehicles:
                    break

                # A. Dynamic Obstacle Waves on Lead Car
                if "lead_car" in active_vehicles:
                    if step_count % obstacle_trigger == 0:
                        traci.vehicle.setSpeed("lead_car", random.uniform(1.5, 6.0))
                    elif step_count % obstacle_trigger == (obstacle_trigger // 2):
                        traci.vehicle.setSpeed("lead_car", random.uniform(11.0, 14.0))

                # B. Ego Lane Exploration (Captures both OVERTAKE_LEFT & OVERTAKE_RIGHT scenarios)
                if EGO_VEHICLE_ID in active_vehicles and step_count % 120 == 0:
                    try:
                        traci.vehicle.changeLane(EGO_VEHICLE_ID, random.choice([0, 1]), 2.0)
                    except traci.TraCIException:
                        pass

                # C. Dynamic Adjacent Traffic Injection
                if step_count % 25 == 0 and EGO_VEHICLE_ID in active_vehicles:
                    bg_counter += 1
                    bg_id = f"dyn_ep{episode}_{bg_counter}"
                    try:
                        traci.vehicle.add(
                            vehID=bg_id,
                            routeID="r_loop",
                            typeID="traffic_car",
                            depart="now",
                            departLane=random.choice(["0", "1"]),
                            departPos="free",
                            departSpeed=str(random.uniform(6.5, 13.5))
                        )
                    except traci.TraCIException:
                        pass

                # D. Ego Vehicle Telemetry & Labeling via SensorUtils
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

                    # 1. Physics Ground Truth Decision with Driver Hesitation Boundary
                    critical_dist = 22.0 + random.gauss(0, 1.4)
                    safety_margin = 18.0 + random.gauss(0, 1.6)
                    slow_thresh = 7.5

                    is_front_blocked = (lead_gap <= critical_dist) and (lead_speed <= slow_thresh)
                    left_really_safe = lateral["has_left"] and (lateral["left_gap"] >= safety_margin)
                    right_really_safe = lateral["has_right"] and (lateral["right_gap"] >= safety_margin)

                    if not is_front_blocked:
                        decision = 0  # KEEP_LANE
                    elif left_really_safe:
                        decision = 1  # OVERTAKE_LEFT
                    elif right_really_safe:
                        decision = 2  # OVERTAKE_RIGHT
                    else:
                        decision = 0  # KEEP_LANE (Blocked)

                    # 2. Realistic Automotive Sensor Jitter (Prevents 99.9% Overfitting)
                    noisy_ego_speed = max(0.0, ego_speed + random.gauss(0, 0.30))
                    noisy_lead_dist = max(0.5, min(100.0, lead_gap + random.gauss(0, 1.20)))
                    noisy_lead_speed = max(0.0, lead_speed + random.gauss(0, 0.50))
                    noisy_left_gap = max(0.0, min(100.0, lateral["left_gap"] + random.gauss(0, 1.50)))
                    noisy_right_gap = max(0.0, min(100.0, lateral["right_gap"] + random.gauss(0, 1.50)))

                    sensed_left_avail = 1 if (lateral["has_left"] and noisy_left_gap >= 18.0) else 0
                    sensed_right_avail = 1 if (lateral["has_right"] and noisy_right_gap >= 18.0) else 0

                    # Occasional radar occlusion flicker (2.5%)
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

            traci.close()

    print(f"\nDone! Successfully collected {samples_collected} calibrated SUMO samples at:\n{OVERTAKE_DATASET_RAW_PATH}")
    print(f"Final Decision Distribution (0=KEEP, 1=LEFT, 2=RIGHT): {class_counts}")


if __name__ == "__main__":
    run_overtaking_data_generation()