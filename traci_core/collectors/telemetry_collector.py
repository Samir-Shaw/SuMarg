import csv
import random
import sys
from pathlib import Path

# Root directory path resolve karna
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import traci
from traci_core.config import SIM_CONFIG_PATH, DATASET_RAW_PATH, MAX_SPEED


def calculate_row_data(vehicle_speed: float, leader_distance: float, leader_speed: float, speed_limit: float):
    # 1. Wider Radar Sensor Noise (~1.0m distance variance, ~0.6m/s speed variance)
    noisy_distance = max(0.5, leader_distance + random.gauss(0, 1.0))
    noisy_leader_speed = max(0.0, leader_speed + random.gauss(0, 0.6))
    noisy_ego_speed = max(0.0, vehicle_speed + random.gauss(0, 0.4))

    # 2. Relative speed aur TTC
    relative_speed = noisy_ego_speed - noisy_leader_speed

    if relative_speed > 0.1:
        time_to_collision = min(99.0, max(0.1, noisy_distance / relative_speed))
    else:
        time_to_collision = 99.0

    # 3. Base Target Speed Calculation
    min_safe_gap = 6.0
    safe_following_gap = min_safe_gap + (noisy_ego_speed * 1.5)

    if noisy_distance <= min_safe_gap or time_to_collision <= 1.2:
        # Full stop / Emergency braking remains strict
        target_speed = 0.0

    elif noisy_distance < safe_following_gap:
        # Auto-braking with actuation variance
        ratio = (noisy_distance - min_safe_gap) / max(1.0, (safe_following_gap - min_safe_gap))
        target_speed = (noisy_leader_speed * ratio) + random.gauss(0, 0.8)

    elif noisy_distance < 45.0 and noisy_leader_speed < noisy_ego_speed:
        # Adaptive cruise with driver delay
        target_speed = (
            noisy_leader_speed
            + (noisy_ego_speed - noisy_leader_speed) * ((noisy_distance - safe_following_gap) / 45.0)
            + random.gauss(0, 1.0)
        )

    else:
        # Cruising with dynamic throttle acceleration
        accel_step = random.uniform(0.5, 3.0)
        allowed_max = min(MAX_SPEED, speed_limit)
        target_speed = min(allowed_max, noisy_ego_speed + accel_step) + random.gauss(0, 1.2)

    # 4. Physical safety clamping
    target_speed = max(0.0, min(speed_limit, target_speed))

    return [
        round(noisy_ego_speed, 2),
        round(noisy_leader_speed, 2),
        round(noisy_distance, 2),
        round(relative_speed, 2),
        round(time_to_collision, 2),
        round(speed_limit, 2),
        round(target_speed, 2)
    ]   
def collect_braking_data(max_steps=1500):
    DATASET_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []

    # =========================================================================
    # PART A: Live SUMO Simulation Telemetry Collection
    # =========================================================================
    if SIM_CONFIG_PATH.exists():
        print(f"🚀 Step 1: SUMO Simulation start ho raha hai ({SIM_CONFIG_PATH.name})...")
        try:
            traci.start(["sumo", "-c", str(SIM_CONFIG_PATH), "--no-step-log", "true", "--no-warnings", "true"])

            step = 0
            while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
                traci.simulationStep()
                step += 1

                for veh_id in traci.vehicle.getIDList():
                    vehicle_speed = traci.vehicle.getSpeed(veh_id)
                    lane_id = traci.vehicle.getLaneID(veh_id)

                    if lane_id:
                        try:
                            speed_limit = traci.lane.getMaxSpeed(lane_id)
                        except traci.TraCIException:
                            speed_limit = MAX_SPEED
                    else:
                        speed_limit = MAX_SPEED

                    leader_info = traci.vehicle.getLeader(veh_id, 100.0)
                    if leader_info is not None:
                        leader_id, leader_distance = leader_info
                        leader_speed = traci.vehicle.getSpeed(leader_id)
                    else:
                        leader_distance = 100.0
                        leader_speed = speed_limit

                    row = calculate_row_data(vehicle_speed, leader_distance, leader_speed, speed_limit)
                    all_rows.append(row)

            traci.close()
            print(f"✅ SUMO se {len(all_rows)} real traffic rows collect ho gayi!")

        except Exception as e:
            print(f"⚠️ Note: SUMO run skip hua ya XML issue ({e}).")
            print("👉 Physics-based synthetic generator fallback use ho raha hai...")

    # =========================================================================
    # PART B: Physics Scenarios & Edge Cases
    # =========================================================================
    print("🧪 Step 2: 5,000 Edge-case scenarios generate ho rahe hain...")

    for _ in range(5000):
        speed_limit = random.choice([13.89, 16.67, 22.22, 25.0])
        vehicle_speed = random.uniform(0.0, speed_limit)
        scenario = random.choice(["emergency_stop", "auto_brake", "follow_leader", "free_road"])

        if scenario == "emergency_stop":
            leader_distance = random.uniform(1.0, 8.0)
            leader_speed = random.uniform(0.0, 3.0)

        elif scenario == "auto_brake":
            leader_distance = random.uniform(8.0, 25.0)
            leader_speed = random.uniform(2.0, 12.0)

        elif scenario == "follow_leader":
            leader_distance = random.uniform(20.0, 50.0)
            leader_speed = random.uniform(8.0, 18.0)

        else:
            leader_distance = random.uniform(50.0, 100.0)
            leader_speed = random.uniform(18.0, speed_limit)

        row = calculate_row_data(vehicle_speed, leader_distance, leader_speed, speed_limit)
        all_rows.append(row)

    # =========================================================================
    # PART C: Save Dataset to ml_engine/data/raw/braking_data.csv
    # =========================================================================
    headers = [
        "vehicle_speed",
        "leader_speed",
        "leader_distance",
        "relative_speed",
        "time_to_collision",
        "speed_limit",
        "target_speed"
    ]

    with open(DATASET_RAW_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_rows)

    print("\n" + "=" * 65)
    print(f"🎉 Dataset Successfully Updated With Realistic Physics Noise!")
    print(f"📊 Total Rows Saved : {len(all_rows)}")
    print(f"📂 Saved Location   : {DATASET_RAW_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    collect_braking_data()