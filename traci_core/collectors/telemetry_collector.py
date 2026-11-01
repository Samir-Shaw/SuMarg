# 📄 File: ml_engine/training/collect_data.py
import csv
import random
import sys
from pathlib import Path

# -------------------------------------------------------------------------
# STEP 1: Root folder ('sumarg_v1') ko Python path mein add karna
# Taaki 'from backend.config import ...' karne par ModuleNotFoundError na aaye
# -------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import traci
from backend.config import SIM_CONFIG_PATH, DATASET_CSV_PATH, MAX_SPEED


def calculate_row_data(vehicle_speed: float, leader_distance: float, leader_speed: float, speed_limit: float):
    """
    Yeh function 4 raw sensor values leta hai aur baki ke features (relative_speed, 
    time_to_collision) + ML ka target answer (target_speed) calculate karta hai.
    """
    # 1. Relative Speed: Hum samne wali gadi ke kitni tezi se kareeb ja rahe hain?
    # Agar +ve hai matlab hum tez hain; agar -ve hai matlab samne wala humse tez bhag raha hai
    relative_speed = vehicle_speed - leader_speed

    # 2. Time-To-Collision (TTC): Agar isi speed par chale toh kitne second mein takkar hogi?
    if relative_speed > 0.1:
        time_to_collision = min(99.0, max(0.1, leader_distance / relative_speed))
    else:
        # Agar relative_speed <= 0 hai, toh kabhi takkar nahi hogi (99.0 matlab 100% Safe)
        time_to_collision = 99.0

    # 3. Target Speed (Label Y): Agle 0.1s step par gadi ki safe speed kya honi chahiye?
    min_safe_gap = 6.0  # Kam se kam 6 meter ki doori zaroori hai
    safe_following_gap = min_safe_gap + (vehicle_speed * 1.5)  # 1.5 second rule ke hisaab se safe gap

    if leader_distance <= min_safe_gap or time_to_collision <= 1.2:
        # 🚨 DANGER: Gadi bilkul samne hai -> Turant Full Brake (Speed = 0)
        target_speed = 0.0

    elif leader_distance < safe_following_gap:
        # ⚠️ AUTO-BRAKE: Safe gap se andar aa gaye -> Smoothly speed kam karo
        ratio = (leader_distance - min_safe_gap) / max(1.0, (safe_following_gap - min_safe_gap))
        target_speed = leader_speed * ratio

    elif leader_distance < 45.0 and leader_speed < vehicle_speed:
        # 🚗 ADAPTIVE CRUISE: Aage slow gadi hai -> Dheere-dheere uski speed match karo
        target_speed = leader_speed + (vehicle_speed - leader_speed) * ((leader_distance - safe_following_gap) / 45.0)

    else:
        # 🛣️ OPEN ROAD: Samne koi khatra nahi -> Speed limit tak smoothly accelerate karo
        allowed_max = min(MAX_SPEED, speed_limit)
        target_speed = min(allowed_max, vehicle_speed + 2.0)

    # Speed ko 0 aur speed_limit ke beech lock kar do (Negative speed nahi ho sakti)
    target_speed = max(0.0, min(speed_limit, target_speed))

    # Saare 7 columns (6 Features + 1 Target) return kar do
    return [
        round(vehicle_speed, 2),
        round(leader_speed, 2),
        round(leader_distance, 2),
        round(relative_speed, 2),
        round(time_to_collision, 2),
        round(speed_limit, 2),
        round(target_speed, 2)
    ]


def collect_braking_data(max_steps=1500):
    # Ensure karo ki 'ml_engine/datasets/' folder bana hua hai
    DATASET_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []

    # =========================================================================
    # PART A: Live SUMO Simulation Chala Kar Real Gadiyon Se Data Nikalna
    # =========================================================================
    if SIM_CONFIG_PATH.exists():
        print(f"🚀 Step 1: SUMO Simulation start ho raha hai ({SIM_CONFIG_PATH.name})...")
        try:
            # 'sumo' (headless) use kiya hai taaki bina GUI khule 3 second mein data nikal aaye
            traci.start(["sumo", "-c", str(SIM_CONFIG_PATH), "--no-step-log", "true", "--no-warnings", "true"])
            
            step = 0
            while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
                traci.simulationStep()
                step += 1

                # Road par jitni bhi gadiyan hain, un sabke sensors read karo
                for veh_id in traci.vehicle.getIDList():
                    vehicle_speed = traci.vehicle.getSpeed(veh_id)
                    lane_id = traci.vehicle.getLaneID(veh_id)
                    speed_limit = traci.lane.getMaxSpeed(lane_id)

                    # Radar Sensor: 100 meter tak samne wali gadi (Leader) ko dhoondho
                    leader_info = traci.vehicle.getLeader(veh_id, 100.0)
                    if leader_info is not None:
                        leader_id, leader_distance = leader_info
                        leader_speed = traci.vehicle.getSpeed(leader_id)
                    else:
                        # Agar 100m tak samne koi nahi hai
                        leader_distance = 100.0
                        leader_speed = speed_limit

                    row = calculate_row_data(vehicle_speed, leader_distance, leader_speed, speed_limit)
                    all_rows.append(row)

            traci.close()
            print(f"✅ SUMO se {len(all_rows)} real traffic rows collect ho gayin!")

        except Exception as e:
            print(f"⚠️ Note: SUMO XML abhi set nahi hai ya skip hua ({e}).")
            print("👉 Chinta mat karo! Hum Part-B se full physics dataset bana rahe hain...")

    # =========================================================================
    # PART B: Emergency Braking & Edge-Cases Add Karna (Bahut Zaroori!)
    # Kyunki normal SUMO mein gadiyan kabhi 2m-5m tak kareeb nahi aati, 
    # isliye ML ko Emergency Brake sikhane ke liye hum 5000 test cases jodenge.
    # =========================================================================
    print("🧪 Step 2: 5,000 Emergency Brake, Slow Leader aur Open Road scenarios add ho rahe hain...")
    
    for _ in range(5000):
        # Alag-alag sadkon ki speed limit (e.g., 13.89 m/s = 50 km/h, 22.22 m/s = 80 km/h)
        speed_limit = random.choice([13.89, 16.67, 22.22, 25.0])
        vehicle_speed = random.uniform(0.0, speed_limit)

        scenario = random.choice(["emergency_stop", "auto_brake", "follow_leader", "free_road"])

        if scenario == "emergency_stop":
            # Samne wali gadi bilkul paas (1m se 8m) hai aur ruki hui hai!
            leader_distance = random.uniform(1.0, 8.0)
            leader_speed = random.uniform(0.0, 3.0)

        elif scenario == "auto_brake":
            # Samne wali gadi 8m se 25m door hai aur slow chal rahi hai
            leader_distance = random.uniform(8.0, 25.0)
            leader_speed = random.uniform(2.0, 12.0)

        elif scenario == "follow_leader":
            # Samne wali gadi medium distance (20m se 50m) par hai
            leader_distance = random.uniform(20.0, 50.0)
            leader_speed = random.uniform(8.0, 18.0)

        else:
            # Khali sadak (50m se 100m tak koi khatra nahi)
            leader_distance = random.uniform(50.0, 100.0)
            leader_speed = random.uniform(18.0, speed_limit)

        row = calculate_row_data(vehicle_speed, leader_distance, leader_speed, speed_limit)
        all_rows.append(row)

    # =========================================================================
    # PART C: Saara Data CSV File Mein Save Karna
    # =========================================================================
    headers = [
        "vehicle_speed",
        "leader_speed",
        "leader_distance",
        "relative_speed",
        "time_to_collision",
        "speed_limit",
        "target_speed"      # <-- Yeh humara Target (Y) hai jo ML predict karega!
    ]

    with open(DATASET_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_rows)

    print("\n" + "="*65)
    print(f"🎉 BOOM! Dataset 100% Ready!")
    print(f"📊 Total Rows Saved : {len(all_rows)}")
    print(f"📂 File Location    : {DATASET_CSV_PATH}")
    print("="*65)


if __name__ == "__main__":
    collect_braking_data()