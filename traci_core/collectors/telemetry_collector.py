import csv
import random
from traci_core.config import DATASET_RAW_PATH, MAX_SPEED

TOTAL_SAMPLES = 50000

def calculate_row_data(vehicle_speed: float, leader_distance: float, leader_speed: float, speed_limit: float):
    #1.Wider Radar Sensor Noise (~1.0m distance variance, ~0.6m/s speed variance)
    noisy_distance = max(0.5, leader_distance + random.gauss(0, 1.0))
    noisy_leader_speed = max(0.0, leader_speed + random.gauss(0, 0.6))
    noisy_ego_speed = max(0.0, vehicle_speed + random.gauss(0, 0.4))

    #2.Relative speed aur TTC
    relative_speed = noisy_ego_speed - noisy_leader_speed

    if relative_speed > 0.1:
        time_to_collision = min(99.0, max(0.1, noisy_distance / relative_speed))
    else:
        time_to_collision = 99.0

    #Determine V2V Emergency Flag (1 if critical gap/TTC or random emergency broadcast simulation, else 0)
    #Approximately 15% to 20% samples will feature V2V emergency active scenarios
    is_critical = (noisy_distance <= 8.0 or time_to_collision <= 1.5)
    v2v_emergency_flag = 1 if (is_critical or random.random() < 0.12) else 0

    #3.Base Target Speed Calculation incorporating V2V Sensor Fusion
    min_safe_gap = 6.0
    safe_following_gap = min_safe_gap + (noisy_ego_speed * 1.5)

    if v2v_emergency_flag == 1 or noisy_distance <= min_safe_gap or time_to_collision <= 1.2:
        #Full stop / Emergency braking triggered by V2V or On-board sensors
        target_speed = 0.0

    elif noisy_distance < safe_following_gap:
        #Auto-braking with actuation variance
        ratio = (noisy_distance - min_safe_gap) / max(1.0, (safe_following_gap - min_safe_gap))
        target_speed = (noisy_leader_speed * ratio) + random.gauss(0, 0.8)

    elif noisy_distance < 45.0 and noisy_leader_speed < noisy_ego_speed:
        #Adaptive cruise control
        target_speed = (
            noisy_leader_speed
            + (noisy_ego_speed - noisy_leader_speed) * ((noisy_distance - safe_following_gap) / 45.0)
            + random.gauss(0, 1.0)
        )

    else:
        #Free cruising with dynamic acceleration
        accel_step = random.uniform(0.5, 3.0)
        allowed_max = min(MAX_SPEED, speed_limit)
        target_speed = min(allowed_max, noisy_ego_speed + accel_step) + random.gauss(0, 1.2)

    #4.Physical safety clamping
    target_speed = max(0.0, min(speed_limit, target_speed))

    return [
        round(noisy_ego_speed, 2),
        round(noisy_leader_speed, 2),
        round(noisy_distance, 2),
        round(relative_speed, 2),
        round(time_to_collision, 2),
        round(speed_limit, 2),
        int(v2v_emergency_flag), 
        round(target_speed, 2)  
    ]

def generate_telemetry_dataset(num_samples: int = TOTAL_SAMPLES):
    DATASET_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        "vehicle_speed",
        "leader_speed",
        "leader_distance",
        "relative_speed",
        "time_to_collision",
        "speed_limit",
        "v2v_emergency_flag", 
        "target_speed"
    ]

    with open(DATASET_RAW_PATH, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for _ in range(num_samples):
            #Speed limits: 30 km/h (8.33), 50 km/h (13.89), 80 km/h (22.22)
            speed_limit = random.choice([8.33, 13.89, 22.22])
            vehicle_speed = random.uniform(0.0, speed_limit)
            leader_distance = random.uniform(2.0, 80.0)
            leader_speed = random.uniform(0.0, speed_limit)

            row = calculate_row_data(vehicle_speed, leader_distance, leader_speed, speed_limit)
            writer.writerow(row)

    print(f"Successfully generated {num_samples} samples with V2V flag at: {DATASET_RAW_PATH}")

if __name__ == "__main__":
    generate_telemetry_dataset()