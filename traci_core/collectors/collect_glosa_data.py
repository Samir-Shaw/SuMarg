import csv
import sys
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import traci
from traci_core.config import ROOT_DIR, GLOSA_SIM_CONFIG_PATH, STEP_LENGTH, EGO_VEHICLE_ID
from traci_core.collectors.sensor_utils import SensorUtils

GLOSA_DATASET_RAW_PATH = ROOT_DIR / "ml_engine" / "data" / "raw" / "glosa_data.csv"
TOTAL_SAMPLES = 50000


def compute_optimal_glide_speed(
    dist_to_tls: float,
    signal_phase: int,
    time_to_switch: float,
    ego_speed: float,
    queue_vehicles: int,
    speed_limit: float,
) -> float:
    """
    Physics-guided & queue-aware GLOSA reference advisor.
    Includes natural driver/timing micro-variance so model generalizes realistically.
    signal_phase: 0 = RED, 1 = YELLOW, 2 = GREEN
    """
    min_glide = 3.0
    safety_buffer = 1.5 + random.gauss(0, 0.30)
    queue_delay = queue_vehicles * (1.8 + random.gauss(0, 0.20))

    # Case 1: GREEN Signal (signal_phase == 2)
    if signal_phase == 2:
        effective_green = max(0.5, time_to_switch - 0.5)
        net_pass_time = effective_green - (queue_delay * 0.5)

        if net_pass_time > 0.5:
            req_speed = dist_to_tls / net_pass_time
            if req_speed <= ego_speed:
                return min(speed_limit, max(ego_speed, min_glide))
            elif req_speed <= speed_limit:
                return req_speed

        # Dilemma zone: Green expiring soon, start smooth glide for next cycle
        total_wait = time_to_switch + 22.0 + queue_delay + safety_buffer
        glide = dist_to_tls / max(total_wait, 1.0)
        if dist_to_tls < 18.0 and glide < min_glide:
            return 0.0
        return max(min_glide, min(speed_limit, glide))

    # Case 2: YELLOW Signal (signal_phase == 1)
    elif signal_phase == 1:
        if dist_to_tls <= 12.0 and ego_speed >= 8.0 and queue_vehicles == 0:
            return min(ego_speed, speed_limit)

        total_wait = time_to_switch + 22.0 + queue_delay + safety_buffer
        glide = dist_to_tls / max(total_wait, 1.0)
        if dist_to_tls < 18.0 and glide < min_glide:
            return 0.0
        return max(min_glide, min(speed_limit, glide))

    # Case 3: RED Signal (signal_phase == 0)
    else:
        effective_wait = time_to_switch + queue_delay + safety_buffer
        ideal_glide = dist_to_tls / max(effective_wait, 1.0)

        if dist_to_tls < 18.0 and ideal_glide < min_glide:
            return 0.0
        if ideal_glide >= speed_limit:
            return speed_limit
        return max(min_glide, min(speed_limit, ideal_glide))


def run_glosa_data_generation(target_samples: int = TOTAL_SAMPLES):
    GLOSA_DATASET_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "dist_to_tls",
        "signal_phase",
        "time_to_switch",
        "ego_speed",
        "queue_vehicles",
        "speed_limit",
        "optimal_glide_speed",
    ]

    sumo_cmd = [
        "sumo",
        "-c", str(GLOSA_SIM_CONFIG_PATH),
        "--step-length", str(STEP_LENGTH),
        "--no-step-log", "true",
        "--collision.action", "none",
    ]

    samples_collected = 0
    phase_counts = {0: 0, 1: 0, 2: 0}
    episode = 0

    print(f"Starting Balanced Headless SUMO GLOSA Collection (Target: {target_samples} samples)...")

    with open(GLOSA_DATASET_RAW_PATH, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        while samples_collected < target_samples:
            episode += 1
            traci.start(sumo_cmd)
            step_count = 0
            bg_counter = 0

            # Remove static lead_car so ego_car gets clean, unobstructed signal approaches
            try:
                if "lead_car" in traci.vehicle.getIDList():
                    traci.vehicle.remove("lead_car")
            except traci.TraCIException:
                pass

            try:
                while traci.simulation.getMinExpectedNumber() > 0 and samples_collected < target_samples:
                    traci.simulationStep()
                    step_count += 1
                    active_vehicles = traci.vehicle.getIDList()

                    if step_count > 20 and EGO_VEHICLE_ID not in active_vehicles:
                        break

                    # A. Periodically vary signal phase & duration for full SPaT coverage
                    if step_count % 120 == 0:
                        try:
                            tls_ids = traci.trafficlight.getIDList()
                            if tls_ids:
                                random_phase = random.choice([0, 1, 2, 3])
                                traci.trafficlight.setPhase(tls_ids[0], random_phase)
                                traci.trafficlight.setPhaseDuration(tls_ids[0], random.uniform(3.0, 25.0))
                        except traci.TraCIException:
                            pass

                    # B. Inject occasional traffic on r_loop so queue_vehicles varies naturally
                    if step_count % 45 == 0 and EGO_VEHICLE_ID in active_vehicles:
                        bg_counter += 1
                        bg_id = f"glosa_q_ep{episode}_{bg_counter}"
                        try:
                            traci.vehicle.add(
                                vehID=bg_id,
                                routeID="r_loop",
                                typeID="traffic_car",
                                depart="now",
                                departLane="random",
                                departPos="free",
                                departSpeed=str(random.uniform(5.0, 13.0)),
                            )
                        except traci.TraCIException:
                            pass

                    # C. Vary ego approach speed across steps so model sees diverse speeds
                    if EGO_VEHICLE_ID in active_vehicles and step_count % 50 == 1:
                        traci.vehicle.setSpeed(EGO_VEHICLE_ID, random.uniform(3.5, 15.5))

                    # D. Read V2I + Ego Telemetry via SensorUtils
                    if EGO_VEHICLE_ID in active_vehicles:
                        features = SensorUtils.get_glosa_features_for_ml(EGO_VEHICLE_ID, max_range=200.0)
                        if not features:
                            continue

                        phase = int(features["signal_phase"])

                        # Keep balanced distribution (~45% RED, ~10% YELLOW, ~45% GREEN)
                        if phase == 0 and phase_counts[0] > (target_samples * 0.45):
                            continue
                        if phase == 2 and phase_counts[2] > (target_samples * 0.45):
                            continue

                        target_glide = compute_optimal_glide_speed(
                            dist_to_tls=features["dist_to_tls"],
                            signal_phase=phase,
                            time_to_switch=features["time_to_switch"],
                            ego_speed=features["ego_speed"],
                            queue_vehicles=int(features["queue_vehicles"]),
                            speed_limit=features["speed_limit"],
                        )

                        # Realistic V2I / GPS / Radar Sensor Jitter (prevents 99.9% overfitting)
                        noisy_dist = max(1.0, min(200.0, features["dist_to_tls"] + random.gauss(0, 1.50)))
                        noisy_tts = max(0.1, features["time_to_switch"] + random.gauss(0, 0.30))
                        noisy_ego_speed = max(0.0, features["ego_speed"] + random.gauss(0, 0.35))
                        noisy_queue = max(0, int(round(features["queue_vehicles"] + random.gauss(0, 0.40))))
                        noisy_target = max(0.0, min(features["speed_limit"], target_glide + random.gauss(0, 0.25)))

                        writer.writerow([
                            round(noisy_dist, 2),
                            phase,
                            round(noisy_tts, 2),
                            round(noisy_ego_speed, 2),
                            noisy_queue,
                            round(features["speed_limit"], 2),
                            round(noisy_target, 2),
                        ])

                        samples_collected += 1
                        phase_counts[phase] += 1

                        if samples_collected % 5000 == 0:
                            print(f"Captured [{samples_collected}/{target_samples}] samples | Phase counts: {phase_counts}")
            finally:
                traci.close()

    print(f"\nDone! Saved {samples_collected} realistic SUMO GLOSA samples to:\n{GLOSA_DATASET_RAW_PATH}")
    print(f"Final Phase Distribution (0=RED, 1=YELLOW, 2=GREEN): {phase_counts}")


if __name__ == "__main__":
    run_glosa_data_generation()