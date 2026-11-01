# Extract vehicle_speed, leader_distance, TTC, speed_limit
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Root directory path resolve karna
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import traci


class SensorUtils:
    """
    Virtual Radar aur Telemetry Sensor utilities jo SUMO car ke
    real-time physical dynamics aur environmental context ko read karta hai.
    """

    @staticmethod
    def get_vehicle_telemetry(veh_id: str) -> Optional[Dict[str, float]]:
        """
        Ego vehicle ki current kinematic state fetch karta hai.
        """
        try:
            speed = traci.vehicle.getSpeed(veh_id)
            acceleration = traci.vehicle.getAcceleration(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id)
            speed_limit = traci.lane.getMaxSpeed(lane_id) if lane_id else 22.22

            return {
                "speed": float(speed),
                "acceleration": float(acceleration),
                "speed_limit": float(speed_limit)
            }
        except traci.TraCIException:
            return None

    @staticmethod
    def detect_leader(veh_id: str, radar_range: float = 100.0) -> Dict[str, Any]:
        """
        Radar simulation: Aage chalne wali car (Leader) ko detect karta hai,
        uska gap, relative speed aur Time-To-Collision (TTC) calculate karta hai.
        """
        telemetry = SensorUtils.get_vehicle_telemetry(veh_id)
        if not telemetry:
            return {
                "has_leader": False,
                "leader_id": None,
                "gap": radar_range,
                "leader_speed": 0.0,
                "relative_speed": 0.0,
                "ttc": 99.0
            }

        ego_speed = telemetry["speed"]

        try:
            # traci.vehicle.getLeader returns: (leader_id, distance) ya None
            leader_info = traci.vehicle.getLeader(veh_id, radar_range)

            if leader_info is not None:
                leader_id, gap = leader_info
                leader_speed = traci.vehicle.getSpeed(leader_id)
                relative_speed = ego_speed - leader_speed

                # Time-to-Collision (TTC) calculation in seconds
                if relative_speed > 0.1:
                    ttc = min(99.0, max(0.1, gap / relative_speed))
                else:
                    ttc = 99.0  # Leader fast ya stationary relative pace par hai

                return {
                    "has_leader": True,
                    "leader_id": leader_id,
                    "gap": round(float(gap), 2),
                    "leader_speed": round(float(leader_speed), 2),
                    "relative_speed": round(float(relative_speed), 2),
                    "ttc": round(float(ttc), 2)
                }

        except traci.TraCIException:
            pass

        # Open Road Fallback (Jab aage koi gaadi nahi ho)
        return {
            "has_leader": False,
            "leader_id": None,
            "gap": float(radar_range),
            "leader_speed": telemetry["speed_limit"],
            "relative_speed": 0.0,
            "ttc": 99.0
        }

    @staticmethod
    def get_features_for_ml(veh_id: str, radar_range: float = 100.0) -> Optional[list]:
        """
        ML model ke exact 6 input feature vector banata hai:
        [vehicle_speed, leader_speed, leader_distance, relative_speed, time_to_collision, speed_limit]
        """
        telemetry = SensorUtils.get_vehicle_telemetry(veh_id)
        if not telemetry:
            return None

        radar = SensorUtils.detect_leader(veh_id, radar_range=radar_range)

        return [
            round(telemetry["speed"], 2),
            round(radar["leader_speed"], 2),
            round(radar["gap"], 2),
            round(radar["relative_speed"], 2),
            round(radar["ttc"], 2),
            round(telemetry["speed_limit"], 2)
        ]