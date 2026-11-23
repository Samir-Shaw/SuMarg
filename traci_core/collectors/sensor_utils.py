import sys
from pathlib import Path
from typing import Dict, Any, Optional
import traci
from traci_core.config import ROOT_DIR, MAX_SPEED

class SensorUtils:
  
    @staticmethod
    def get_vehicle_telemetry(veh_id: str) -> Optional[Dict[str, float]]:

        try:
            speed = traci.vehicle.getSpeed(veh_id)
            acceleration = traci.vehicle.getAcceleration(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id)
            speed_limit = traci.lane.getMaxSpeed(lane_id) if lane_id else MAX_SPEED

            return {
                "speed": float(speed),
                "acceleration": float(acceleration),
                "speed_limit": float(speed_limit)
            }
        except traci.TraCIException:
            return None

    @staticmethod
    def detect_leader(veh_id: str, radar_range: float = 100.0, current_speed: Optional[float] = None) -> Dict[str, Any]:
        
        try:
            if current_speed is None:
                current_speed = traci.vehicle.getSpeed(veh_id)

            lane_id = traci.vehicle.getLaneID(veh_id)
            speed_limit = traci.lane.getMaxSpeed(lane_id) if lane_id else MAX_SPEED

            leader_info = traci.vehicle.getLeader(veh_id, radar_range)

            if leader_info is not None:
                leader_id, gap = leader_info
                leader_speed = traci.vehicle.getSpeed(leader_id)
                relative_speed = current_speed - leader_speed

                if relative_speed > 0.1:
                    ttc = min(99.0, max(0.1, gap / relative_speed))
                else:
                    ttc = 99.0  #Safe

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

        return {
            "has_leader": False,
            "leader_id": None,
            "gap": float(radar_range),
            "leader_speed": float(speed_limit),
            "relative_speed": 0.0,
            "ttc": 99.0
        }
    @staticmethod
    def scan_360_surroundings(ego_id: str, active_vehicles: list, safety_margin: float = 18.0) -> dict:
        """
        Scans left and right adjacent lanes for blind-spot clearance and gap estimation.
        Dynamically supports 2-lane and future 3+ lane networks.
        """
        ego_edge = traci.vehicle.getRoadID(ego_id)
        if not ego_edge or ego_edge.startswith(":"):
            return {"valid": False}

        ego_lane = traci.vehicle.getLaneIndex(ego_id)
        ego_pos = traci.vehicle.getLanePosition(ego_id)
        total_lanes = traci.edge.getLaneNumber(ego_edge)

        has_left = (ego_lane + 1) < total_lanes
        has_right = (ego_lane - 1) >= 0

        min_left_gap = 100.0
        min_right_gap = 100.0

        for veh_id in active_vehicles:
            if veh_id == ego_id:
                continue
            if traci.vehicle.getRoadID(veh_id) == ego_edge:
                v_lane = traci.vehicle.getLaneIndex(veh_id)
                v_pos = traci.vehicle.getLanePosition(veh_id)
                gap = abs(v_pos - ego_pos)

                if has_left and v_lane == (ego_lane + 1):
                    min_left_gap = min(min_left_gap, gap)

                if has_right and v_lane == (ego_lane - 1):
                    min_right_gap = min(min_right_gap, gap)

        return {
            "valid": True,
            "current_lane": ego_lane,
            "total_lanes": total_lanes,
            "has_left": has_left,
            "has_right": has_right,
            "left_gap": min_left_gap,
            "right_gap": min_right_gap,
            "left_safe": has_left and (min_left_gap >= safety_margin),
            "right_safe": has_right and (min_right_gap >= safety_margin)
        }
        
    @staticmethod
    def get_features_for_ml(veh_id: str, radar_range: float = 100.0) -> Optional[Dict[str, float]]:

        telemetry = SensorUtils.get_vehicle_telemetry(veh_id)
        if not telemetry:
            return None

        radar = SensorUtils.detect_leader(veh_id, radar_range=radar_range, current_speed=telemetry["speed"])

        return {
            "vehicle_speed": round(telemetry["speed"], 2),
            "leader_speed": round(radar["leader_speed"], 2),
            "leader_distance": round(radar["gap"], 2),
            "relative_speed": round(radar["relative_speed"], 2),
            "time_to_collision": round(radar["ttc"], 2),
            "speed_limit": round(telemetry["speed_limit"], 2)
        }
    @staticmethod
    def get_overtaking_features_for_ml(
        veh_id: str,
        active_vehicles: list,
        radar_range: float = 100.0,
        safety_margin: float = 18.0
    ) -> Optional[Dict[str, Any]]:

        telemetry = SensorUtils.get_vehicle_telemetry(veh_id)
        if not telemetry:
            return None

        radar = SensorUtils.detect_leader(veh_id, radar_range=radar_range, current_speed=telemetry["speed"])
        surroundings = SensorUtils.scan_360_surroundings(
            veh_id, active_vehicles=active_vehicles, safety_margin=safety_margin
        )

        if not surroundings.get("valid", False):
            return None

        return {
            "ego_speed": round(telemetry["speed"], 2),
            "lead_distance": round(radar["gap"], 2),
            "lead_speed": round(radar["leader_speed"], 2),
            "left_lane_available": 1 if surroundings["has_left"] else 0,
            "right_lane_available": 1 if surroundings["has_right"] else 0,
            "left_gap": round(float(surroundings["left_gap"]), 2),
            "right_gap": round(float(surroundings["right_gap"]), 2),
            "current_lane": surroundings["current_lane"],
            "total_lanes": surroundings["total_lanes"]
        }
    @staticmethod
    def scan_upcoming_tls(veh_id: str, max_range: float = 200.0) -> Dict[str, Any]:
        """
        V2I SPaT (Signal Phase and Timing) reader for GLOSA.
        Reads upcoming traffic light distance, phase (0=RED, 1=YELLOW, 2=GREEN),
        remaining time until phase switch, and halting queue ahead on the lane.
        """
        try:
            tls_list = traci.vehicle.getNextTLS(veh_id)
            if not tls_list:
                return {"has_tls": False}

            # tls_list[0] -> (tlsID, tlsLinkIndex, distance, stateChar)
            tls_id, _, dist_to_tls, state_char = tls_list[0]
            dist_to_tls = float(dist_to_tls)

            if dist_to_tls > max_range:
                return {"has_tls": False}

            # Convert SUMO signal state character into numeric phase
            state_lower = state_char.lower()
            if state_lower == "g":
                signal_phase = 2
            elif state_lower == "y":
                signal_phase = 1
            else:
                signal_phase = 0

            sim_time = float(traci.simulation.getTime())
            next_switch = float(traci.trafficlight.getNextSwitch(tls_id))
            time_to_switch = max(0.1, next_switch - sim_time)

            lane_id = traci.vehicle.getLaneID(veh_id)
            queue_vehicles = int(traci.lane.getLastStepHaltingNumber(lane_id)) if lane_id else 0
            speed_limit = float(traci.lane.getMaxSpeed(lane_id)) if lane_id else MAX_SPEED

            return {
                "has_tls": True,
                "tls_id": tls_id,
                "dist_to_tls": round(dist_to_tls, 2),
                "signal_phase": signal_phase,
                "time_to_switch": round(time_to_switch, 2),
                "queue_vehicles": queue_vehicles,
                "speed_limit": round(speed_limit, 2)
            }
        except traci.TraCIException:
            return {"has_tls": False}

    @staticmethod
    def get_glosa_features_for_ml(veh_id: str, max_range: float = 200.0) -> Optional[Dict[str, float]]:
        telemetry = SensorUtils.get_vehicle_telemetry(veh_id)
        if not telemetry:
            return None

        tls = SensorUtils.scan_upcoming_tls(veh_id, max_range=max_range)
        if not tls.get("has_tls", False):
            return None

        return {
            "dist_to_tls": tls["dist_to_tls"],
            "signal_phase": tls["signal_phase"],
            "time_to_switch": tls["time_to_switch"],
            "ego_speed": round(telemetry["speed"], 2),
            "queue_vehicles": tls["queue_vehicles"],
            "speed_limit": tls["speed_limit"]
        }