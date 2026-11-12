import traci
from ml_engine.src.logger import logging


class OvertakingController:
    COLOR_OVERTAKING = (0, 220, 255, 255)  # Cyan color jab gaadi overtake kare
    LANE_CHANGE_DURATION = 2.5             # Smooth lane shift ke liye seconds
    COOLDOWN_STEPS = 25                    # Agle lane change se pehle minimum wait steps

    _last_change_step: dict = {}

    @staticmethod
    def initialize_vehicle(ego_id: str) -> None:
        try:
            if ego_id in traci.vehicle.getIDList():
                # LaneChangeMode 0 = SUMO khud lane nahi badlega, sirf ML model ke order par badlega
                traci.vehicle.setLaneChangeMode(ego_id, 0)
                OvertakingController._last_change_step[ego_id] = -OvertakingController.COOLDOWN_STEPS
        except traci.TraCIException:
            pass

    @staticmethod
    def apply_decision(
        ego_id: str,
        decision: int,
        current_lane: int,
        total_lanes: int,
        current_step: int
    ) -> bool:
        """
        OvertakingPredictPipeline ke decision ko SUMO me execute karta hai:
        0 -> KEEP_LANE
        1 -> OVERTAKE_LEFT  (current_lane + 1)
        2 -> OVERTAKE_RIGHT (current_lane - 1)
        """
        try:
            if ego_id not in traci.vehicle.getIDList() or decision == 0:
                return False

            # Cooldown check: Taaki gaadi baar-baar left-right zigzag na kare
            last_step = OvertakingController._last_change_step.get(ego_id, -OvertakingController.COOLDOWN_STEPS)
            if (current_step - last_step) < OvertakingController.COOLDOWN_STEPS:
                return False

            target_lane = None
            direction_label = ""

            if decision == 1 and (current_lane + 1) < total_lanes:
                target_lane = current_lane + 1
                direction_label = "LEFT"
            elif decision == 2 and (current_lane - 1) >= 0:
                target_lane = current_lane - 1
                direction_label = "RIGHT"

            if target_lane is not None:
                traci.vehicle.setLaneChangeMode(ego_id, 0)
                traci.vehicle.changeLane(ego_id, target_lane, OvertakingController.LANE_CHANGE_DURATION)
                traci.vehicle.setColor(ego_id, OvertakingController.COLOR_OVERTAKING)
                OvertakingController._last_change_step[ego_id] = current_step
                logging.info(
                    f"Step {current_step}: ML SMART OVERTAKE -> {direction_label} "
                    f"(Lane {current_lane} -> {target_lane})"
                )
                return True

            return False

        except traci.TraCIException as err:
            logging.warning(f"OvertakingController TraCI warning at step {current_step}: {err}")
            return False