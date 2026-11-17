import traci
from ml_engine.src.logger import logging


class OvertakingController:
    COLOR_OVERTAKING = (0, 220, 255, 255)  
    LANE_CHANGE_DURATION = 2.5            
    COOLDOWN_STEPS = 60                  
    _last_change_step: dict = {}

    @staticmethod
    def initialize_vehicle(ego_id: str) -> None:
        try:
            if ego_id in traci.vehicle.getIDList():
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
        try:
            if ego_id not in traci.vehicle.getIDList() or decision == 0:
                return False

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