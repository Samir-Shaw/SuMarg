import traci
from ml_engine.src.logger import logging


class SmartOvertakingScenario:
    # --- BLOCKER CAR TUNING CONTROLS ---
    CRAWL_SPEED = 1              # Ego rukne ke baad blocker ki negligible speed (m/s)
    CRAWL_STEPS_AFTER_STOP = 4   # Kitne steps (10 steps = 1 sec) tak negligible speed par rahegi
    EXIT_TARGET_SPEED = 15.5       # Phir kis speed se aage nikal jayegi (m/s)
    EXIT_RAMP_DURATION = 2.5       # Kitne seconds me smoothly EXIT_TARGET_SPEED pakdegi

    def __init__(
        self,
        ego_id: str = "ego_car",
        lead_id: str = "lead_car",
        blocker_id: str = "blocker_car",
        trigger_step: int = 60,
        clear_left_step: int = 95,
        stop_duration: int = 35,
        resume_speed: float = 7.5
    ):
        self.ego_id = ego_id
        self.lead_id = lead_id
        self.blocker_id = blocker_id
        self.trigger_step = trigger_step
        self.clear_left_step = clear_left_step
        self.stop_duration = stop_duration
        self.resume_speed = resume_speed

        self.initial_setup_done = False
        self.brake_applied = False
        self.ego_stop_step = None
        self.left_cleared = False
        self.resumed = False
        self.blocker_spawned = False

    def _ensure_blocker_car(self, active_vehicles: list):
        if self.blocker_id in active_vehicles or self.blocker_spawned:
            return
        if self.ego_id not in active_vehicles:
            return

        try:
            ego_route = traci.vehicle.getRouteID(self.ego_id)
            ego_lane = traci.vehicle.getLaneIndex(self.ego_id)
            ego_edge = traci.vehicle.getRoadID(self.ego_id)
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            ego_speed = traci.vehicle.getSpeed(self.ego_id)

            if not ego_edge or ego_edge.startswith(":"):
                return

            total_lanes = traci.edge.getLaneNumber(ego_edge)
            target_lane = ego_lane + 1 if (ego_lane + 1) < total_lanes else ego_lane
            spawn_pos = max(5.0, ego_pos + 3.0)

            traci.vehicle.add(
                vehID=self.blocker_id,
                routeID=ego_route,
                departLane=str(target_lane),
                departPos=str(round(spawn_pos, 2)),
                departSpeed=str(round(max(10.0, ego_speed), 2))
            )
            traci.vehicle.setColor(self.blocker_id, (255, 165, 0, 255))
            self.blocker_spawned = True
            logging.info(f"Spawned '{self.blocker_id}' alongside ego_car in lane {target_lane}.")
        except traci.TraCIException as err:
            logging.warning(f"Could not spawn blocker_car: {err}")

    def update(self, current_step: int, active_vehicles: list, v2v_module=None):
        if self.lead_id not in active_vehicles:
            return

        try:
            #Initial Setup
            if not self.initial_setup_done:
                traci.vehicle.setSpeedMode(self.lead_id, 31)
                traci.vehicle.setLaneChangeMode(self.lead_id, 0)
                traci.vehicle.setSpeed(self.lead_id, 13.0)
                traci.vehicle.setColor(self.lead_id, (0, 150, 255, 255))
                self.initial_setup_done = True

            if current_step >= 2 and not self.blocker_spawned:
                self._ensure_blocker_car(active_vehicles)

            lead_pos = traci.vehicle.getPosition(self.lead_id)

            #Phase 1: Lead Car Emergency Brake
            if current_step >= self.trigger_step and not self.brake_applied:
                traci.vehicle.setSpeedMode(self.lead_id, 0)
                traci.vehicle.setSpeed(self.lead_id, 0.0)
                traci.vehicle.setColor(self.lead_id, (255, 0, 0, 255))
                logging.warning(f"Step {current_step}: Lead car EMERGENCY BRAKE!")
                self.brake_applied = True

            #blocker_car controlled sequence:
            if self.blocker_id in active_vehicles and self.ego_id in active_vehicles:
                traci.vehicle.setLaneChangeMode(self.blocker_id, 0)
                ego_speed = traci.vehicle.getSpeed(self.ego_id)

                # Jab ego_car 0 par aaye, us step ko record karein
                if self.brake_applied and ego_speed <= 0.2 and self.ego_stop_step is None:
                    self.ego_stop_step = current_step
                    logging.info(f"Step {current_step}: Ego car reached 0 m/s. Blocker holding crawl speed...")

                if self.ego_stop_step is None:
                    #Phase A: Brake ke dauran ego_car ko copy karke saath slow hogi
                    traci.vehicle.setSpeedMode(self.blocker_id, 0)
                    ego_pos = traci.vehicle.getLanePosition(self.ego_id)
                    blocker_pos = traci.vehicle.getLanePosition(self.blocker_id)

                    pos_diff = (ego_pos + 2.0) - blocker_pos
                    sync_speed = max(0.0, ego_speed + (pos_diff * 0.5))
                    traci.vehicle.setSpeed(self.blocker_id, sync_speed)

                elif (current_step - self.ego_stop_step) < self.CRAWL_STEPS_AFTER_STOP:
                    #Phase B: Ego 0 par aane ke baad controllable negligible speed par crawl karegi
                    traci.vehicle.setSpeedMode(self.blocker_id, 0)
                    traci.vehicle.setSpeed(self.blocker_id, self.CRAWL_SPEED)

                else:
                    #Phase C: Controlled smooth ramp-up ke saath aage nikal jayegi
                    if not self.left_cleared:
                        logging.info(
                            f"Step {current_step}: blocker_car ramping up to {self.EXIT_TARGET_SPEED} m/s "
                            f"over {self.EXIT_RAMP_DURATION}s..."
                        )
                        traci.vehicle.setSpeedMode(self.blocker_id, 31)
                        traci.vehicle.slowDown(
                            self.blocker_id,
                            self.EXIT_TARGET_SPEED,
                            duration=self.EXIT_RAMP_DURATION
                        )
                        self.left_cleared = True

            #Active V2V Broadcast
            if self.brake_applied and not self.resumed and v2v_module:
                v2v_module.broadcast(
                    sender_id=self.lead_id,
                    event_type="EMERGENCY_BRAKE",
                    speed=0.0,
                    decel=9.0,
                    pos=lead_pos
                )

            #Lead car resume
            resume_step = self.trigger_step + self.stop_duration
            if current_step >= resume_step and not self.resumed:
                traci.vehicle.setSpeedMode(self.lead_id, 31)
                traci.vehicle.setSpeed(self.lead_id, self.resume_speed)
                traci.vehicle.setColor(self.lead_id, (0, 150, 255, 255))
                logging.info(f"Step {current_step}: Lead car resumed driving at {self.resume_speed} m/s.")
                self.resumed = True

        except traci.TraCIException as err:
            logging.warning(f"SmartOvertakingScenario TraCI warning at step {current_step}: {err}")