import traci


class LongitudinalController:
    # TraCI bitset mode: Default driving checks disable karke pure ML control enable karta hai
    SPEED_MODE_SAFE_ML = 31

    # Visual State Colors: RGBA (Red, Green, Blue, Alpha)
    COLOR_CRUISE = (0, 255, 0, 255)       # Green: Normal cruising / accelerating
    COLOR_DECEL = (255, 165, 0, 255)      # Orange: Smooth ACC auto-braking
    COLOR_EMERGENCY = (255, 0, 0, 255)    # Red: Full stop / Emergency brake

    @staticmethod
    def initialize_vehicle(ego_id: str) -> None:
        """
        Vehicle ke safety parameters, speed mode aur initial visual state set karta hai.
        Runner loop shuru hote hi ise ek baar call karein.
        """
        if ego_id in traci.vehicle.getIDList():
            # SpeedMode 31: External speed control smoothly accept karne ke liye
            traci.vehicle.setSpeedMode(ego_id, LongitudinalController.SPEED_MODE_SAFE_ML)
            traci.vehicle.setEmergencyDecel(ego_id, 9.0)
            traci.vehicle.setColor(ego_id, LongitudinalController.COLOR_CRUISE)

    @staticmethod
    def apply_control(ego_id: str, target_speed: float, duration: float = 0.5) -> None:
        """
        Target speed apply karta hai aur live state ke hisaab se vehicle body color update karta hai.
        """
        # Safety Guard: Check vehicle is active in current simulation step
        if ego_id not in traci.vehicle.getIDList():
            return

        current_speed = traci.vehicle.getSpeed(ego_id)

        # 1. Emergency Braking State (Target speed <= 0.5 m/s)
        if target_speed <= 0.5:
            traci.vehicle.setColor(ego_id, LongitudinalController.COLOR_EMERGENCY)
            traci.vehicle.setEmergencyDecel(ego_id, 9.0)
            # Duration 0.1s - 0.2s taaki instant hard braking action ho
            traci.vehicle.slowDown(ego_id, 0.0, duration=0.2)

        # 2. Smooth Deceleration / Following State
        elif target_speed < current_speed - 0.8:
            traci.vehicle.setColor(ego_id, LongitudinalController.COLOR_DECEL)
            traci.vehicle.slowDown(ego_id, float(target_speed), duration=duration)

        # 3. Free Road Cruising / Acceleration State
        else:
            traci.vehicle.setColor(ego_id, LongitudinalController.COLOR_CRUISE)
            traci.vehicle.slowDown(ego_id, float(target_speed), duration=duration)