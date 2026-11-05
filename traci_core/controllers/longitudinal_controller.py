import traci

class LongitudinalController:
    SPEED_MODE_SAFE_ML = 31

    COLOR_CRUISE = (0, 255, 0, 255)       
    COLOR_DECEL = (255, 165, 0, 255)    
    COLOR_EMERGENCY = (255, 0, 0, 255)   

    @staticmethod
    def initialize_vehicle(ego_id: str) -> None:

        if ego_id in traci.vehicle.getIDList():
            #SpeedMode 31-External speed control smoothly accept karne ke liye
            traci.vehicle.setSpeedMode(ego_id, LongitudinalController.SPEED_MODE_SAFE_ML)
            traci.vehicle.setEmergencyDecel(ego_id, 9.0)
            traci.vehicle.setColor(ego_id, LongitudinalController.COLOR_CRUISE)

    @staticmethod
    def apply_control(ego_id: str, target_speed: float, duration: float = 0.5) -> None:

        if ego_id not in traci.vehicle.getIDList():
            return

        current_speed = traci.vehicle.getSpeed(ego_id)

        #Emergency Braking State
        if target_speed <= 0.5:
            traci.vehicle.setColor(ego_id, LongitudinalController.COLOR_EMERGENCY)
            traci.vehicle.setEmergencyDecel(ego_id, 9.0)
            # Duration 0.1s - 0.2s taaki instant hard braking action ho
            traci.vehicle.slowDown(ego_id, 0.0, duration=0.2)

        #Smooth Deceleration
        elif target_speed < current_speed - 0.8:
            traci.vehicle.setColor(ego_id, LongitudinalController.COLOR_DECEL)
            traci.vehicle.slowDown(ego_id, float(target_speed), duration=duration)

        #Free Road Cruising / Acceleration State
        else:
            traci.vehicle.setColor(ego_id, LongitudinalController.COLOR_CRUISE)
            traci.vehicle.slowDown(ego_id, float(target_speed), duration=duration)