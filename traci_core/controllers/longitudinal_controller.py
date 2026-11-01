# Apply traci.vehicle.slowDown() for smooth auto-braking
import traci

class LongitudinalController:
    @staticmethod
    def apply_control(ego_id, target_speed):
        """
        Target speed ke mutabiq slow down ya full brake lagata hai.
        """
        if target_speed <= 0.5:
            # Full stop / Emergency brake
            traci.vehicle.setEmergencyDecel(ego_id, 9.0)
            traci.vehicle.slowDown(ego_id, 0.0, duration=0.5)
        else:
            # Smooth ACC Slow Down
            traci.vehicle.slowDown(ego_id, float(target_speed), duration=1.0)