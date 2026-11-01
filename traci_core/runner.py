# Main TraCI while loop connecting Sensors -> ML -> Actuators
import traci
from traci_core.config import SUMO_CONFIG_PATH, STEP_LENGTH, EGO_VEHICLE_ID
from traci_core.controllers.longitudinal_controller import LongitudinalController
from ml_engine.src.pipeline.predict_pipeline import RiskPredictor

def run():
    predictor = RiskPredictor()
    traci.start(["sumo-gui", "-c", SUMO_CONFIG_PATH, "--step-length", str(STEP_LENGTH)])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        if EGO_VEHICLE_ID in traci.vehicle.getIDList():
            ego_speed = traci.vehicle.getSpeed(EGO_VEHICLE_ID)
            leader = traci.vehicle.getLeader(EGO_VEHICLE_ID, 150.0)
            
            if leader:
                lead_id, gap = leader
                lead_speed = traci.vehicle.getSpeed(lead_id)
                rel_speed = ego_speed - lead_speed
                
                # ML model se target speed predict karwana
                target_speed = predictor.predict_target_speed(ego_speed, lead_speed, gap, rel_speed)
                
                # Actuator controller ke through command apply karna
                LongitudinalController.apply_control(EGO_VEHICLE_ID, target_speed)
                
    traci.close()

if __name__ == "__main__":
    run()