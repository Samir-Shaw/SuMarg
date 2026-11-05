import math

class V2VModule:

    def __init__(self, comm_range_meters: float = 150.0):
        self.comm_range = comm_range_meters
        self._message_bus = []

    def broadcast(self, sender_id: str, event_type: str, speed: float, decel: float, pos: tuple):
        packet = {
            "sender_id": sender_id,
            "event": event_type,          
            "speed": speed,
            "decel": decel,
            "position": pos
        }
        self._message_bus.append(packet)

    def receive_for_vehicle(self, receiver_id: str, receiver_pos: tuple) -> list:
        received = []
        for packet in self._message_bus:
            if packet["sender_id"] == receiver_id:
                continue

            #calc dist btw sender and receiver
            dx = packet["position"][0] - receiver_pos[0]
            dy = packet["position"][1] - receiver_pos[1]
            dist = math.hypot(dx, dy)

            if dist <= self.comm_range:
                received.append(packet)

        return received

    def step_cleanup(self):
        self._message_bus.clear()