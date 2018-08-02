class Journey:
    """Simple journey class. Contains:
    * One journey (sequence of positions)
    * Stops made during the journey
    * The vehicle_id for the bus making the journey.
    """
    def __init__(self, vehicle_id, line_number):
        self.vehicle_id = vehicle_id
        self.line_number = line_number
        self.route = []
        self.bus_stops = []

    def add_stop(self, stop):
        if stop["event.type"] in ["ArrivedEvent", "DepartedEvent", "PassedEvent"]:
            name = stop["stop.name"]
            if name not in self.bus_stops:
                self.bus_stops.append(name)
        self.route.append(stop)