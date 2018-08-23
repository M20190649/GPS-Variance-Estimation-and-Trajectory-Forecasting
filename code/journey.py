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


    def segment_at(self, bus_stop_name):
        """ Nice function which takes a bus_stop_name and 
        segments the journey into two journeys.

        For example: The journey (A->B->C->D->E) segmented at C
        returns two new journeys: (A->B->C) and (C->D->E).
        """
        if bus_stop_name == self.bus_stops[-1]: # No segmentation needed
            return self, None
        seg1 = Journey(self.vehicle_id, self.line_number)
        seg2 = Journey(self.vehicle_id, self.line_number)
        bus_stop_seg_index = self.bus_stops.index(bus_stop_name)
        seg1.bus_stops = self.bus_stops[:bus_stop_seg_index+1] # Include stop we segment at!
        seg2.bus_stops = self.bus_stops[bus_stop_seg_index:]

        seg_index = None
        # Get route index at segmentation:
        for i, event in enumerate(self.route):
            if "stop.name" in event and event["stop.name"] == bus_stop_name:
                seg_index = i
                break
        if seg_index is None:
            return None, None

        seg1.route = self.route[:seg_index+1]
        seg2.route = self.route[seg_index:]
        return seg1, seg2


    def add_stop(self, stop):
        if stop["event.type"] in ["ArrivedEvent", "DepartedEvent", "PassedEvent"]:
            name = stop["stop.name"]
            if name not in self.bus_stops:
                self.bus_stops.append(name)
        self.route.append(stop)