from collections import defaultdict
from journey import Journey

class BusLine:
    """Simple Bus Line class containing:
    * Bus Line Number
    * Routes for the bus line
    * Stops
    """
    def __init__(self, line_number):
        """Constructor. line_number is the bus line number."""
        self.line_number = line_number
        self.journeys = {}  # key: vehicle_id, value: Journey instances

    def add_journey(self, journey):
        if isinstance(journey, Journey):
            if not journey.vehicle_id in self.journeys:
                self.journeys[journey.vehicle_id] = []
            self.journeys[journey.vehicle_id].append(journey)
        else:
            raise Exception("Trying to add something which is not a journey", journey)


            


