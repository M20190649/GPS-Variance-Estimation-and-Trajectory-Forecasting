# # -*- coding: utf-8 -*-
"""
This file performs Exploratory Data Analysis on the data received from 
Norrlandsvagnar. The data is plotted using the pandas library.

5508 - Is line 13 on morning of 2018-02-18 

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import sys, getopt
import logging
import pprint
import time
from collections import defaultdict
from gmplot import gmplot
from helpers import setup_logging, epochify, str_gps_to_tuple, print_progress_bar, file_len

def main(argv):
    input_file = '../ostgotatrafiken/VehicleEvents.20180218.log'
    output_file = 'my_map.html'
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('eda.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('eda.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg

    run(input_file, output_file)
    
def run(input_file, output_file):
    setup_logging()
    logging.info("Starting execution of eda.py!")
    #events, timeline_events = read_events_from_file(input_file, max_events=4_000_000, group_by_id=False, vehicle_id=5508)#, max_events = 10_000_000)
    #pprint.pprint(events["PassedEvent"])
    logging.info("Events read and parsed")

    logging.info("Reading in stop locations")
    #stops = read_stop_locations_from_file("../ostgotatrafiken/location-nearbystops-Linkoping.json")

    sorted_events = read_events_from_file(input_file, skip_n=15_000_000, max_events=None)
    state_events = process_events(sorted_events)
    #state_events = read_journeys_from_file(input_file, max_events=4_000_000)

    logging.info("Creating Google Map Plotter")
    gmap = gmplot.GoogleMapPlotter(58.408958, 15.61887, 13)

    logging.info("Plotting Stop Locations")
    #for stop in stops:
    #    gmap.marker(stop["lat"], stop["lon"], title=stop["name"])

    logging.info("Plotting assigned events")
    for event_list in state_events["assigned"]:
        if event_list:
            plot_polygon(gmap, event_list, "orange")

    logging.info("Plotting other events")
    for event_list in state_events["other"]:
        if event_list:
            plot_polygon(gmap, event_list, "red")

    logging.info("Plotting completed events")
    for event_list in state_events["completed"]:
        if event_list:
            plot_polygon(gmap, event_list, "springgreen")

    logging.info("Plotting started events")
    for event_list in state_events["started"]:
        if event_list:
            plot_polygon(gmap, event_list, "cornflowerblue")

    logging.info("Plotting garage events")
    for event_list in state_events["garage"]:
        if event_list:
            plot_polygon(gmap, event_list, "black")
    
    #logging.info("Plotting Observed Position Events")
    #for vehicle_id, vehicle_events in events.items():
    #    if "ObservedPositionEvent" in vehicle_events:
    #        plot_polygon(gmap, vehicle_events["ObservedPositionEvent"], "cornflowerblue")

    #logging.info("Plotting Started Events")
    #plot_scatter(gmap, events["StartedEvent"], 'cornflowerblue')

    #logging.info("Plotting Stopped Events")
    #plot_scatter(gmap, events["StoppedEvent"], 'mistyrose')

    logging.info("Plotting Passed Events")
    #this one: plot_markers(gmap, events["PassedEvent"], 'orange', title="Passed 1")
    #plot_markers(gmap, events["PassedEvent"], 'violet', gps_key="gps2")
    #this one: plot_markers(gmap, events["PassedEvent"], 'yellow', gps_key="stop.gps1", title_key="stop.name")
    #plot_markers(gmap, events["PassedEvent"], 'sandybrown', gps_key="stop.gps2")

    #logging.info("Plotting ArrivedEvent")
    #this one: plot_markers(gmap, events["ArrivedEvent"], 'springgreen', title="Arrived 1")
    #plot_markers(gmap, events["ArrivedEvent"], 'lightblue', gps_key="gps2", title="Arrived 2")
    #this one: plot_markers(gmap, events["ArrivedEvent"], 'yellow', gps_key="stop.gps1", title_key="stop.name")
    #plot_markers(gmap, events["ArrivedEvent"], 'violet', gps_key="stop.gps2", title="Arrived 4")
    #pprint.pprint(events["ArrivedEvent"])
    #pprint.pprint(events["DepartedEvent"])
    #pprint.pprint(events)

    #logging.info("Plotting DepartedEvent")
    #plot_markers(gmap, events["DepartedEvent"], 'springgreen', title="Departed 1")
    #plot_markers(gmap, events["DepartedEvent"], 'lightblue', gps_key="gps2", title="Departed 2")
    #plot_markers(gmap, events["DepartedEvent"], 'orange', gps_key="stop.gps1", title_key="stop.name")
    #plot_markers(gmap, events["DepartedEvent"], 'violet', gps_key="stop.gps2", title="Departed 4")

    logging.info("Drawing Google Maps, saving to: %s", output_file)
    gmap.draw(output_file)

    logging.info("Execution of eda.py finished!")

def plot_polygon(gmap, events, colour, gps_key="gps", edge_width=2):
    """Helper function that takes events, extracts latitudes and longitudes 
    and draws a line between them. Colour is the colour of the line.
    """
    pos = extract_lats_lons(events, gps_key)
    gmap.plot(pos["lats"], pos["lons"], colour, edge_width)

def plot_markers(gmap, events, colour, gps_key="gps", title=None, title_key=None):
    """Helper function that takes events and places them on a google map with a given colour.
    The gps_key parameter determines which gps key-value item to use.
    Title is an optional title on the marker. Will be overridden if title_key is given.
    """
    for event in events:
        gmap.marker(event[gps_key][0], event[gps_key][1], colour, title=(title if title_key is None else event[title_key]))

def plot_heatmap(gmap, events):
    pos = extract_lats_lons(events)
    gmap.heatmap(pos["lats"], pos["lons"])

def plot_scatter(gmap, events, colour, gps_key="gps", size=1):
    pos = extract_lats_lons(events)
    gmap.scatter(pos["lats"], pos["lons"], colour, size, marker=False)

def extract_lats_lons(events, gps_key="gps"):
    lats = []
    lons = []
    #pprint.pprint(events)
    for pos_event in events:
        #pprint.pprint(pos_event)
        lats.append(pos_event[gps_key][0])
        lons.append(pos_event[gps_key][1])
    return {"lats": lats, "lons": lons}

def filter_event(event, field_list=[]):
    """Helper function that takes an event (dict) and retains the fields in the field_list.
    If field_list is empty, all fields are retained in the event (no filtering).
    """
    if not field_list:
        return event
    return {k: v for k, v in event.items() if k in field_list}       


def group_events(events, field_list=[]):
    """Function which takes a list of events and groups them by id.
    If field_list is given then only the fields in the list is included in the filtered event.
    Othewise, all fields are included in the event.
    """
    grouped_events = defaultdict(list)
    [grouped_events[event["id"]].append(filter_event(event, field_list)) for event in events]
    return grouped_events

def read_stop_locations_from_file(file_name):
    data = json.load(open(file_name))
    if not "StopLocation" in data:
        logging.error("No stop locations found in %s!", file_name)
        return None
    return data["StopLocation"]

def read_events_from_file(file_name, skip_n=0, max_events=None):
    """Opens a file containing events and parses them.
    Checks if a journey has begun and saves all the position updates from the bus on that journey.
    Bus stops stopped at or passed are also recorded.
    """
    if max_events is None:
        logging.info("Calculating number of events in file...")
        T = file_len(file_name)
        logging.info("File has %i events", T)
        T -= skip_n
    else:
        T = max_events
    percent = T // 100

    events = []

    with open(file_name, 'r', encoding="latin-1") as f:
        for _ in range(skip_n):
            f.readline()
        for i in range(T):
            if (i+1) % percent == 0 or (i+1) == T:
                print_progress_bar(i+1, T, prefix = 'Progress:', suffix = 'Complete', length = 50)   
            event = parse_event(f.readline())

            if event is None or event["vehicle.id"] != 5508:
                continue

            events.append(event)
    return sorted(events, key=lambda k: k["event.id"])

def process_events(events):
    started_events = []
    assigned_events = []
    completed_events = []
    garage_events = []
    other_events = []

    curr_started_events = []
    curr_assigned_events = []
    curr_completed_events = []
    curr_garage_events = []
    curr_other_events = []

    state = None

    for event in events:
        event_type = event["event.type"]

        if event_type == "JourneyStartedEvent":
            #pprint.pprint("JourneyStarted!")
            print(event)
            state = "started"
            started_events.append(curr_started_events)
            curr_started_events = []
        elif event_type == "JourneyCompletedEvent":
            #pprint.pprint("JourneyCompleted!")
            print(event)
            if state == "assigned":
                print("Journey was assigned before completed!")
                curr_started_events.extend(curr_assigned_events)
                curr_assigned_events = []
            state = "completed"
            completed_events.append(curr_completed_events)
            curr_completed_events = []
        elif event_type == "JourneyAssignedEvent":
            print(event)
        elif event_type == "ParameterChangedEvent":
            print(event)
            if event["line.new"]:
                state = "assigned"
                assigned_events.append(curr_assigned_events)
                curr_assigned_events = []
            else:
                print("GARAGE")
                state = "garage"
                garage_events.append(curr_garage_events)
            curr_garage_events = []
        elif event_type == "ObservedPositionEvent":
            if state == "assigned":
                curr_assigned_events.append(event)
            elif state == "garage":
                curr_garage_events.append(event)
            elif state == "started":
                curr_started_events.append(event)
            elif state == "completed":
                curr_completed_events.append(event)
            else:
                curr_other_events.append(event)

    assigned_events.append(curr_assigned_events)
    started_events.append(curr_started_events)
    garage_events.append(curr_garage_events)
    completed_events.append(curr_completed_events)
    other_events.append(curr_other_events)

    return {
        "assigned": assigned_events,
        "started": started_events,
        "completed": completed_events,
        "garage": garage_events,
        "other": other_events
    }

def read_events_from_file_old(file_name, vehicle_id=None, max_events=None, create_timeline=False, filter_vehicle="Bus", group_by_id=True):
    """Functions that takes a file_name and returns events in the form of key-value objects.

    Parameter "vehicle_id" is an optional parameter that is used to filter 
    results to only contain events for the given vehicle_id.

    Returns a list of objects containing data in a key-value format.
    """
    if group_by_id:
        events = defaultdict(dict)
    else:
        events = defaultdict(list)
    timeline_events = []
    if max_events is None:
        logging.info("Calculating number of events in file...")
        T = file_len(file_name)
        logging.info("File has %i events", T)
    else:
        T = max_events
    percent = T // 100
    with open(file_name, 'r', encoding="latin-1") as f:
        for i in range(T):
            if (i+1) % percent == 0 or (i+1) == T:
                print_progress_bar(i+1, T, prefix = 'Progress:', suffix = 'Complete', length = 50)
            
            event = parse_event(f.readline(), filter_vehicle)
            if event is None:
                continue
            
            event_type = event["event.type"]
            event_v_id = event["vehicle.id"]

            if vehicle_id is None or vehicle_id == event_v_id:
                if group_by_id:
                    if not event_type in events[event_v_id]:
                        events[event_v_id][event_type] = []
                    events[event_v_id][event_type].append(event)
                else:
                    events[event_type].append(event)
                if create_timeline:
                    if not timeline_events or timeline_events[len(timeline_events) - 1]["event.type"] != event_type:
                        timeline_events.append(event)
    return events, timeline_events


def parse_event(line, filter_vehicle="Bus"):
    """Helper function that takes a line containing an event.
    Returns a JSON object of the parsed line.
    """
    if "Normal" not in line:
        return None  # Filter out Warning and Error entries.
    if filter_vehicle is not None and filter_vehicle not in line:
        return None  # Filter out entries with different vehicle type

    header, body = line.split("|||")
    header_fields = header.split("|")    
    body_fields = body.split("|")

    if "ObservedPositionEvent" in header_fields:
        return parse_obspos_event(header_fields, body_fields)
    if "StartedEvent" in header_fields or "StoppedEvent" in header_fields:
        return parse_startstop_event(header_fields, body_fields)
    if "ArrivedEvent" in header_fields:
        return parse_arrived_event(header_fields, body_fields)
    if "DepartedEvent" in header_fields:
        return parse_departed_event(header_fields, body_fields)
    if "PassedEvent" in header_fields:
        return parse_passed_event(header_fields, body_fields)
    if "EnteredEvent" in header_fields:
        return parse_entered_event(header_fields, body_fields)
    if "ExitedEvent" in header_fields:
        return parse_exited_event(header_fields, body_fields)

    if "ParameterChangedEvent" in header_fields and "JourneyRef" in body_fields:
        return parse_paramchanged_event(header_fields, body_fields)
    if "JourneyAssignedEvent" in header_fields:
        return parse_journeyassigned_event(header_fields, body_fields)
    if "JourneyStartedEvent" in header_fields or "JourneyCompletedEvent" in header_fields:
        return parse_journeystartend_event(header_fields, body_fields)

    #print("No match!", header_fields, body_fields)
    return None

def parse_obspos_event(header_fields, body_fields):
    """Parses an "ObservedPositionEvent" and extracts relevant fields.
    Header: date | 2 (?) | event_type | event_id | status (Normal/Warning/Error)
    Body: 0.0 (?)| vehicle (Bus/Train) | junk | gid | vehicle_id | gps (lat, lon) | gps repeat | dir | speed | 3061889 (?)
    """
    return {
        "date": header_fields[0],
        "event.type": header_fields[2],
        "event.id": int(header_fields[3]),
        "vehicle.type": body_fields[1],
        "gid": int(body_fields[3]),
        "vehicle.id": int(body_fields[4]),
        "gps": str_gps_to_tuple(body_fields[5])
    }

# TODO: Examine all types that have functions below!!!!!
# TODO: Events not parsed: "ParameterChangedEvent", "JourneyAssignedEvent", "JourneyStartedEvent", "JourneyCompletedEvent"

def parse_startstop_event(header_fields, body_fields):
    """Parses a "StartedEvent" or a "StoppedEvent" and extracts relevant fields.
    Header: date | 2 (?), event_type, event_id, status (Normal/Warning/Error), 2793946901 (ref event_id?) 
    Body: vehicle | junk | gid | vehicle_id | gps (lat, lon)
    """
    return {
        "date": header_fields[0],
        "event.type": header_fields[2],
        "event.id": int(header_fields[3]),
        "ref.id": int(header_fields[5]),
        "vehicle.type": body_fields[0],
        "gid": int(body_fields[2]),
        "vehicle.id": int(body_fields[3]),
        "gps": str_gps_to_tuple(body_fields[4])
    }

def parse_arrived_event(header_fields, body_fields):
    """Parses an "ArrivedEvent" and extracts relevant fields.
    Header: date | 2 (?), event_type, event_id, status (Normal/Warning/Error), 2793946901 (ref event_id?) 
    Body: 
        0: vehicle
        1: junk
        2: gid (vehicle_id)
        3: vehicle_id
        4: gps (lat, lon)
        5: gid (linje)
        6: gps (repeat)
        7: int (?)
        8: gid@int (node)
        9: name
        10: gps (stop1?)
        11: int (?)
        12: gps (stop2?)
        13: time (arrived?)
        14: time (departed)
        15: Arrived (?)
        16: time (?)
        17: empty
        18: AtStop (?)
    """
    #print(header_fields, body_fields)
    return {
        "date": header_fields[0],
        "event.type": header_fields[2],
        "event.id": int(header_fields[3]),
        "ref.id": int(header_fields[5]),
        "vehicle.type": body_fields[0],
        "gid": int(body_fields[2]),
        "vehicle.id": int(body_fields[3]),
        "gps": str_gps_to_tuple(body_fields[4]),
        "line": int(body_fields[5][7:11]),
        "line.id": int(body_fields[5][11:]),
        "gps2": str_gps_to_tuple(body_fields[6]),
        "stop.int1": float(body_fields[7]),
        "stop.name": body_fields[9],
        "stop.gps1": str_gps_to_tuple(body_fields[10]),
        "stop.int2": float(body_fields[11]),
        "stop.gps2": str_gps_to_tuple(body_fields[12]),
        "time1": body_fields[13],
        "time2": body_fields[14],
        "time3": body_fields[16]
    }

def parse_departed_event(header_fields, body_fields):
    """Parses a "DepartedEvent" and extracts relevant fields.
    Header: date | 2 (?), event_type, event_id, status (Normal/Warning/Error), 2793946901 (ref event_id?) 
    Body: 
        0: vehicle
        1: junk
        2: gid (vehicle_id)
        3: vehicle_id
        4: gps (lat, lon)
        5: gid (linje)
        6: gps (inte repeat?!)
        7: int (?)
        8: gid@int (node)
        9: name
        10: gps (stop1?)
        11: int (?)
        12: gps (stop2?)
        13: time (arrived?)
        14: time (departed)
        15: Arrived (?)
        16: time (?)
        17: time (?)
        18: Departed (?)
    """
    #print(header_fields, body_fields)
    return {
        "date": header_fields[0],
        "event.type": header_fields[2],
        "event.id": int(header_fields[3]),
        "ref.id": int(header_fields[5]),
        "vehicle.type": body_fields[0],
        "gid": int(body_fields[2]),
        "vehicle.id": int(body_fields[3]),
        "gps": str_gps_to_tuple(body_fields[4]),
        "line": int(body_fields[5][7:11]),
        "line.id": int(body_fields[5][11:]),
        "gps2": str_gps_to_tuple(body_fields[6]),
        "stop.int1": float(body_fields[7]),
        "stop.name": body_fields[9],
        "stop.gps1": str_gps_to_tuple(body_fields[10]),
        "stop.int2": float(body_fields[11]),
        "stop.gps2": str_gps_to_tuple(body_fields[12]),
        "time1": body_fields[13],
        "time2": body_fields[14],
        "time3": body_fields[16],
        "time4": body_fields[17]
    }

def parse_passed_event(header_fields, body_fields):
    """Parses a "PassedEvent" and extracts relevant fields.
    Header: date | 2 (?), event_type, event_id, status (Normal/Warning/Error), 2793946901 (ref event_id?) 
        Body: 
        0: vehicle
        1: junk
        2: gid (vehicle_id)
        3: vehicle_id
        4: gps (lat, lon)
        5: gid (linje)
        6: gps (inte repeat?!)
        7: int (?)
        8: gid@int (node)
        9: name
        10: gps (stop1?)
        11: int (?)
        12: gps (stop2? repeat av gps#6?)
        13: time (arrived?)
        14: time (departed)
        15: Passed (?)
        16: time (?)
        17: time (?)
        18: Passed (?)
    """
    #print(header_fields, body_fields)
    return {
        "date": header_fields[0],
        "event.type": header_fields[2],
        "event.id": int(header_fields[3]),
        "ref.id": int(header_fields[5]),
        "vehicle.type": body_fields[0],
        "gid": int(body_fields[2]),
        "vehicle.id": int(body_fields[3]),
        "gps": str_gps_to_tuple(body_fields[4]),
        "line": int(body_fields[5][7:11]),
        "line.id": int(body_fields[5][11:]),
        "gps2": str_gps_to_tuple(body_fields[6]),
        "stop.int1": float(body_fields[7]),
        "stop.name": body_fields[9],
        "stop.gps1": str_gps_to_tuple(body_fields[10]),
        "stop.int2": float(body_fields[11]),
        "stop.gps2": str_gps_to_tuple(body_fields[12]),
        "time1": body_fields[13],
        "time2": body_fields[14],
        "time3": body_fields[16],
        "time4": body_fields[17]
    }

def parse_entered_event(header_fields, body_fields):
    """Parses a "EnteredEvent" and extracts relevant fields.
    Header: date | 2 (?), event_type, event_id, status (Normal/Warning/Error), 2793946901 (ref event_id?) 
    Body: vehicle | junk | gid | vehicle_id | gps1 (lat, lon) | gid (?) | gps (?) | int | gid@int | name | gps (?) | int (?) | gps1 (repeat?)
    """
    #print(header_fields, body_fields)
    return None

def parse_exited_event(header_fields, body_fields):
    """Parses a "ExitedEvent" and extracts relevant fields.
    Header: date | 2 (?), event_type, event_id, status (Normal/Warning/Error), 2793946901 (ref event_id?) 
    Body: vehicle | junk | gid | vehicle_id | gps1 | gid (?) | gps (?) | int | gid@int | name | gps (?) | int (?) | gps (?)
    """
    #print(header_fields, body_fields)
    return None

def parse_paramchanged_event(header_fields, body_fields):
    """
    Header: date | 2 (?), event_type, event_id, status (Normal/Warning/Error), 2793946901 (ref event_id?) 
    Body:
        0: vehicle
        1: junk
        2: gid (vehicle_id)
        3: vehicle_id
        4: gps
        5: JourneyRef (?)
        6: gid (old line?)
        7: gid (new line?)
    """
    #print(header_fields, body_fields)
    return {
        "date": header_fields[0],
        "event.type": header_fields[2],
        "event.id": int(header_fields[3]),
        "ref.id": int(header_fields[5]),
        "vehicle.type": body_fields[0],
        "gid": int(body_fields[2]),
        "vehicle.id": int(body_fields[3]),
        # TODO: These lines below need to be changed.
        #"gps": str_gps_to_tuple(body_fields[4]) if body_fields[4] else "",
        "line.old": int(body_fields[6][7:11]) if body_fields[6] else "",
        "line.old.id": int(body_fields[6][11:]) if body_fields[6] else "",
        "line.new": int(body_fields[7][7:11]) if body_fields[7] != '\n' else "",
        "line.new.id": int(body_fields[7][11:]) if body_fields[7] != '\n' else ""
    }

def parse_journeyassigned_event(header_fields, body_fields):
    """
    Header: date | 2 (?), event_type, event_id, status (Normal/Warning/Error), 2793946901 (ref event_id?) 
    Body:
        0: vehicle
        1: junk
        2: gid (vehicle_id)
        3: vehicle_id
        4: gps
        5: gid (line)
        6: date (when journey begins?)
    """
    #print(header_fields, body_fields)
    return {
        "date": header_fields[0],
        "event.type": header_fields[2],
        "event.id": int(header_fields[3]),
        "ref.id": int(header_fields[5]),
        "vehicle.type": body_fields[0],
        "gid": int(body_fields[2]),
        "vehicle.id": int(body_fields[3]),
        #"gps": str_gps_to_tuple(body_fields[4]),
        "line": int(body_fields[5][7:11]),
        "line.id": int(body_fields[5][11:]),
        "date.new": body_fields[6]
    }

def parse_journeystartend_event(header_fields, body_fields):
    """
    Header: date | 2 (?), event_type, event_id, status (Normal/Warning/Error), 2793946901 (ref event_id?) 
    Body:
        0: vehicle
        1: junk
        2: gid (vehicle_id)
        3: vehicle_id
        4: gps
        5: gid (line)
    """
    #print(header_fields, body_fields)
    return {
        "date": header_fields[0],
        "event.type": header_fields[2],
        "event.id": int(header_fields[3]),
        "ref.id": int(header_fields[5]),
        "vehicle.type": body_fields[0],
        "gid": int(body_fields[2]),
        "vehicle.id": int(body_fields[3]),
        #"gps": str_gps_to_tuple(body_fields[4]),
        "line": int(body_fields[5][7:11]),
        "line.id": int(body_fields[5][11:]),
    }


if __name__ == "__main__":
    main(sys.argv[1:])