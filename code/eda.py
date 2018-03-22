# # -*- coding: utf-8 -*-
"""
This file performs Exploratory Data Analysis on the data received from 
Norrlandsvagnar. The data is plotted using the pandas library.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import logging
import time
from collections import defaultdict
from helpers import setup_logging, epochify

def main():
    setup_logging()
    logging.info("Starting execution of eda.py!")
    events = read_events_from_file("example.data")
    
    relevant_fields = [
        "date", 
        "can.diesel1", 
        "can.diesel2", 
        "can.current_battery",
        "can.temp_in",
        "can.temp_out",
        "can.current_sun",
        "can.voltage",
        "can.heating_temp"]
    grouped_events = group_events(events, field_list=relevant_fields)
    
    event_id = events[0]["id"]
    print(grouped_events[event_id])
    return

    current_battery_levels = [(int(event["can.temp_out"], base=16), epochify(event["date"])) for event in events if event["id"] == generator_id]
    print(current_battery_levels)
    logging.info("Execution of eda.py finished!")


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


def read_events_from_file(file_name):
    """Functions that takes a file_name and returns events in the form of JSON objects.
    The events are validated to contain relevant sensor data.
    """
    re_pattern = r'(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?P<type_header>.*?)(?P<json_obj>{.*?})'
    date_reg = re.compile(re_pattern, flags=re.DOTALL)

    with open(file_name, 'r') as f:
        events_string = " ".join(f.readlines())

    events = []
    for match_obj in date_reg.finditer(events_string):
        event = parse_event(match_obj.group("date"), match_obj.group("type_header"), match_obj.group("json_obj"))
        if event is not None:
            events.append(event)
    return events


def parse_event(event_date, event_header, event_obj):
    """Helper function that takes three arguments: 
    the date of the event, what type of event it is, and the content of the event.

    Returns None if event is not relevant or if it does not contain relevant sensor data.
    Otherwise, a parsed JSON object is returned.
    """
    if "pushData" not in event_header or not event_relevant(event_obj):
        return None  # We only want to process relevant pushData events (not image uploads, etc.)

    event_obj = event_obj.replace("'", '"')    # Re-format single to double-quotes.

    if " id:" in event_obj:
        event_obj = event_obj.replace(" id:", ' "id":')  # Encapsulate id: in double-quotes
    if " hwid:" in event_obj:
        event_obj = event_obj.replace(" hwid:", ' "hwid":')  # Encapsulate hwid: in double-quotes
        
    json_obj = json.loads(event_obj)  # Convert to JSON object.
    json_obj["date"] = event_date  # Add date to json_obj for later reference.
    return json_obj


def event_relevant(event_obj):
    """Helper function that checks if the event object contains relevant sensor data.
    Returns True if it does, False if it does not.
    """
    # TODO: Some of these might be relevant to extract!
    # connection    => False    (Events with this payload is probably not relevant)
    # gps           => False    (Events with only gps payload is not relevant)
    # power.status1 => False    (Might be relevant! Maybe it contains error messages)
    return "can.voltage" in event_obj


if __name__ == "__main__":
    main()