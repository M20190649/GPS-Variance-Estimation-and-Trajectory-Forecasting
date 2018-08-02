import pandas
import pickle
import sys, getopt
import logging
import gpflow
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import ml_helper as hlp
from pprint import pprint
from datetime import timedelta, datetime
from sklearn.preprocessing import StandardScaler
from helpers import setup_logging
from collections import defaultdict
from gmplot import gmplot
from operator import add
from scipy.stats import multivariate_normal

def main(argv):
    line_number = None
    create_variation_GPs = True
    help_line = 'usage: forecasting.py -l <line_number> --load-variation-gp'
    try:
        opts, args = getopt.getopt(argv,"hl:",["line=", "load-variation-gp"])
    except getopt.GetoptError:
        print(help_line)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_line)
            sys.exit()
        elif opt in ("-l", "--line"):
            line_number = arg
        elif opt == "--load-variation-gp":
            create_variation_GPs = False

    run(line_number, create_variation_GPs)
    
def run(line_number, create_variation_GPs):
    logger.info("Starting execution of forecasting.py!")

    visualise_trajectory_gp = False
    visualise_tau_gp = False

    trajectories = hlp.load_trajectories_from_file(line_number, logger)
    
    trajectory_key = "Lötgatan:Linköpings resecentrum"
    vehicle_id, journey = trajectories[trajectory_key][0]
    #pprint.pprint(journey.route)

    f1_gp, scaler = create_f1_GP(journey.route)

    trajectory_i = 0
    for vehicle_id, journey in trajectories[trajectory_key]:
        filtered_events = list(filter(filter_func, journey.route))
        
        # Find segment starts:
        segment_starts = []
        compressed_events = []
        stop_delay = timedelta(seconds=0)
        start_stop_time = None
        for event in filtered_events:
            if event["event.type"] == "ObservedPositionEvent":
                if start_stop_time is not None:
                    print("New End Stop Time")
                    end_stop_time = event["date"]
                    stop_delay += (end_stop_time - start_stop_time)
                    start_stop_time = None
                event["date"] -= stop_delay
                print(event["date"])
                compressed_events.append(event)
            else:
                index = len(compressed_events)
                if segment_starts and segment_starts[-1][0] == index:
                    segment_starts[-1] += (event["event.type"],)
                else:
                    print("New Stop")
                    start_stop_time = event["date"]
                    segment_starts.append((index, event["event.type"]))
        pprint(segment_starts)

        #hlp.plot_speed_stops(journey.route, filter_speed=0.1)
        #hlp.plot_speed_time(journey)

        # Do segmentation:
        segments = []
        segment_overlap = 5
        start = 0
        for next_start, *_types in segment_starts: #  We don't care about slice [1:] of the tuple (_types)
            # TODO: Make sure this works even if a route does not end with speed = 0.
            if next_start != 0:
                overlapped_start = max(0, start - segment_overlap) #  overlapped_start >= 0
                overlapped_stop = next_start + segment_overlap
                print(overlapped_start, next_start, overlapped_stop)
                segments.append(compressed_events[overlapped_start:overlapped_stop])
                start = next_start

        for i, segment in enumerate(segments):
            hlp.plot_speed_stops(segment, file_name="speed_stops/{}/segment_{}".format(trajectory_i, i), f1_gp=f1_gp, scaler=scaler)
        trajectory_i += 1
    

def create_f1_GP(route):
    events = [e for e in route if e["event.type"] == "ObservedPositionEvent" and e["speed"] > 1]
    events = hlp.filter_duplicates(events)

    X = np.vstack([e["gps"] for e in events])
    Y = np.linspace(0, 1, num = X.shape[0]).reshape(-1,1)
    scaler = StandardScaler().fit(X)
    X_fitted = scaler.transform(X)

    f1_gp = hlp.train_f1_gp(X_fitted, Y, logger)
    return (f1_gp, scaler)


def filter_func(event):
    if event["event.type"] == "ObservedPositionEvent":
        return event["speed"] > 0.1
    return True

if __name__ == "__main__":
    logger = setup_logging("forecasting.py", "forecasting.log")
    main(sys.argv[1:])