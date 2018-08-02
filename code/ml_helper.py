import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import numpy as np
import gpflow
from pprint import pprint


def load_trajectories_from_file(line_number, logger):
    with open("line_{}.pickle".format(line_number), 'rb') as file:
        bus_line = pickle.load(file)
    logger.info("Line number %s", bus_line.line_number)
    logger.info("Journeys: %s", len([x for y in bus_line.journeys.values() for x in y]))

    trajectories = defaultdict(list)
    for vehicle_id, journeys in bus_line.journeys.items():
        for journey in journeys:
            key = "{}:{}".format(journey.bus_stops[0], journey.bus_stops[-1])
            trajectories[key].append((vehicle_id, journey))
    return trajectories


def train_f1_gp(X_train, Y_train, logger):
    """GP which maps lat, lng -> Tau.
    X_train should be standardised and should not contain any stops."""
    with gpflow.defer_build():
        logger.info(X_train.shape)
        
        logger.info(X_train)
        logger.info(Y_train)

        m = gpflow.models.GPR(X_train, Y_train, kern=gpflow.kernels.Matern32(2))
        logger.info(m)

        m.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m)
        logger.info("Model optimized")
        logger.info(m)
        logger.info("f1 GP trained.")
    return m


def filter_duplicates(events):
        new_events = []
        for j, event in enumerate(events):
            if j > 0 and event["date"] == events[j-1]["date"]:
                assert event["gps"] == events[j-1]["gps"]
                continue
            new_events.append(event)
        return new_events


def plot_speed_time(journey):
    plt.figure()
    plt.subplot(211)
    events = [event for event in journey.route if event["event.type"] == "ObservedPositionEvent" and event["speed"] > 3]
    positions = np.vstack([event["gps"] for event in events])
    start_time = events[0]["date"]
    relative_times = [timedelta.total_seconds(event["date"] - start_time) for event in events]
    plt.scatter(relative_times, [e["speed"] for e in events], s=0.5)
    plt.title('Speed (m/s) and relative journey time')
    plt.subplot(212)
    plt.title("GPS coordinates")
    plt.scatter(positions[:,1], positions[:,0], s=0.5)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    plt.savefig("time_and_speed.png")


def plot_speed_stops(route, filter_speed=0, file_name=None, f1_gp=None, scaler=None):
    file_name = "speed_and_stops_filter_{}".format(filter_speed) if file_name is None else file_name
    plt.figure()
    speed_events = [e for e in route if e["event.type"] == "ObservedPositionEvent" and e["speed"] > filter_speed]
    other_events = filter(lambda e: e["event.type"] != "ObservedPositionEvent", route)
    start_time = speed_events[0]["date"]
    relative_times = [timedelta.total_seconds(event["date"] - start_time) for event in speed_events]
    #print(relative_times)   
    plt.scatter(relative_times, [e["speed"] for e in speed_events], s=0.5)
    
    for event in other_events:
        rel_time = timedelta.total_seconds(event["date"] - start_time)
        if event["event.type"] == "StoppedEvent":
            plt.axvline(x=rel_time, color="r", linestyle='--', lw=0.5)
        elif event["event.type"] == "StartedEvent":
            plt.axvline(x=rel_time, color="g", linestyle='--', lw=0.5)
        elif event["event.type"] == "ArrivedEvent":
            plt.axvline(x=rel_time, color="m", linestyle='--', lw=0.5)
        elif event["event.type"] == "DepartedEvent":
            plt.axvline(x=rel_time, color="k", linestyle='--', lw=0.5)
        elif event["event.type"] == "PassedEvent":
            plt.axvline(x=rel_time, color="y", linestyle='--', lw=0.5)
    
    if f1_gp is not None and scaler is not None:
        start_end = scaler.transform([route[0]["gps"], route[-1]["gps"]])
        taus, _ = f1_gp.predict_y(start_end)
        title = 'Speed (m/s) Segment: {}-{}'.format(round(taus[0][0], 3), round(taus[-1][0], 3))
    else:
        title = 'Speed (m/s)'
    plt.title(title)

    plt.savefig("{}.png".format(file_name))
    plt.clf()
    plt.close()