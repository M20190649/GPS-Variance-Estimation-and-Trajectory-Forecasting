import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
import gpflow
from gmplot import gmplot
import os
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


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_f1_GP(route, logger, visualise_gp=False):
    events = [e for e in route if e["event.type"] == "ObservedPositionEvent" and e["speed"] > 3]
    events = filter_duplicates(events)

    X = np.vstack([e["gps"][::-1] for e in events])
    Y = np.linspace(0, 1, num = X.shape[0]).reshape(-1,1)
    scaler = StandardScaler().fit(X)
    X_fitted = scaler.transform(X)

    f1_gp = train_f1_gp(X_fitted, Y, logger)

    if visualise_gp:
        visualise_f1_gp(X_fitted, Y, file_name="trained", title="f1 (trained)")
    return (f1_gp, scaler)


def visualise_f1_gp(X, Y, file_name, title):
    plt.figure()
    plt.title(title)
    plt.scatter(X[:,0], X[:,1], c=Y[:,0], cmap='coolwarm', s=0.5)
    plt.colorbar()
    plt.savefig("tau_gp/{}.png".format(file_name))


def train_f1_gp(X_train, Y_train, logger):
    """GP which maps lng, lat -> Tau.
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


def load_array(array_name, logger=None, base_dir=""):
    with open("{}{}.pickle".format(base_dir, array_name), 'rb') as file:
        if logger is None: 
            print("Array loaded from {}{}.pickle".format(base_dir, array_name))
        else:
            logger.info("Array loaded from %s%s.pickle", base_dir, array_name)
        return pickle.load(file)


def save_array(array, array_name, logger=None, base_dir=""):
    with open("{}{}.pickle".format(base_dir, array_name), 'wb') as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if logger is None: 
            print("Array saved to {}{}.pickle".format(base_dir, array_name))
        else:
            logger.info("Array saved to %s%s.pickle", base_dir, array_name)


def plot_speed_time(journey, filter_speed=0.1, file_id=""):
    plt.figure()
    plt.subplot(211)
    events = [event for event in journey.route if event["event.type"] == "ObservedPositionEvent" and event["speed"] > filter_speed]
    positions = np.vstack([event["gps"] for event in events])
    speeds = [e["speed"] for e in events]
    start_time = events[0]["date"]
    relative_times = [timedelta.total_seconds(event["date"] - start_time) for event in events]
    plt.scatter(relative_times, speeds, s=0.5)
    plt.title('Compressed Speeds (Filter: {} m/s)'.format(filter_speed))
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.subplot(212)
    plt.title("Journey Coordinates")
    plt.scatter(positions[:,1], positions[:,0], c=speeds, cmap='hot', s=0.5)
    stops = [event for event in journey.route if event["event.type"] == "ObservedPositionEvent" and event["speed"] == 0]
    stops = np.vstack([stop["gps"] for stop in stops])
    plt.scatter(positions[:,1], positions[:,0], c=speeds, cmap='hot', s=0.5)
    plt.scatter(stops[:,1], stops[:,0], color="black", s=0.5)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.savefig("speed_and_path_{}.png".format(file_id))


def plot_coordinates(trajectories):
    plt.figure()
    plt.title("Trajectories")
    for key, journey in trajectories:
        pos = np.vstack([event["gps"][::-1] for event in journey.route])
        plt.scatter(pos[:,0], pos[:,1], s=0.5)
    plt.savefig("coordinates.png")


def plot_trajectories(trajectories, logger, print_only=False):
    """ Function that prints out trajectories to console and plots trajectories in Google Maps.
        if print_only=True, it will only print trajectories to the console."""
    for key, journeys in trajectories.items():
        if not print_only:
            gmap = gmplot.GoogleMapPlotter(58.408958, 15.61887, 13)
        logger.info("{} ({})".format(key, len(journeys)))
        ids = set()
        for vehicle_id, journey in journeys:
            ids.add(vehicle_id)
            if not print_only:
                lats, lngs = zip(*[e["gps"] for e in journey.route])
                gmap.plot(lats, lngs, color="red")
        logger.info("    {}".format(ids))
        if not print_only:
            gmap.draw("ml_map/{}.html".format(key))


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
        start_end = scaler.transform([route[0]["gps"][::-1], route[-1]["gps"][::-1]])
        taus, _ = f1_gp.predict_y(start_end)
        print(taus)
        title = 'Speed (m/s) Segment: {}-{}'.format(round(taus[0][0], 3), round(taus[-1][0], 3))
    else:
        title = 'Speed (m/s)'
    plt.title(title)

    plt.savefig("{}.png".format(file_name))
    plt.clf()
    plt.close()