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


def distance(p1, p2):
    x1, y1 = p1["gps"][::-1]
    x2, y2 = p2["gps"][::-1]
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def load_f1_GP_revamped(session, logger, base_dir, version=""):
    f1_gp = session.load(base_dir + "f1_gp{}".format(version))
    f1_scaler = load_array("f1_scaler", logger, base_dir)
    return f1_gp, f1_scaler

def create_f1_GP_revamped(route, session, logger, base_dir, load_model, version=""):
    events = [e for e in route if e["event.type"] == "ObservedPositionEvent" and e["speed"] > -1]
    events = filter_duplicates(events)
    prev_point = events[0]
    distance_filtered_events = [prev_point]
    for event in events[1:]:
        dist = distance(event, prev_point)
        if  dist > 6e-05:
            distance_filtered_events.append(event)
            prev_point = event
    
            
    X = np.vstack([e["gps"][::-1] for e in distance_filtered_events])
    Y = np.linspace(0, 1, num = X.shape[0]).reshape(-1,1)
    
    if load_model:
        f1_gp, f1_scaler = load_f1_GP_revamped(session, logger, base_dir, version)
    else:
        f1_scaler = StandardScaler().fit(X)
        X_fitted = f1_scaler.transform(X)
        f1_gp = train_f1_gp_revamped(X_fitted, Y, logger, constrain=True)
        session.save(base_dir + "f1_gp{}".format(version), f1_gp)
        save_array(f1_scaler, "f1_scaler", logger, base_dir)
    
    return {
        "f1_gp": f1_gp, 
        "f1_scaler": f1_scaler,
        "X": X,
        "Y": Y
    }


def get_all_trajectories(trajectories, trajectory_key):
    trajectory_start, trajectory_end = trajectory_key.split(":")
    all_trajectories = []
    for key, values in trajectories.items():
        k_start, k_end = key.split(":")
        if trajectory_start == k_start:
            for vehicle_id, journey in values:
                segmented_journey, _rest = journey.segment_at(trajectory_end)
                _start, main_journey = segmented_journey.segment_at("Mariedalsgatan")
                all_trajectories.append((vehicle_id, main_journey))
    return all_trajectories


def visualise_f1_gp_revamped(X, Y, f1_gp, f1_scaler, other_points, file_name="f1_gp", title="f1", base_dir=""):
    res = 200
    #lat, lng = np.mgrid[58.410317:58.427006:res*1j, 15.490352:15.523200:res*1j] # Big Grid
    #lat, lng = np.mgrid[58.416317:58.42256:res*1j, 15.494352:15.503200:res*1j] # Middle Grid
    #lat, lng = np.mgrid[58.4173:58.419:res*1j, 15.4965:15.499:res*1j] # Small Grid
    lat, lng = np.mgrid[58.4190:58.422:res*1j, 15.500:15.502:res*1j] # Small Grid (new start)

    pos_grid = np.dstack((lng, lat)).reshape(-1, 2)
    print("Grid created.")
    pos_grid_fitted = f1_scaler.transform(pos_grid)
    print("Grid scaled.")
    #print(pos_grid_fitted)
    grid_tau_mapped, _var = f1_gp.predict_y(pos_grid_fitted)
    print("Grid predicted.")
    #grid_tau_mapped_fitted = inv_scaler.transform(grid_tau_mapped)

    #proj_x, _var = inv_f1_gp[0].predict_y(grid_tau_mapped_fitted)
    #proj_y, _var = inv_f1_gp[1].predict_y(grid_tau_mapped_fitted)
    #proj_points = np.array([[x[0], y[0]] for (x, y) in zip(proj_x, proj_y)])
    #proj_points_inv = f1_scaler.inverse_transform(proj_points)
    #print(proj_points_inv)
    print(len(X))
    X_test = X[:42]
    X_test_fitted = f1_scaler.transform(X_test)
    means, _var = f1_gp.predict_y(X_test_fitted)
    #means_fitted = inv_scaler.transform(means)
    #test_proj_x, _var = inv_f1_gp[0].predict_y(means_fitted)
    #test_proj_y, _var = inv_f1_gp[1].predict_y(means_fitted)
    #test_proj_points = np.array([[x[0], y[0]] for (x, y) in zip(test_proj_x, test_proj_y)])
    #test_proj_points_inv = f1_scaler.inverse_transform(test_proj_points) 
    #for i, x in enumerate(X_test):
    #    print(x, test_proj_points_inv[i])
    plt.figure()
    plt.title(title)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.scatter(pos_grid[:,0], pos_grid[:,1]),#, c=grid_tau_mapped[:,0], cmap="cool", s=0.5)
    min_x = min(pos_grid[:,0])
    max_x = max(pos_grid[:,0])
    min_y = min(pos_grid[:,1])
    max_y = max(pos_grid[:,1])
    v_min = min(means[:,0])#min(min(means[:,0]), min(grid_tau_mapped[:,0]))
    v_max = max(means[:,0])#max(max(min(means[:,0]), max(grid_tau_mapped[:,0]))
    #plt.plot(X_test[:,0], X_test[:,1], "wx", mew=1.5, zorder=2)
    plt.scatter(X_test[:,0], X_test[:,1], c=means[:,0], edgecolors="w", vmin=v_min, vmax=v_max, cmap="cool", zorder=4)
    plt.colorbar()
    #plt.scatter(X_test[:,0], X_test[:,1], color="w", s=0.25, zorder=5)
    tightness = 0
    plt.xlim(min_x - (max_x - min_x)*tightness, max_x + (max_x - min_x)*tightness)
    plt.ylim(min_y - (max_y - min_y)*tightness, max_y + (max_y - min_y)*tightness)
    
    plt.scatter(pos_grid[:,0], pos_grid[:,1], c=grid_tau_mapped[:,0], vmin=v_min, vmax=v_max, cmap="cool", s=3, zorder=1)
    for points in other_points:
        plt.scatter(points[:,0], points[:,1], color="k", s=0.2, zorder=3)
    #plt.contourf(pos_grid, pos_grid[:,1], c=grid_tau_mapped[:,0], cmap="cool")
    
    #plt.plot(proj_points_inv[:,0], proj_points_inv[:,1], "kx", mew=2)

    #plt.scatter(X[:,0], X[:,1], c=Y[:,0], cmap='coolwarm', s=0.5)
    plt.savefig("{}{}.png".format(base_dir, file_name))


def train_f1_gp_revamped(X_train, Y_train, logger, constrain=False):
    """GP which maps lng, lat -> Tau.
    X_train should be standardised and should not contain any stops."""
    with gpflow.defer_build():
        #m = gpflow.models.GPR(X_train, Y_train, kern=gpflow.kernels.RBF(2, lengthscales=0.01))
        m = gpflow.models.GPR(X_train, Y_train, kern=gpflow.kernels.RBF(2, ARD=True))
        if constrain:
            m.likelihood.variance = 1e-05
            m.likelihood.variance.trainable = False
        else:
            m.likelihood.variance = 30
        print(m)
        #m = gpflow.models.GPMC(X_train, Y_train, gpflow.kernels.RBF(2), gpflow.likelihoods.Gaussian())
        m.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m)
        logger.info("Model optimized")
        logger.info(m)
        logger.info("Revamped f1 GP trained.")
    return m


def train_inv_f1_gp(X_train, Y_train, logger):
    """GP which maps Tau -> lng or lat."""
    with gpflow.defer_build():
        m = gpflow.models.GPR(X_train, Y_train, kern=gpflow.kernels.RBF(1))
        m.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m)
        logger.info("Model optimized")
        logger.info(m)
        logger.info("Inverse f1 GP trained.")
    return m


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