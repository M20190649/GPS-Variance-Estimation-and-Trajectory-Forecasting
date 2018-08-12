import pandas
import pickle
import sys, getopt
import warnings
import logging
import gpflow
import os
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

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

def main(argv):
    line_number = None
    load_model = False
    help_line = 'usage: forecasting.py -l <line_number> --load-model'
    try:
        opts, args = getopt.getopt(argv,"hl:",["line=", "load-model"])
    except getopt.GetoptError:
        print(help_line)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_line)
            sys.exit()
        elif opt in ("-l", "--line"):
            line_number = arg
        elif opt == "--load-model":
            load_model = True

    run(line_number, load_model)
    
def run(line_number, load_model):
    logger.info("Starting execution of forecasting.py!")

    trajectories = hlp.load_trajectories_from_file(line_number, logger)
    
    trajectory_key = "Lötgatan:Linköpings resecentrum"
    kernels = ["rbf_linear"]
    
    all_trajectories = get_all_trajectories(trajectories, trajectory_key)

    if load_model:
        model = load_forecasting_model(kernels=kernels)
    else:
        create_forecasting_model(
            all_trajectories, 
            model_trajectory=trajectories[trajectory_key][0], 
            kernels=kernels)
        exit(1)

    f1_gp = model["f1_gp"]
    scaler = model["f1_scaler"]
    segment_GPs = model["segment_GPs"]
    pprint(segment_GPs)
    #speed_scaler = model["speed_scaler"]
    #tau_scalers = model["tau_scalers"]

    segments_list, arrival_times_list = segment_trajectories(all_trajectories[31:32], level=0, f1_gp=f1_gp)
    #hlp.plot_coordinates(all_trajectories[31:32])
    #hlp.plot_speed_time(all_trajectories[31:32][0][1], filter_speed=-1, file_id="bug_31")

    # for j, segments in enumerate(segments_list):
    #     for i, segment in enumerate(segments):
    #         hlp.plot_speed_stops(segment, filter_speed=-1, file_name="speed_stops/{}/segment_{}".format(31, i), f1_gp=f1_gp, scaler=scaler)


    logger.info("Creating test data...")

    trajectory_results = []
    for j, (segments, arrival_times) in enumerate(zip(segments_list, arrival_times_list)):
        segment_results = []
        for i, (segment, arrival_time) in enumerate(zip(segments, arrival_times)):
            if i != 18: continue
            #speeds = np.vstack([e["speed"] for e in segment])
            #speeds = speed_scaler.transform(speeds)
            #pos = np.vstack([e["gps"][::-1] for e in segment])
            #pos = scaler.transform(pos)
            #means, _vars = f1_gp.predict_y(pos)
            #taus = tau_scalers[i].transform(means)
            #feature = np.vstack([[tau[0], speed[0]] for tau, speed in zip(taus, speeds)])
            feature = np.vstack([event_to_tau(e, scaler, f1_gp) for e in segment])
            truth = np.vstack([(arrival_time - e["date"]).total_seconds() for e in segment])
            pprint(feature)
            print(feature.shape)
            result = defaultdict(list)
            result["true"] = truth
            for kernel in kernels:
                for gp_i, gp in enumerate(segment_GPs[i][kernel]):
                    mean, var = gp.predict_y(feature)
                    result["pred"].append([mean, var, kernel])
                    print(gp)
                    abs_errors = [abs(t - m) for t, m in zip(truth, mean)]
                    mae = mean_absolute_error(truth, mean)
                    if mae > 50:
                        logger.warn("{}, {}: {}".format(j, i, mae))
                    metrics = {
                        "mae": mae,
                        "mse": mean_squared_error(truth, mean),
                        "median_abs_err": median_absolute_error(truth, mean),
                        "max_err": max(abs_errors),
                        "min_err": min(abs_errors)
                    }
                    result["metrics"].append(metrics)
                    pprint(list(zip(mean, truth)))
                    logger.info(metrics)

                    xx = np.linspace(min(feature), max(feature), 100)[:,None]
                    mu, sigma = gp.predict_y(xx)
                    plt.figure()
                    plt.plot(xx, mu, "C1", lw=2)

                    plt.plot(feature, mean, "2", mew=2)
                    #plt.fill_between(feature[:,0], mean[:,0] -  2*np.sqrt(var[:,0]), mean[:,0] +  2*np.sqrt(var[:,0]), color="C0", alpha=0.2)

                    plt.fill_between(xx[:,0], mu[:,0] -  2*np.sqrt(sigma[:,0]), mu[:,0] +  2*np.sqrt(sigma[:,0]), color="C1", alpha=0.2)
                    plt.plot(feature, truth, 'kx', mew=2)
                    plt.savefig("{}test_gp_{}_{}_{}.png".format(BASE_DIR, gp_i, i, kernel))

            segment_results.append(result)

        trajectory_results.append(segment_results)
    hlp.save_array(trajectory_results, "trajectory_results", logger, BASE_DIR)
    #print(trajectory_results)

    # vehicle_id, journey = all_trajectories[5]
    # filtered_events = list(filter(lambda e: filter_func(e, level=0, only_pos=True), journey.route))
    # features = defaultdict(list)

    # for event in filtered_events:
    #     tau = f1_gp.predict_y(scaler.transform([event["gps"][::-1]]))[0].reshape(-1, 1)
    #     v = speed_scaler.transform(np.array([event["speed"]]).reshape(-1, 1))[0][0]
    #     for j, tau_scaler in tau_scalers.items():
    #         feature = np.array([tau_scaler.transform(tau)[0][0], v]).reshape(1, -1)
    #         print(feature)
    #         features[j].append(feature)
    #         for gp_i, gp in enumerate(segment_GPs[j]):
    #             print(j, gp_i, gp.predict_y(feature))
    #     exit(1)

    logger.info("%s, %s", model["segment_GPs"].keys(), len(model["segment_GPs"].keys()))
    for key, values in model["segment_GPs"].items():
        logger.info("%s", len(values))


def load_forecasting_model(kernels=None):
    loader = gpflow.saver.Saver()

    if kernels is None:
        kernels = [f for f in os.listdir(BASE_DIR + "segment_GPs") if not "." in f]

    segment_GPs = defaultdict(lambda: defaultdict(list))
    for kernel in kernels:
        models = sorted([f.split("_")[1:] for f in os.listdir("{}segment_GPs/{}/".format(BASE_DIR, kernel)) if "model_" in f], key=lambda x: (int(x[0]), int(x[1])))
        for j, i in models:
            if int(j) > 0: break
            logger.info("Loading GPs for trajectory #%s, %s", j, i)
            path = BASE_DIR + "segment_GPs/{}/model_{}_{}".format(kernel, j, i)
            gp = loader.load(path)
            segment_GPs[int(i)][kernel].append(gp)

    return {
        "segment_GPs": segment_GPs,
        "f1_gp": loader.load(BASE_DIR + "f1_gp"),
        "f1_scaler": hlp.load_array("f1_scaler", logger, BASE_DIR),
        #"speed_scaler": hlp.load_array("speed_scaler", logger, BASE_DIR),
        #"tau_scalers": hlp.load_array("tau_scalers", logger, BASE_DIR)
    }


def get_all_trajectories(trajectories, trajectory_key):
    trajectory_start, trajectory_end = trajectory_key.split(":")
    all_trajectories = trajectories[trajectory_key]
    
    for key, values in trajectories.items():
        k_start, k_end = key.split(":")
        if trajectory_start == k_start and trajectory_end != k_end:
            for vehicle_id, journey in values:
                segmented_journey, _ = journey.segment_at(trajectory_end)
                if segmented_journey is None:
                    logger.info("Journey not segmented when it should have been: %s, %s", trajectory_start, trajectory_end)               
                all_trajectories.append((vehicle_id, segmented_journey))
    return all_trajectories


def create_forecasting_model(all_trajectories, model_trajectory, kernels=None):
    vehicle_id, journey = model_trajectory
    saver = gpflow.saver.Saver()
    f1_gp, scaler = hlp.create_f1_GP(journey.route, logger)

    #saver.save(BASE_DIR + "f1_gp", f1_gp)
    #hlp.save_array(scaler, "f1_scaler", logger, BASE_DIR)

    #tau_scalers = {}
    #speeds = []
    #segment_taus = defaultdict(list)
    segments_list, arrival_times_list = segment_trajectories(all_trajectories, level=0)

    # logger.info("Creating scaler training data...")
    # for j, (segments, arrival_times) in enumerate(zip(segments_list, arrival_times_list)):
    #     for i, (segment, arrival_time) in enumerate(zip(segments, arrival_times)):
    #         #speeds.extend([e["speed"] for e in segment])
    #         pos = np.vstack([e["gps"][::-1] for e in segment])
    #         pos = scaler.transform(pos)
    #         means, _vars = f1_gp.predict_y(pos)
    #         means = means[:,0]
    #         segment_taus[i].extend(means)
    # speeds = np.array(speeds).reshape(-1, 1)
    # logger.info("Training scalers...")    
    # for segment_i, taus in enumerate(segment_taus):
    #     tau_scalers[segment_i] = StandardScaler().fit(taus)
    # speed_scaler = StandardScaler().fit(speeds)
    
    #hlp.save_array(speed_scaler, "speed_scaler", logger, BASE_DIR)
    #hlp.save_array(tau_scalers, "tau_scalers", logger, BASE_DIR)

    logger.info("Training GPs...")
    for j, (segments, arrival_times) in enumerate(zip(segments_list, arrival_times_list)):
        #if j < 16: continue
        if j != 34: continue
        for i, (segment, arrival_time) in enumerate(zip(segments, arrival_times)):
            #if j == 16 and i <= 18: continue
            if i != 3: continue
            pprint(segment)
            pprint(arrival_time)
            print(arrival_time)
            X = np.vstack([event_to_tau(e, scaler, f1_gp) for e in segment])
            Y = np.vstack([(arrival_time - e["date"]).total_seconds() for e in segment])

            print(X)
            print(Y)
            # for event in segment:
            #     pos = scaler.transform([event["gps"][::-1]]) # (lat, lng) -> (lng, lat)
            #     mean, _var = f1_gp.predict_y(pos)
            #     X.append(mean)
            #     #X.append([tau_scalers[i].transform(mean)[:,0][0], speed_scaler.transform(event["speed"])[0][0]])
            #     Y.append(().total_seconds())
            if kernels is None:
                try:
                    gp = train_arrival_time_gp(X, Y, "{},{}".format(j, i))
                except Exception:
                    logger.warn("{}, {} chomsky".format(j, i))
                    continue
                
                saver.save(BASE_DIR + "segment_GPs/{}/model_{}_{}".format("white", j, i), gp)
            else:
                for kernel in kernels:
                    gp = train_arrival_time_gp(X, Y, "{},{}".format(j, i), kernel=kernel)
                    xx = np.linspace(min(X), max(X), 100)[:,None]
                    mu, var = gp.predict_y(xx)
                    plt.figure()
                    plt.plot(xx, mu, "C1", lw=2)
                    plt.fill_between(xx[:,0], mu[:,0] -  2*np.sqrt(var[:,0]), mu[:,0] +  2*np.sqrt(var[:,0]), color="C1", alpha=0.2)
                    plt.plot(X, Y, 'kx', mew=2)
                    plt.savefig("{}gp_{}_{}_{}.png".format(BASE_DIR, j, i, kernel))
                    saver.save(BASE_DIR + "segment_GPs/{}/model_{}_{}".format(kernel, j, i), gp)


def event_to_tau(event, scaler, f1_gp):
    gps = [event["gps"][::-1]] # (lat, lng) -> (lng, lat)
    gps_scaled = scaler.transform(gps)
    mean, _var = f1_gp.predict_y(gps_scaled)
    return mean[0]


def train_tau_segment_gp(segment_taus):
    logger.info("Training tau->segment GP...")
    X = []
    Y = []
    for segment_i, taus in segment_taus.items():
        X.extend(taus)
        Y.extend([float(segment_i)] * len(taus))
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y).reshape(-1, 1)
    print(X)
    print(Y)
    print(X.shape, Y.shape)
    m = gpflow.models.GPR(
        X, Y, 
        kern=gpflow.kernels.Matern32(1))
    gpflow.train.ScipyOptimizer().minimize(m)
    print(m)
    logger.info("tau-segment GP done!")
    return m


def train_arrival_time_gp(X, Y, number=None, kernel=None):
    """GP which maps (%, v) -> arrival_time."""
    with gpflow.defer_build():
        if kernel is None or kernel == "white":
            kern = gpflow.kernels.White(input_dim=1)
        elif kernel == "matern32":
            kern = gpflow.kernels.Matern32(input_dim=1)
        elif kernel == "linear":
            kern = gpflow.kernels.Linear(input_dim=1)
        elif kernel == "rbf":
            kern = gpflow.kernels.RBF(input_dim=1)
        elif kernel == "rbf_linear":
            kern = gpflow.kernels.RBF(input_dim=1, lengthscales=0.1) + gpflow.kernels.Linear(input_dim=1, variance=500)
        else:
            raise Exception("Kernel {} unknown!".format(kernel))
        
        print(X, Y, kern)
        m = gpflow.models.GPR(X, Y, kern=kern)
        m.likelihood.variance = 1#10
        m.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m, disp=True)
        print(m)
        logger.info("f2 GP #{} trained.".format("" if number is None else number))
    return m

def filter_func(event, level, only_pos=False):
    if event["event.type"] == "ObservedPositionEvent":
        return event["speed"] > 0.1
    if only_pos:
        return False
    if level == 0:
        return event["event.type"] in ["PassedEvent", "ArrivedEvent", "DepartedEvent"]
    if level == 1:
        pass  # TODO: Implement
    return True


def segment_trajectories(trajectories, level=0, f1_gp=None):
    """Segment trajectories with trajectory_key at segmentation level "level".
    
    level: the level of segmentation to perform,
        0: Only segment bus stops;
        1: Segment bus stops and red lights (TODO: not yet implemented!);
        2: Segment all stops;
    """
    segments_list = []
    arrival_times_list = []
    for trajectory_i, (vehicle_id, journey) in enumerate(trajectories):
        filtered_events = list(filter(lambda e: filter_func(e, level), journey.route))
        
        # Find segment starts:
        segment_starts = []
        compressed_events = []
        stop_delay = timedelta(seconds=0)
        start_stop_time = None
        for event in filtered_events:
            if event["event.type"] == "ObservedPositionEvent":
                if start_stop_time is not None:
                    #print("New End Stop Time")
                    end_stop_time = event["date"]
                    stop_delay += (end_stop_time - start_stop_time)
                    start_stop_time = None
                event["date"] -= stop_delay
                #print(event["date"])
                compressed_events.append(event)
            else:
                index = len(compressed_events)
                stop = (event["date"] - stop_delay, event["event.type"])

                if is_stop_added(segment_starts, event, index):
                    segment_starts[-1][-1].append(stop)
                    segment_starts[-1][0][-1] = index
                else:
                    #print("New Stop")
                    start_stop_time = event["date"]
                    name = event["stop.name"] if "stop.name" in event else "Stop"
                    segment_starts.append(([index, index], name, [stop]))
        
        print()
        print(trajectory_i)
        pprint(segment_starts)
        print("segment_starts:", len(segment_starts))
        #hlp.plot_speed_stops(journey.route, filter_speed=-1, file_name="31_bug", f1_gp=f1_gp)
        #hlp.plot_speed_time(journey)

        # Do segmentation:
        segments = []
        segment_overlap = 5
        start = 0
        for k, (indices, *_rest) in enumerate(segment_starts): #  We don't care about slice [2:] of the tuple (_types)
            if k == 0:
                start = indices[-1]
                continue

            next_start = indices[-1]
            # TODO: Make sure this works even if a route does not end with speed = 0.
            if next_start - start >= 4:
                if not segments:
                    overlapped_start = start
                else:
                    overlapped_start = start - segment_overlap
                overlapped_stop = next_start + segment_overlap
                print(overlapped_start, next_start, overlapped_stop)
                segments.append(compressed_events[overlapped_start:overlapped_stop])
            start = next_start

        if segments: segments_list.append(segments)

        arrival_times = []
        for i, (_start, name, stop_events) in enumerate(segment_starts):
            if i > 0:
                arrival_times.append(stop_events[0][0])

        # starting_time = segments[0][0]["date"]
        # for i, (_start, name, stop_events) in enumerate(segment_starts):
        #     arrival_date = stop_events[0][0]
        #     departure_date = stop_events[-1][0]
        #     if i > 0:
        #         diff = (arrival_date - starting_time[-1]).total_seconds()
        #         if diff < 0:
        #             diff = (arrival_date -starting_time[0]).total_seconds()
        #         arrival_times.append((diff, name))
        #     starting_time = (arrival_date, departure_date)
        #pprint(arrival_times)
        arrival_times_list.append(arrival_times)
    assert len(segments_list) == len(arrival_times_list)
    return segments_list, arrival_times_list


def is_stop_added(segment_starts, event, index):
    if not segment_starts:
        return False

    if segment_starts[-1][0][-1] == index:
        return True

    if "stop.name" in event and segment_starts[-1][1] == event["stop.name"]:
        return True
    
    return False




if __name__ == "__main__":
    logger = setup_logging("forecasting.py", "forecasting.log")
    BASE_DIR = "forecasting/"
    main(sys.argv[1:])