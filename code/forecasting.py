import pandas
import pickle
import sys, getopt
import warnings
import logging
import gpflow
import os
import math
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
from scipy.stats import norm

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

def main(argv):
    line_number = None
    load_model = False
    load_result = False
    load_dist = False
    train_f1 = False

    help_line = 'usage: forecasting.py -l <line_number> --train-f1 --load-model --load-result --load-dist'
    try:
        opts, args = getopt.getopt(argv,"hl:",["line=", "train-f1", "load-model", "load-result", "load-dist"])
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
        elif opt == "--load-result":
            load_result = True
        elif opt == "--load-dist":
            load_dist = True
        elif opt == "--train-f1":
            train_f1 = True

    if train_f1:
        run_train_f1(line_number)
    elif load_dist:
        run_plot_dist()
    elif load_result:
        run_load_result()
    else:
        run(line_number, load_model)
    

def run_plot_dist(trajectory_results=None):
    if trajectory_results is None:
        distributions = hlp.load_array("trajectory_distributions", logger, BASE_DIR)
    else:
        distributions = create_trajectory_distributions(trajectory_results)
    plot_arrival_time_distributions(distributions)


def run_load_result():
    trajectory_results = hlp.load_array("trajectory_results", logger, BASE_DIR)
    run_plot_dist(trajectory_results)


def run_train_f1(line_number):
    session = gpflow.saver.Saver()
    trajectories = hlp.load_trajectories_from_file(line_number, logger)
    
    trajectory_key = "Lötgatan:Linköpings resecentrum"
    kernels = ["rbf_linear"]
    all_trajectories = hlp.get_all_trajectories(trajectories, trajectory_key)
    # segments_list, arrival_times_list = segment_trajectories(all_trajectories[3:8], level=0)
    # for segment in [3, 4, 5, 6]:
    #     hlp.plot_speeds([segments[segment] for segments in segments_list], filter_speed=-1, file_id=segment)
    # exit()
    logging.info("Test GP f1...")
    F1_DIR = "f1_test/"
    vehicle_id, journey = all_trajectories[len(all_trajectories) //2]
    model = hlp.create_f1_GP_revamped(journey.route, session, logger, F1_DIR, load_model=False, version="_ard")
    
    f1_gp = model["f1_gp"]
    f1_scaler = model["f1_scaler"]
    X = model["X"]
    Y = model["Y"]
    
    X_test = []
    for vehicle_id, journey in all_trajectories[1:]:
        X_test.append(np.vstack([e["gps"][::-1] for e in journey.route if e["event.type"] == "ObservedPositionEvent" and e["speed"] > 0.1]))
    X_test = np.array(X_test)

    hlp.visualise_f1_gp_revamped(X, Y, f1_gp, f1_scaler, X_test, file_name="f1_gp_ARD", base_dir=F1_DIR)


def run(line_number, load_model):
    logger.info("Starting execution of forecasting.py!")
    session = gpflow.saver.Saver()

    trajectories = hlp.load_trajectories_from_file(line_number, logger)
    trajectory_key = "Lötgatan:Linköpings resecentrum"
    all_trajectories = hlp.get_all_trajectories(trajectories, trajectory_key)

    if load_model:
        model = load_forecasting_model(session)
    else:
        create_forecasting_model(all_trajectories, session)
        exit(0)

    test_ids = list(set(np.random.randint(37, size=10)))
    logger.info(test_ids)
    logger.info("Test models...")
    trajectory_results = test_model(test_ids, all_trajectories, model, logger)    

    run_plot_dist(trajectory_results)


def create_trajectory_distributions(trajectory_results):
    logger.info("Create trajectory distributions...")
    logger.info("Test trajectory IDs: {}".format(trajectory_results.keys()))
    trajectory_distributions = defaultdict(object)
    for test_id, test_trajectory_result in trajectory_results.items():
        logger.info("Segments: {}".format(len(test_trajectory_result)))
        segment_distributions = []
        for segment_i, segment_result in enumerate(test_trajectory_result):
            logger.info("Segment {} tested on {} GPs".format(segment_i, len(segment_result["pred"])))
            # segment_result elements have 2 keys: 
            # pred: Arrival time predictions (means, vars)
            # true: True arrival times.
            # metrics: Metrics for the prediction.

            # TODO: Plot ATP distribution for segment_i.
            # How? Well, we create PDF distributions with the (mean, var)-pairs 
            # we get from the results. We can then plot the PDFs, as they are 1D on input.
            # We also need to show the true arival time, so we can get a grasp of how far "off" we are.
            y_true = segment_result["true"]
            feature_fitted = segment_result["feature"]
            distribution = defaultdict(list)
            #print(segment_result["feature_fitted"])
            #for mu, var, feature_fitted in zip(segment_result["pred"], segment_result["feature_fitted"]):
            #    for point_i, (mu_i, std_i, feature_i) in enumerate(zip(mu, np.sqrt(var), feature_fitted)):
            for mu, var in segment_result["pred"]:
                for point_i, (mu_i, std_i) in enumerate(zip(mu, np.sqrt(var))):
                    distribution[point_i].append(norm(mu_i, std_i))#, feature_i])
            segment_distributions.append((distribution, y_true, feature_fitted))
        trajectory_distributions[test_id] = segment_distributions
    hlp.save_array(trajectory_distributions, "trajectory_distributions", logger, BASE_DIR)
    return trajectory_distributions


def plot_arrival_time_distributions(trajectory_distributions):
    logger.info("Plot Arrival Time Prediction Distributions...")
    path = BASE_DIR + "arrival_time_distributions"
    hlp.ensure_dir(path)
    precision = 1e-03
    for trajectory_i, segment_distributions in trajectory_distributions.items():
        logger.info("Trajectory {} has {} segment(s).".format(trajectory_i, len(segment_distributions)))
        traj_path = path + "/{}".format(trajectory_i)
        hlp.ensure_dir(traj_path)
        for segment_i, (point_distribution, truth, feature) in enumerate(segment_distributions):
            logger.info("Plotting Segment {}...".format(segment_i))
            seg_path = traj_path + "/{}".format(segment_i)
            hlp.ensure_dir(seg_path)
            arrival_time_naive = []
            for point_i, distribution in point_distribution.items():
            #temp    plt.figure()
            #temp    plt.axvline(x=truth[point_i], color="r", linestyle='-', lw=0.5)
            #temp    min_t = 1000
            #temp    max_t = -1000
            #temp    sum_of_weights = 0
            #temp    for k in distribution:
            #temp        min_t = min(min_t, math.floor(k.ppf(precision)[0]))
            #temp        max_t = max(max_t, math.ceil(k.ppf(1-precision)[0]))
                    #sum_of_weights += calculate_weight(feature, k)
            #temp    pdf_res = np.zeros((max_t-min_t)*100)
            #temp    pdf_res_weighted = np.zeros((max_t-min_t)*100)
            #temp    xx = np.linspace(min_t, max_t, (max_t-min_t)*100)
                distr_mean = 0
                for k in distribution:
                    # TODO: Implement different things here.
            #temp        pdf = k.pdf(xx)
                    distr_mean += k.mean()
            #temp        np.add(pdf_res, pdf, pdf_res) # naive mixture
                    #weight = calculate_weight(feature, k)
                    #np.add(pdf_res_weighted, pdf * (weight - sum_of_weights), pdf_res_weighted) # weighted mixture 
            #temp        plt.plot(xx, pdf)
                distr_mean = distr_mean / len(distribution)
                arrival_time_naive.append(distr_mean)
            #temp    plt.savefig("{}/{}".format(seg_path, point_i))
            #temp    plt.close()
            #temp    pdf_res = pdf_res / len(distribution)
                #pdf_res_weighted = pdf_res_weighted / sum_of_weights

                # naive mixture
            #temp    plt.figure()
            #temp    plt.axvline(x=distr_mean, color="k", linestyle="--", lw=0.5)
            #temp    plt.axvline(x=truth[point_i], color="r", linestyle='-', lw=0.5)
            #temp    plt.plot(xx, pdf_res)
            #temp    plt.savefig("{}/res_{}".format(seg_path, point_i))
            #temp    plt.close()

                # weighted mixture TODO: No idea if this works
                #plt.figure()
                #plt.axvline(x=distr_mean, color="k", linestyle="--", lw=0.5)
                #plt.axvline(x=truth[point_i], color="r", linestyle='-', lw=0.5)
                #plt.plot(xx, pdf_res_weighted)
                #plt.savefig("{}/weighted_res_{}".format(seg_path, point_i))
                #plt.close()
            # abs_errors = [abs(t - m) for t, m in zip(truth, arrival_time_naive)]
            # mae = mean_absolute_error(truth, arrival_time_naive)
            # mse = mean_squared_error(truth, arrival_time_naive)
            # metrics = {
            #     "mae": mae,
            #     "mse": mse,
            #     "rmse": np.sqrt(mse),
            #     "median_abs_err": median_absolute_error(truth, arrival_time_naive),
            #     "max_err": max(abs_errors),
            #     "min_err": min(abs_errors)
            # }
            # logger.info(metrics)
            hlp.save_array({"truth": truth, "predicted": arrival_time_naive, "feature": feature}, "{}/predicted".format(seg_path))
            # np.savetxt("{}/metrics.txt".format(seg_path), [
            #     metrics["mae"], 
            #     metrics["mse"], 
            #     metrics["rmse"],
            #     metrics["median_abs_err"],
            #     metrics["max_err"],
            #     metrics["min_err"]])


def calculate_weight(feature, M_k):
    # TODO: No idea if this works.
    return ((-1/2) * (feature - M_k.mean()) * np.power(M_k.var(), -1) * (feature - M_k.mean())) - ((1/2) * np.log(M_k.var()))


    
def test_model(test_ids, all_trajectories, model, logger):
    f1_gp = model["f1_gp"]
    f1_scaler = model["f1_scaler"]
    segment_GPs = model["segment_GPs"]
    segment_scalers = model["segment_scalers"]

    test_trajectories = []
    for test_id in test_ids:
        test_trajectories.append(all_trajectories[test_id])
    segments_list, arrival_times_list = segment_trajectories(test_trajectories, level=0, f1_gp=f1_gp)

    trajectory_results = defaultdict(object)
    for traj_j, (segments, arrival_times) in enumerate(zip(segments_list, arrival_times_list)):
        logger.info("Testing model {}".format(traj_j))
        segment_results = []
        for seg_i, (segment, arrival_time) in enumerate(zip(segments, arrival_times)):
            #if i > 1: continue
    
            feature = np.vstack([event_to_tau(e, f1_scaler, f1_gp) for e in segment])
            truth = np.vstack([(arrival_time - e["date"]).total_seconds() for e in segment])

            result = defaultdict(list)
            result["true"] = truth
            result["feature"] = feature
            for gp_k, gp in enumerate(segment_GPs[seg_i]):
                if gp_k == test_ids[traj_j]: # GP is trained on this data
                    continue
                
                feature_fitted = segment_scalers[seg_i][gp_k].transform(feature)
                mean, var = gp.predict_y(feature_fitted)
                result["pred"].append([mean, var])
                #abs_errors = [abs(t - m) for t, m in zip(truth, mean)]
                #mae = mean_absolute_error(truth, mean)
                #if mae > 50:
                #    logger.warn("{}, {}: {}".format(traj_j, seg_i, mae))
                #mse = mean_squared_error(truth, mean)
                #metrics = {
                #    "mae": mae,
                #    "mse": mse,
                #    "rmse": np.sqrt(mse),
                #    "median_abs_err": median_absolute_error(truth, mean),
                #    "max_err": max(abs_errors),
                #    "min_err": min(abs_errors)
                #}
                #result["metrics"].append(metrics)
                #logger.info("Segment {}, GP {} metrics: {}".format(seg_i, gp_k, metrics))

            segment_results.append(result)

        trajectory_results[test_ids[traj_j]] = segment_results
    hlp.save_array(trajectory_results, "trajectory_results", logger, BASE_DIR)
    return trajectory_results


def load_forecasting_model(session):
    logger.info("Loading good f1 model...")
    f1_gp, f1_scaler = hlp.load_f1_GP_revamped(session, logger, "f1_test/", version="_ard")

    segment_GPs = defaultdict(list)
    segment_scalers = defaultdict(list)
    models = sorted([f.split("_")[1:] for f in os.listdir(BASE_DIR + "GP/") if "model_" in f], key=lambda x: (int(x[0]), int(x[1])))
    for j, i in models:
        #if int(j) > 3: break
        logger.info("Loading GPs for trajectory #%s, %s", j, i)
        gp = session.load(BASE_DIR + "GP/model_{}_{}".format(j, i))
        scaler = hlp.load_array("GP/scaler_{}_{}".format(j, i), logger, BASE_DIR)
        segment_i = int(i)
        segment_GPs[segment_i].append(gp)
        segment_scalers[segment_i].append(scaler)

    return {
        "f1_gp": f1_gp,
        "f1_scaler": f1_scaler,
        "segment_GPs": segment_GPs,
        "segment_scalers": segment_scalers
    }


def create_forecasting_model(all_trajectories, session):
    logger.info("Loading good f1 model...")
    f1_gp, f1_scaler = hlp.load_f1_GP_revamped(session, logger, "f1_test/", version="_ard")

    logger.info("Creating Segments...")
    segments_list, arrival_times_list = segment_trajectories(all_trajectories, level=0)

    logger.info("Training GPs...")
    for j, (segments, arrival_times) in enumerate(zip(segments_list, arrival_times_list)):
        if j != 10 and j != 22: continue
        #if j != 34: continue
        for i, (segment, arrival_time) in enumerate(zip(segments, arrival_times)):
            #if j == 16 and i <= 18: continue
            #if i != 13 and i != 6: continue
            #pprint(segment)
            #pprint(arrival_time)
            #print(arrival_time)
            X = np.vstack([event_to_tau(e, f1_scaler, f1_gp) for e in segment])
            Y = np.vstack([(arrival_time - e["date"]).total_seconds() for e in segment])
            segment_scaler = StandardScaler().fit(X)
            X_fitted = segment_scaler.transform(X)
            #hlp.save_array(segment_scaler, "GP/scaler_{}_{}".format(j, i), logger, BASE_DIR)
            
            for kernel in ["rbf", "rbf_linear"]:
                gp = train_arrival_time_gp(X_fitted, Y, number="{},{}".format(j, i), kernel=kernel)
                #session.save(BASE_DIR + "GP/model_{}_{}".format(j, i), gp)
            
                # Visualise:
                xx = np.linspace(min(X_fitted), max(X_fitted), 100)[:,None]
                mu, var = gp.predict_y(xx)
                plt.figure()
                plt.plot(xx, mu, "C1", lw=2)
                plt.fill_between(xx[:,0], mu[:,0] -  2*np.sqrt(var[:,0]), mu[:,0] +  2*np.sqrt(var[:,0]), color="C1", alpha=0.2)
                plt.plot(X_fitted, Y, 'kx', mew=2)
                plt.savefig("{}gp_vis_report/{}/{}_{}.png".format(BASE_DIR, kernel, j, i))

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


def train_arrival_time_gp(X, Y, number, kernel="rbf_linear"):
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
        
        m = gpflow.models.GPR(X, Y, kern=kern)
        m.likelihood.variance = 1#10
        m.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m)
        logger.info("Arrival Time Pred GP #{} trained.".format(number))
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
        
        #print()
        #print(trajectory_i)
        #pprint(segment_starts)
        #print("segment_starts:", len(segment_starts))
        #hlp.plot_speed_stops(journey.route, filter_speed=-1, file_name="31_bug", f1_gp=f1_gp)
        #hlp.plot_speed_time(journey)

        # Do segmentation:
        segments = []
        segment_overlap = 0
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
                #print(overlapped_start, next_start, overlapped_stop)
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