import pandas
import pickle
import sys, getopt
import logging
import gpflow
import numpy as np
import ml_helper as hlp
import matplotlib as mpl
import dateutil.parser
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import pprint
import os
import seaborn as sns
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from helpers import setup_logging, print_progress_bar
from collections import defaultdict
from gmplot import gmplot
from operator import add
from scipy.stats import multivariate_normal, norm
from scipy.integrate import quad
from matplotlib.colors import ListedColormap

def main(argv):
    line_number = None
    load_model = False
    load_heatmap = False
    help_line = 'usage: gps_var.py -l <line_number> --load-model --load-heatmap'
    try:
        opts, args = getopt.getopt(argv,"hl:",["line=", "load-model", "load-heatmap"])
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
        elif opt == "--load-heatmap":
            load_heatmap = True

    logger.info("Starting execution of gps_var.py!")

    if not load_heatmap:
        run(line_number, load_model)
    else:
        #heatmap = hlp.load_array("heatmap_7500x7500", logger)
        for j in range(6):
            heatmap = hlp.load_array("combined_gp_heat_focused/{}_1000x1000".format(j), logger)
            plot_heatmap(heatmap, identifier="_focused/{}_1000x1000".format(j), show_title=False, with_alpha=True)


def plot_heatmap(heatmap, identifier="", show_title=True, with_alpha=False):
    cmap = plt.get_cmap("magma")
    my_cmap = cmap(np.arange(cmap.N))
    alphas = np.ones(cmap.N)
    alpha_range = len(alphas)//4
    if with_alpha: alphas[:alpha_range] = np.linspace(0, 1, alpha_range)
    my_cmap[:,-1] = alphas
    my_cmap = ListedColormap(my_cmap)
    img = np.flip(heatmap, 0)
    plt.figure()
    sns.heatmap(img, cmap=my_cmap, cbar=False, xticklabels=False, yticklabels=False)
    if show_title: plt.title("Combined GP Heatmap")
    plt.savefig("combined_gp_heat{}.png".format(identifier))
    plt.clf()
    plt.close()
    
    
def run(line_number, load_model):
    visualise_trajectory_gp = False # Quicker if False
    visualise_tau_gp = False # Quicker if False

    session = gpflow.saver.Saver()

    trajectories = hlp.load_trajectories_from_file(line_number, logger)
    #hlp.plot_trajectories(trajectories, logger, print_only=True)
    
    trajectory_key = "Lötgatan:Linköpings resecentrum"
    #vehicle_id, journey = trajectories[trajectory_key][0]
    #pprint.pprint(journey.route)
    #hlp.plot_speed_time(trajectories["Lötgatan:Fönvindsvägen östra"][4][1].segment_at("Linköpings resecentrum")[0])
    #hlp.plot_speed_stops(journey.route)

    #plot_speed_stops(journey.route)
    #plot_speed_time(journey)

    all_trajectories = hlp.get_all_trajectories(trajectories, trajectory_key)

    if load_model:
        model = load_GPS_variation_GPs(session, load_only="all", f1_version="_ard")
        output_file = "f1_ard_contour_segment3_pdf"
    else:
        create_GPS_variation_GPs(all_trajectories, session, f1_version="_ard")
        exit(0)
        
    f1_gp = model["f1_gp"]
    f1_scaler = model["f1_scaler"]
    f2_f3_GPs = model["f2_f3_GPs"]
    f2_f3_scalers = model["f2_f3_scalers"]

    # for i, (vehicle_id, journey) in enumerate(all_trajectories[1:]):
    #     events = [e for e in journey.route if e["event.type"] == "ObservedPositionEvent"]
    #     X = [[e["date"].isoformat(), e["gps"][::-1][0], e["gps"][::-1][-1], e["speed"], e["dir"]] for e in events]
    #     np.savetxt("d{}.txt".format(i + 2), X, fmt='%s', delimiter=";")

    # exit(1)


    res = 50
    #lat_step = get_step_size(lat_start, lat_stop, res)
    #lng_step = get_step_size(lng_start, lng_stop, res)

    #lat, lng = np.mgrid[58.410317:58.427006:res*1j, 15.490352:15.523200:res*1j] # Big Grid
    #lat, lng = np.mgrid[58.416317:58.42256:res*1j, 15.494352:15.503200:res*1j] # Middle Grid
    #lat, lng = np.mgrid[58.4173:58.419:res*1j, 15.4965:15.499:res*1j] # Small Grid
    #lat, lng = np.mgrid[58.4174:58.4178:res*1j, 15.4967:15.4974:res*1j] # Super small Grid (krök)
    #lat, lng = np.mgrid[58.4185:58.4188:res*1j, 15.4985:15.49875:res*1j] # Super small Grid (sträcka)

    #lat, lng = np.mgrid[58.4190:58.422:res*1j, 15.500:15.502:res*1j] # Small Grid (new start)
    #lat, lng = np.mgrid[58.4175:58.422:res*1j, 15.508:15.517:res*1j] # Small Grid (segment 2) 
    lat, lng = np.mgrid[58.408:58.418:res*1j, 15.61:15.63:res*1j] # Small Grid (segment 3, final) 
    
    pos_grid = np.dstack((lng, lat)).reshape(-1, 2)
    logger.info("Grid created.")
    pos_grid_fitted = f1_scaler.transform(pos_grid)
    logger.info("Grid scaled.")
    
    grid_tau, _var = f1_gp.predict_y(pos_grid_fitted)
    logger.info("Grid predicted.")
    hlp.save_array(grid_tau, "grid_tau", logger, BASE_DIR)

    logger.info("Evaluate Grid with GPs...")
    probs = calculate_prob_grip(grid_tau, f2_f3_GPs, f2_f3_scalers, pos_grid_fitted, res, method="pdf")

    pdf_grid_sum = None
    for traj_j, pdf_grid in probs.items():
        visualise_probabilities(pdf_grid, all_trajectories, pos_grid, lat, lng, res, file_name=output_file + "_{}".format(traj_j))
        if pdf_grid_sum is None:
            pdf_grid_sum = pdf_grid
        else:
            np.add(pdf_grid_sum, pdf_grid, pdf_grid_sum)
    pdf_grid_sum /= len(probs.keys())
    visualise_probabilities(pdf_grid_sum, all_trajectories, pos_grid, lat, lng, res, file_name=output_file + "_all")
    exit(1)
    # TODO: Just testing this. #MidnightIdeas
    # combined_f2_means = combine_mean(f2_means)
    # combined_f2_variances = combine_variance(f2_variances, f2_means, combined_f2_means)
    # combined_f3_means = combine_mean(f3_means)
    # combined_f3_variances = combine_variance(f3_variances, f3_means, combined_f3_means)


    # kernels = []
    # logger.info("Creating kernels.")
    # for i in range(len(combined_f2_means)):
    #     # We want (lng, lat) coords for this kernel
    #     k = multivariate_normal(
    #         mean=[combined_f3_means[i], combined_f2_means[i]],
    #         cov=np.diag([combined_f3_variances[i], combined_f2_variances[i]])
    #     )
    #     kernels.append(k)
    # logger.info("Kernels created.")

    # res = 5000
    # x, y = np.mgrid[-1.2845:-1.2305:res*1j, 0.445:0.695:res*1j]#np.mgrid[-1.27:-1.25:res*1j, 0.59:0.61:res*1j]
    # pos = np.dstack((x, y))
    # logger.info("Grid Created.")
    # #x = np.linspace(-1.5, 1.75, num=res)
    # #y = np.linspace(-1.75, 2, num=res)
    # plt.figure()
    # for i, k in enumerate(kernels[:10]):
    #     if i == 2:
    #         break
    #     print(i)
    #     z = np.ma.masked_less(k.pdf(pos), 0.1)

    #     print(z)
    #     print(np.max(z))
    #     plt.contourf(x, y, z)
    # plt.title("Contour 1 Kernel GP")
    # plt.savefig("contour_test.png")
    # plt.clf()
    # plt.close()
    # return

    
    # xxyy = np.c_[xx.ravel(), yy.ravel()]
    # logger.info("%sx%s Grid created.", res, res)
    # zz = sum([k.pdf(xxyy) for k in kernels])
    # logger.info("Probabilities for grid calculated.")

    # img = np.zz.reshape((res, res))
    # print(img)
    # plt.figure()
    # plt.imshow(img)
    # plt.title("1 Kernel GP")
    # plt.savefig("heatmap_test.png")
    # plt.clf()
    # plt.close()

    # combined_f2_means = combine_mean(f2_means)
    # combined_f2_variances = combine_variance(f2_variances, f2_means, combined_f2_means)
    # combined_f3_means = combine_mean(f3_means)
    # combined_f3_variances = combine_variance(f3_variances, f3_means, combined_f3_means)

    # visualise_combined_f2_f3_gp(combined_f2_means, combined_f2_variances, combined_f3_means, combined_f3_variances, scaler)
    # visualise_combined_f2_f3_gp_heatmap(combined_f2_means, combined_f2_variances, combined_f3_means, combined_f3_variances)


def visualise_probabilities(probs, all_trajectories, pos_grid, lat, lng, res, file_name):
    logger.info("Visualise probabilities...")
    X_test = []
    for vehicle_id, journey in all_trajectories:
        X_test.append(np.vstack([e["gps"][::-1] for e in journey.route if e["event.type"] == "ObservedPositionEvent" and e["speed"] > 0.1]))
    X_test = np.array(X_test)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("GPS Variance Estimation")

    min_x = min(pos_grid[:,0])
    max_x = max(pos_grid[:,0])
    min_y = min(pos_grid[:,1])
    max_y = max(pos_grid[:,1])
    v_min = min(probs)
    v_max = max(probs)

    tightness = 0
    plt.xlim(min_x - (max_x - min_x)*tightness, max_x + (max_x - min_x)*tightness)
    plt.ylim(min_y - (max_y - min_y)*tightness, max_y + (max_y - min_y)*tightness)
    
    #sns.heatmap(probs.reshape(50, 50))
    probs = probs.reshape(res, res)
    plt.contourf(lng, lat, probs)
    plt.ticklabel_format(useOffset=False)
    for tick in ax.xaxis.get_ticklabels()[::2]:
        tick.set_visible(False)
    plt.colorbar()
    #plt.scatter(pos_grid[:,0], pos_grid[:,1], c=probs, vmin=v_min, vmax=v_max, cmap="cool", s=3, zorder=1)
    for points in X_test:
        plt.scatter(points[:,0], points[:,1], color="r", s=0.2, zorder=3)

    plt.savefig("{}variance_plot/{}.png".format(BASE_DIR, file_name))
    plt.close()


def calculate_prob_grip(grid_tau, f2_f3_GPs, f2_f3_scalers, pos_grid_fitted, res, method=None):
    lng_ss_scaled = get_step_size(pos_grid_fitted[0][0], pos_grid_fitted[-1][0], res)
    lat_ss_scaled = get_step_size(pos_grid_fitted[0][1], pos_grid_fitted[-1][1], res)

    points = grid_tau.shape[0]
    probs = np.zeros(points)
    if method == "combine":
        lng_means = []
        lng_vars = []
        lat_means = []
        lat_vars = []
    elif method == "pdf":
        kernels_list = defaultdict(list)
        probs = {}
    else:
        result = defaultdict(list)
    
    for traj_j, (f2_gp, f3_gp) in f2_f3_GPs.items():
        grid_tau_fitted = f2_f3_scalers[traj_j].transform(grid_tau)
        lng_mean, lng_var = f2_gp.predict_y(grid_tau_fitted)
        lat_mean, lat_var = f3_gp.predict_y(grid_tau_fitted)

        if method == "combine":
            lng_means.append(lng_mean)
            lng_vars.append(lng_var)
            lat_means.append(lat_mean)
            lat_vars.append(lat_var)
        elif method == "pdf":
            probs[traj_j] = np.zeros(points)
            kernels = []
            for m1, v1, m2, v2 in zip(lng_mean, lng_var, lat_mean, lat_var):
                k = multivariate_normal(
                    mean=[m1[0], m2[0]],
                    cov=np.diag([v1[0], v2[0]])
                )
                kernels.append(k)
            kernels_list[traj_j] = kernels
        else:
            result[traj_j].extend((lng_mean, lng_var, lat_mean, lat_var))

    if method == "combine":
        lng_means = np.array(lng_means)
        lng_vars = np.array(lng_vars)
        lat_means = np.array(lat_means)
        lat_vars = np.array(lat_vars)
        c_lng_means = combine_mean(lng_means)
        c_lng_vars = combine_variance(lng_vars, lng_means, c_lng_means)
        c_lat_means = combine_mean(lat_means)
        c_lat_vars = combine_variance(lat_vars, lat_means, c_lat_means)

    traj_count = len(f2_f3_GPs.keys())
    for grid_i, (lng_i, lat_i) in enumerate(pos_grid_fitted):
        if grid_i % 100 == 0:
            logger.info(grid_i)

        if method == "combine":
            lng_prob = prob_of(lng_i, lng_ss_scaled, c_lng_means[grid_i], c_lng_vars[grid_i])
            lat_prob = prob_of(lat_i, lat_ss_scaled, c_lat_means[grid_i], c_lat_vars[grid_i])
            probs[grid_i] += lng_prob * lat_prob  # We assume independent!
        elif method == "pdf":
            for traj_j, kernels in kernels_list.items():
                probs[traj_j][grid_i] = kernels[grid_i].pdf((lng_i, lat_i))
        else:
            for traj_j, (lng_mean, lng_var, lat_mean, lat_var) in result.items():
                lng_prob = prob_of(lng_i, lng_ss_scaled, lng_mean[grid_i], lng_var[grid_i])
                lat_prob = prob_of(lat_i, lat_ss_scaled, lat_mean[grid_i], lat_var[grid_i])
                probs[grid_i] += lng_prob * lat_prob  # We assume independent!
            probs[grid_i] = probs[grid_i] / traj_count
    print(traj_count)
    #probs[probs < 1e-03] = 0
    hlp.save_array(probs, "grid_probs", logger, BASE_DIR)
    return probs


def prob_of(observed_value, step_size, mean, variance):
    distr = norm(mean, np.sqrt(variance))  # standard deviation as parameter
    half_step = step_size
    prob, _ =  quad(distr.pdf, observed_value - half_step, observed_value + half_step)
    print(mean, variance, prob)
    return prob

def get_step_size(start, stop, points):
    return (stop - start) / (points - 1)


def create_GPS_variation_GPs(all_trajectories, session, f1_version):
    """ Function that creates and saves two GPS variation GPs,
    (f2=lat, f3=lng) for each trajectory.
    """
    logger.info("Loading good f1 model...")
    f1_gp, f1_scaler = hlp.load_f1_GP_revamped(session, logger, "f1_test/", version=f1_version)

    logger.info("Creating new f2, and f3 for GPS variation estimation.")
    train_trajectory_GPs(all_trajectories, f1_gp, f1_scaler, session)


def load_GPS_variation_GPs(session, load_only="all", load_specific=None, f1_version=""):
    """ 'load_only' is the number of GPs to load. Default is "all".
    If an int it will only load that number of GPs.
    'load_specific' loads a GP with that specific number."""
    logger.info("Loading GPs f1, f2, and f3 for GPS variation estimation.")
    
    f1_gp = session.load("f1_test/f1_gp{}".format(f1_version))

    path = BASE_DIR + "GP/"
    GPs = defaultdict(list)
    models = sorted([f.split("_")[1:] for f in os.listdir(path) if "model_" in f])

    for trajectory_i, gp_name in models: # gp_name: f2 always before f3
        if load_specific is not None and int(trajectory_i) != load_specific: continue
        if load_only != "all" and int(trajectory_i) >= load_only: break

        logger.info("{} GP #{} loaded.".format(gp_name, trajectory_i))
        gp = session.load(path + "model_{}_{}".format(trajectory_i, gp_name))
        GPs[int(trajectory_i)].append(gp)

    return {
        "f1_gp": f1_gp,
        "f1_scaler": hlp.load_array("f1_test/f1_scaler", logger),
        "f2_f3_GPs": GPs,
        "f2_f3_scalers": hlp.load_array(BASE_DIR + "f2_f3_scalers", logger)
    }


def combine_mean(means):
    """Combine the mean of each model.
    means is a 2D numpy array where a row is the means for one model.
    Each model is trained with 1 trajectory => N_j = 1 for all j.
    We want to thus sum the means column-wise and normalise by the number of models."""
    return np.sum(means, axis=0) / means.shape[0]


def combine_variance(variances, means, combined_means):
    """Combine the variance of each model.
    variances is a 2D numpy array where a row is the variances for one model.
    means is a 2D numpy array where a row is the means for one model.
    combined_mean is a 1D numpy array with the combined mean for all J models.
    Each model is trained with 1 trajectory => N_j = 1 for all j in J."""
    assert variances.shape == means.shape
    return np.subtract(
        np.sum(np.add(variances, np.square(means)), axis=0) / means.shape[0], 
        np.square(combined_means)
    )


def train_trajectory_GPs(all_trajectories, f1_gp, f1_scaler, session):
    f2_f3_scalers = []
    for i, (vehicle_id, journey) in enumerate(all_trajectories):
        events = [e for e in journey.route if e["event.type"] == "ObservedPositionEvent" and e["speed"] > 0.1]
        events = hlp.filter_duplicates(events)

        xx = np.vstack([e["gps"][::-1] for e in events])
        xx_fit = f1_scaler.transform(xx)
        xx_lng = xx_fit[:,0].reshape(-1,1)
        xx_lat = xx_fit[:,1].reshape(-1,1)
        tau_mean, _var = f1_gp.predict_y(xx_fit)
        tau_mean = tau_mean[:,0].reshape(-1,1)

        f2_f3_scaler = StandardScaler().fit(tau_mean)
        f2_f3_scalers.append(f2_f3_scaler)
        tau_mean_fitted = f2_f3_scaler.transform(tau_mean)
        train_GP(tau_mean_fitted, xx_lng, session, number=i, gp_name="f2")
        train_GP(tau_mean_fitted, xx_lat, session, number=i, gp_name="f3")
    hlp.save_array(f2_f3_scalers, "f2_f3_scalers", logger, BASE_DIR)


def new_vis_combined_gp_mean(all_trajectories):
    # TODO: Implement
    X_test = []
    for vehicle_id, journey in all_trajectories[1:]:
        X_test.append(np.vstack([e["gps"][::-1] for e in journey.route if e["event.type"] == "ObservedPositionEvent" and e["speed"] > 0.1]))
    X_test = np.array(X_test)


def visualise_combined_f2_f3_gp(combined_f2_means, combined_f2_variances, combined_f3_means, combined_f3_variances, scaler, splits=None):
    if splits is None:
        plt.figure()
        plt.title("Predictive Mean of Combined GP")
        lng_lats = np.array(list(zip(combined_f2_means, combined_f3_means)))
        lng_lats = scaler.inverse_transform(lng_lats)
        plt.plot(lng_lats[:,0], lng_lats[:,1], color="blue")
        #plt.colorbar()
        plt.savefig("combined_gp.png")
        return

    interval = len(combined_f2_means)//splits
    color = "blue"
    for i in range(splits):
        start = i * interval
        stop = (i + 1) * interval
        plt.figure()
        plt.title("Combined GP ({}:{}".format(start, stop))
        plt.plot(combined_f2_means[start:stop], combined_f3_means[start:stop], color=color)
        plt.plot(combined_f2_means[start:stop] - 2*np.sqrt(combined_f2_variances[start:stop]), combined_f3_means[start:stop] + 2*np.sqrt(combined_f3_variances[start:stop]), '--', color=color) 
        plt.plot(combined_f2_means[start:stop] + 2*np.sqrt(combined_f2_variances[start:stop]), combined_f3_means[start:stop] - 2*np.sqrt(combined_f3_variances[start:stop]), '--', color=color) 
        #plt.colorbar()
        plt.savefig("combined_gp/{}.png".format(i))
        plt.clf()
        plt.close()


def visualise_MoGP_heatmap(f2_means, f2_variances, f3_means, f3_variances):
    """p(x) = 1 / K \sum_k^K N(mu_k(x), sigma^2_k(x)), where x is a 2D grid of Tau:s."""
    pass


def visualise_combined_f2_f3_gp_heatmap(combined_f2_means, combined_f2_variances, combined_f3_means, combined_f3_variances):
    # We assume that f2 and f3 are independent. This is probably not true in practice.
    kernels = []
    logger.info("Creating kernels.")
    for i in range(len(combined_f2_means)):
        #TODO: cov is std and not variance, right?
        k = multivariate_normal(
            mean=[combined_f2_means[i], combined_f3_means[i]],
            cov=np.diag([combined_f2_variances[i], combined_f3_variances[i]])
        )
        kernels.append(k)
    logger.info("Kernels created.")

    vis_focused_grid(kernels)


def vis_blocked(heatmap, splits=1):
    splits = 10
    interval = heatmap.shape[0]//splits
    blocked_img = hlp.blockshaped(heatmap, interval, interval)
    print(blocked_img.shape)
    #print(blocked_img)
    for i, block in enumerate(blocked_img):
        plot_heatmap(block, identifier="/{}".format(i))


def vis_focused_grid(kernels):
    res = 1000
    lats_lngs = []
    lats_lngs.append(np.mgrid[0.555:0.612:res*1j, -1.265:-1.248:res*1j])
    lats_lngs.append(np.mgrid[-0.565:-0.46:res*1j, 1.46:1.6:res*1j])
    lats_lngs.append(np.mgrid[-0.5:0.2:res*1j, -0.85:-0.7:res*1j])
    lats_lngs.append(np.mgrid[0:0.17:res*1j, -0.78:-0.72:res*1j])
    lats_lngs.append(np.mgrid[-0.65:-0.4:res*1j, -0.6:-0.15:res*1j])
    lats_lngs.append(np.mgrid[0.445:0.695:res*1j, -1.2845:-1.2305:res*1j])

    for j, (lat, lng) in enumerate(lats_lngs[0:1]):
        pos = np.dstack((lng, lat))
        logger.info("%sx%s Grid created.", res, res)

        heatmap = np.zeros((res, res))
        T = len(kernels)
        percent = T//100
        for i, k in enumerate(kernels):
            if (i+1) % percent == 0 or (i+1) == T:
                print_progress_bar(i+1, T, prefix = 'Progress:', suffix = 'Complete', length = 50)
            np.add(heatmap, k.pdf(pos), heatmap)
        logger.info("Probabilities for grid calculated.")

        hlp.save_array(heatmap, "combined_gp_heat_focused/{}_{}x{}".format(j, res, res), logger)
        plot_heatmap(heatmap, identifier="_focused/{}_{}x{}".format(j, res, res), show_title=False, with_alpha=True)


def vis_whole_grid(kernels):
    res = 7500
    lat, lng = np.mgrid[-1.7:2:res*1j, -1.35:1.65:res*1j]
    pos = np.dstack((lng, lat))
    logger.info("%sx%s Grid created.", res, res)

    heatmap = np.zeros((res, res))
    T = len(kernels)
    percent = T//100
    for i, k in enumerate(kernels):
        if (i+1) % percent == 0 or (i+1) == T:
            print_progress_bar(i+1, T, prefix = 'Progress:', suffix = 'Complete', length = 50)
        np.add(heatmap, k.pdf(pos), heatmap)
    logger.info("Probabilities for grid calculated.")

    hlp.save_array(heatmap, "heatmap_{}x{}".format(res, res), logger)
    plot_heatmap(heatmap)


def visualise_f2_f3(f2_gp, f3_gp, trajectory_i):
    size = 2000
    tau_grid = np.linspace(0, 1, num=size).reshape(-1,1)
    f2_grid_mean, f2_grid_var = f2_gp.predict_y(tau_grid)
    f3_grid_mean, f3_grid_var = f3_gp.predict_y(tau_grid)
    tau_grid = tau_grid.reshape(-1)

    splits = 32
    interval = size//splits
    for i in range(splits):
        start = i * interval
        stop = (i + 1) * interval

        directory = "f2_f3/{}".format(trajectory_i)
        hlp.ensure_dir(directory)

        plt.figure()
        plt.title('f2 (lng) ({}:{})'.format(start, stop))
        plot_gp_grid_region(tau_grid[start:stop], f2_grid_mean[start:stop], f2_grid_var[start:stop])
        plt.savefig("{}/{}.png".format(directory, (i * 2) + 1))
        plt.clf()
        plt.close()

        plt.figure()
        plt.title('f3 (lat) ({}:{})'.format(start, stop))
        plot_gp_grid_region(tau_grid[start:stop], f3_grid_mean[start:stop], f3_grid_var[start:stop])
        plt.savefig("{}/{}.png".format(directory, (i * 2) + 2))
        plt.clf()
        plt.close()


def plot_gp_grid_region(grid, gp_mean, gp_var):
    plt.plot(grid, gp_mean, lw=1, color="red")
    plt.fill_between(
        grid,
        gp_mean[:,0] - 2*np.sqrt(gp_var[:,0]), 
        gp_mean[:,0] + 2*np.sqrt(gp_var[:,0]), 
        color="red", alpha=0.2
    )

def train_GP(X_tau, Y, session, number, gp_name):
    """GP which maps tau -> lng or lat."""
    with gpflow.defer_build():
        m = gpflow.models.GPR(X_tau, Y, kern=gpflow.kernels.RBF(1))
        m.likelihood.variance = 1e-03
        m.likelihood.variance.trainable = False
        m.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m)
        logger.info("{} GP #{} trained.".format(gp_name, number))
    session.save(BASE_DIR + "GP/model_{}_{}".format(number, gp_name), m)
    return m


if __name__ == "__main__":
    logger = setup_logging("gps_var.py", "gps_var.log")
    BASE_DIR = "gps_var/"
    main(sys.argv[1:])