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
import seaborn as sns
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from helpers import setup_logging, print_progress_bar
from collections import defaultdict
from gmplot import gmplot
from operator import add
from scipy.stats import multivariate_normal 
from matplotlib.colors import ListedColormap

def main(argv):
    line_number = None
    create_variation_GPs = True
    load_heatmap = False
    help_line = 'usage: gps_var.py -l <line_number> --load-variation-gp --load-heatmap'
    try:
        opts, args = getopt.getopt(argv,"hl:",["line=", "load-variation-gp", "load-heatmap"])
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
        elif opt == "--load-heatmap":
            load_heatmap = True

    logger.info("Starting execution of gps_var.py!")

    if not load_heatmap:
        run(line_number, create_variation_GPs)
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
    
    
def run(line_number, create_variation_GPs):
    visualise_trajectory_gp = False # Quicker if False
    visualise_tau_gp = False # Quicker if False

    trajectories = hlp.load_trajectories_from_file(line_number, logger)
    hlp.plot_trajectories(trajectories, logger, print_only=True)
    
    trajectory_key = "Lötgatan:Linköpings resecentrum"
    vehicle_id, journey = trajectories[trajectory_key][0]
    #pprint.pprint(journey.route)
    #hlp.plot_speed_time(trajectories["Lötgatan:Fönvindsvägen östra"][4][1].segment_at("Linköpings resecentrum")[0])
    #hlp.plot_speed_stops(journey.route)

    #plot_speed_stops(journey.route)
    #plot_speed_time(journey)

    if create_variation_GPs:
        f2_means, f2_variances, f3_means, f3_variances, scaler = create_GPS_variation_GPs(
            journey, 
            trajectories, 
            trajectory_key, 
            visualise_trajectory_gp,
            visualise_tau_gp
        )
    else:
        f2_means, f2_variances, f3_means, f3_variances, scaler = load_GPS_variation_GPs()
    
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

    combined_f2_means = combine_mean(f2_means)
    combined_f2_variances = combine_variance(f2_variances, f2_means, combined_f2_means)
    combined_f3_means = combine_mean(f3_means)
    combined_f3_variances = combine_variance(f3_variances, f3_means, combined_f3_means)

    visualise_combined_f2_f3_gp(combined_f2_means, combined_f2_variances, combined_f3_means, combined_f3_variances, scaler)
    visualise_combined_f2_f3_gp_heatmap(combined_f2_means, combined_f2_variances, combined_f3_means, combined_f3_variances)



def create_GPS_variation_GPs(journey, trajectories, trajectory_key, visualise_trajectory_gp, visualise_tau_gp):
    """ Function that creates two GPS variation GPs (f2=lat, f3=lng) for each trajectory.
    Returns and saves the means and variances for each (f2,f3) pair."""
    logger.info("Creating new GPs f1, f2, and f3 for GPS variation estimation.")
    f1_gp, scaler = hlp.create_f1_GP(journey.route, logger, visualise_tau_gp)
    hlp.save_array(scaler, "scaler", logger)

    trajectory_GPs = train_trajectory_GPs(
        trajectories,
        trajectory_key, 
        scaler, 
        f1_gp, 
        visualise_tau_gp, 
        visualise_trajectory_gp
    )

    size = 2000
    tau_grid = np.linspace(0, 1, num=size).reshape(-1,1)

    f2_means_array = []
    f2_variances_array = []
    f3_means_array = []
    f3_variances_array = []

    for f2, f3 in trajectory_GPs:
        f2_means, f2_variances = f2.predict_y(tau_grid)
        f2_means_array.append(f2_means.reshape(-1))
        f2_variances_array.append(f2_variances.reshape(-1))

        f3_means, f3_variances = f3.predict_y(tau_grid)
        f3_means_array.append(f3_means.reshape(-1))
        f3_variances_array.append(f3_variances.reshape(-1))

    f2_means = np.vstack(f2_means_array)
    hlp.save_array(f2_means, "f2_means", logger)
    f2_variances = np.vstack(f2_variances_array)
    hlp.save_array(f2_variances, "f2_variances", logger)
    f3_means = np.vstack(f3_means_array)
    hlp.save_array(f3_means, "f3_means", logger)
    f3_variances = np.vstack(f3_variances_array)
    hlp.save_array(f3_variances, "f3_variances", logger)
    return f2_means, f2_variances, f3_means, f3_variances, scaler


def load_GPS_variation_GPs():
    logger.info("Loading GPs f1, f2, and f3 for GPS variation estimation.")
    return (
        hlp.load_array("f2_means", logger),
        hlp.load_array("f2_variances", logger),
        hlp.load_array("f3_means", logger),
        hlp.load_array("f3_variances", logger),
        hlp.load_array("scaler", logger)
    )




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


def train_trajectory_GPs(trajectories, trajectory_key, scaler, f1_gp, visualise_tau_gp, visualise_trajectory_gp):
    trajectory_start, trajectory_end = trajectory_key.split(":")
    logger.info("Trajectory start/end: %s/%s", trajectory_start, trajectory_end)
    all_trajectories = trajectories[trajectory_key]
    
    for key, values in trajectories.items():
        k_start, k_end = key.split(":")
        if trajectory_start == k_start and trajectory_end != k_end:
            for vehicle_id, journey in values:
                segmented_journey, _ = journey.segment_at(trajectory_end)
                if segmented_journey is None:
                    logger.info("Journey not segmented when it should have been: %s, %s", trajectory_start, trajectory_end)               
                all_trajectories.append((vehicle_id, segmented_journey))

    trajectory_GPs = []
    for i, (vehicle_id, journey) in enumerate(all_trajectories):
        events = [e for e in journey.route if e["event.type"] == "ObservedPositionEvent" and e["speed"] > 0.1]
        events = hlp.filter_duplicates(events)

        xx = np.vstack([e["gps"][::-1] for e in events])
        xx_fit = scaler.transform(xx)
        xx_lng = xx_fit[:,0].reshape(-1,1)
        xx_lat = xx_fit[:,1].reshape(-1,1)
        mean, var = f1_gp.predict_y(xx_fit)
        mean = mean[:,0].reshape(-1,1)

        if visualise_tau_gp:
            hlp.visualise_f1_gp(xx_fit, mean, file_name="test_{}".format(i), title="f1 (test {})".format(i))

        assert len(mean), len(set(mean))

        f2_gp = train_f2_gp(mean, xx_lng, i)
        f3_gp = train_f3_gp(mean, xx_lat, i)

        trajectory_GPs.append((f2_gp, f3_gp))
        
        if visualise_trajectory_gp:
            visualise_f2_f3(f2_gp, f3_gp, i)
        
    return trajectory_GPs


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
        # We want (lng, lat) coords for this kernel
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


def train_f2_gp(X_tau, Y_lng, number=None):
    """GP which maps tau -> lng."""
    with gpflow.defer_build():
        m = gpflow.models.GPR(X_tau, Y_lng, kern=gpflow.kernels.Matern32(1))
        m.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m)
        logger.info("f2 GP #{} trained.".format("" if number is None else number))
    return m


def train_f3_gp(X_tau, Y_lat, number=None):
    """GP which maps tau -> lat."""
    with gpflow.defer_build():
        m = gpflow.models.GPR(X_tau, Y_lat, kern=gpflow.kernels.Matern32(1))
        m.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m)
        logger.info("f3 GP #{} trained.".format("" if number is None else number))
    return m







if __name__ == "__main__":
    logger = setup_logging("gps_var.py", "gps_var.log")
    main(sys.argv[1:])