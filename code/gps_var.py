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
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from helpers import setup_logging
from collections import defaultdict
from gmplot import gmplot
from operator import add
from scipy.stats import multivariate_normal 

def main(argv):
    line_number = None
    create_variation_GPs = True
    help_line = 'usage: gps_var.py -l <line_number> --load-variation-gp'
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
    logger.info("Starting execution of gps_var.py!")

    visualise_trajectory_gp = False
    visualise_tau_gp = False

    trajectories = hlp.load_trajectories_from_file(line_number, logger)
    
    trajectory_key = "Lötgatan:Linköpings resecentrum"
    vehicle_id, journey = trajectories[trajectory_key][0]
    pprint.pprint(journey.route)

    #plot_speed_stops(journey.route)
    #plot_speed_time(journey)

    if create_variation_GPs:
        f2_means, f2_variances, f3_means, f3_variances = create_GPS_variation_GPs(
            journey, 
            trajectories, 
            trajectory_key, 
            visualise_trajectory_gp,
            visualise_tau_gp
        )
    else:
        f2_means, f2_variances, f3_means, f3_variances = load_GPS_variation_GPs()
    
    # TODO: Just testing this. #MidnightIdeas

    combined_f2_means = combine_mean(f2_means)
    combined_f2_variances = combine_variance(f2_variances, f2_means, combined_f2_means)
    combined_f3_means = combine_mean(f3_means)
    combined_f3_variances = combine_variance(f3_variances, f3_means, combined_f3_means)


    kernels = []
    logger.info("Creating kernels.")
    for i in range(len(combined_f2_means)):
        # We want (lng, lat) coords for this kernel
        k = multivariate_normal(
            mean=[combined_f3_means[i], combined_f2_means[i]],
            cov=np.diag([combined_f3_variances[i], combined_f2_variances[i]])
        )
        kernels.append(k)
    logger.info("Kernels created.")

    res = 5000
    x, y = np.mgrid[-1.2645:-1.2505:res*1j, 0.545:0.595:res*1j]#np.mgrid[-1.27:-1.25:res*1j, 0.59:0.61:res*1j]
    pos = np.dstack((x, y))
    logger.info("Grid Created.")
    #x = np.linspace(-1.5, 1.75, num=res)
    #y = np.linspace(-1.75, 2, num=res)
    plt.figure()
    for i, k in enumerate(kernels[1:10]):
        print(i)
        plt.contourf(x, y, k.pdf(pos))#, color=["red", "blue"][i])
    plt.title("Contour 1 Kernel GP")
    plt.savefig("contour_test.png")
    plt.clf()
    plt.close()
    return

    
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    logger.info("%sx%s Grid created.", res, res)
    zz = sum([k.pdf(xxyy) for k in kernels])
    logger.info("Probabilities for grid calculated.")

    img = np.zz.reshape((res, res))
    print(img)
    plt.figure()
    plt.imshow(img)
    plt.title("1 Kernel GP")
    plt.savefig("heatmap_test.png")
    plt.clf()
    plt.close()




    return
    combined_f2_means = combine_mean(f2_means)
    combined_f2_variances = combine_variance(f2_variances, f2_means, combined_f2_means)
    combined_f3_means = combine_mean(f3_means)
    combined_f3_variances = combine_variance(f3_variances, f3_means, combined_f3_means)

    #visualise_combined_f2_f3_gp(combined_f2_means, combined_f2_variances, combined_f3_means, combined_f3_variances)
    visualise_combined_f2_f3_gp_heatmap(combined_f2_means, combined_f2_variances, combined_f3_means, combined_f3_variances)
    #plot_trajectories(trajectories, print_only=True)


def create_GPS_variation_GPs(journey, trajectories, trajectory_key, visualise_trajectory_gp, visualise_tau_gp):
    """ Function that creates two GPS variation GPs (f2=lat, f3=lng) for each trajectory.
    Returns and saves the means and variances for each (f2,f3) pair."""
    logger.info("Creating new GPs f1, f2, and f3 for GPS variation estimation.")
    # Do stop compression.
    events = [e for e in journey.route if e["event.type"] == "ObservedPositionEvent" and e["speed"] > 3]
    events = hlp.filter_duplicates(events)

    X = np.vstack([e["gps"] for e in events])
    Y = np.linspace(0, 1, num = X.shape[0]).reshape(-1,1)
    scaler = StandardScaler().fit(X)
    X_fitted = scaler.transform(X)

    f1_gp = hlp.train_f1_gp(X_fitted, Y, logger)
    if visualise_tau_gp:
        visualise_f1_gp(X_fitted, Y, file_name="trained", title="f1 (trained)")

    trajectory_GPs = train_trajectory_GPs(
        trajectories[trajectory_key], 
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
    save_array(f2_means, "f2_means")
    f2_variances = np.vstack(f2_variances_array)
    save_array(f2_variances, "f2_variances")
    f3_means = np.vstack(f3_means_array)
    save_array(f3_means, "f3_means")
    f3_variances = np.vstack(f3_variances_array)
    save_array(f3_variances, "f3_variances")
    return f2_means, f2_variances, f3_means, f3_variances


def load_GPS_variation_GPs():
    logger.info("Loading GPs f1, f2, and f3 for GPS variation estimation.")
    return (
        load_array("f2_means"),
        load_array("f2_variances"),
        load_array("f3_means"),
        load_array("f3_variances")
    )


def load_array(array_name):
    with open("{}.pickle".format(array_name), 'rb') as file:
        logger.info("Array loaded from %s.pickle", array_name)
        return pickle.load(file)


def save_array(array, array_name):
    with open("{}.pickle".format(array_name), 'wb') as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Array saved to %s.pickle", array_name)


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


def train_trajectory_GPs(trajectories, scaler, f1_gp, visualise_tau_gp, visualise_trajectory_gp):
    trajectory_i = 0
    trajectory_GPs = []

    for vehicle_id, journey in trajectories:
        events = [e for e in journey.route if e["event.type"] == "ObservedPositionEvent" and e["speed"] > 3]
        events = hlp.filter_duplicates(events)

        xx = np.vstack([e["gps"] for e in events])
        xx_fit = scaler.transform(xx)
        xx_lat = xx_fit[:,0].reshape(-1,1)
        xx_lng = xx_fit[:,1].reshape(-1,1)
        mean, var = f1_gp.predict_y(xx_fit)
        mean = mean[:,0].reshape(-1,1)

        if visualise_tau_gp:
            visualise_f1_gp(xx_fit, mean, file_name="test_{}".format(trajectory_i), title="f1 (test {})".format(trajectory_i))

        assert len(mean), len(set(mean))

        f2_gp = train_f2_gp(mean, xx_lat)
        f3_gp = train_f3_gp(mean, xx_lng)

        trajectory_GPs.append((f2_gp, f3_gp))
        
        if visualise_trajectory_gp:
            visualise_f2_f3(f2_gp, f3_gp, trajectory_i)
        
        trajectory_i += 1
    return trajectory_GPs


def visualise_f1_gp(X, Y, file_name, title):
    plt.figure()
    plt.title(title)
    plt.scatter(X[:,1], X[:,0], c=Y[:,0], cmap='coolwarm', s=0.5)
    plt.colorbar()
    plt.savefig("tau_gp/{}.png".format(file_name))


def visualise_combined_f2_f3_gp(combined_f2_means, combined_f2_variances, combined_f3_means, combined_f3_variances):
    splits = 32
    interval = len(combined_f2_means)//splits
    color = "blue"
    for i in range(splits):
        start = i * interval
        stop = (i + 1) * interval
        plt.figure()
        plt.title("Combined GP ({}:{}".format(start, stop))
        plt.plot(combined_f3_means[start:stop], combined_f2_means[start:stop], color=color)
        plt.plot(combined_f3_means[start:stop] - 2*np.sqrt(combined_f3_variances[start:stop]), combined_f2_means[start:stop] + 2*np.sqrt(combined_f2_variances[start:stop]), '--', color=color) 
        plt.plot(combined_f3_means[start:stop] + 2*np.sqrt(combined_f3_variances[start:stop]), combined_f2_means[start:stop] - 2*np.sqrt(combined_f2_variances[start:stop]), '--', color=color) 
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
            mean=[combined_f3_means[i], combined_f2_means[i]],
            cov=np.diag([combined_f3_variances[i], combined_f2_variances[i]])
        )
        kernels.append(k)
    logger.info("Kernels created.")

    res = 1500
    y = np.linspace(-1.75, 2, num=res)
    x = np.linspace(-1.5, 1.75, num=res)
    xx, yy = np.meshgrid(x,y)
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    logger.info("%sx%s Grid created.", res, res)
    zz = sum([k.pdf(xxyy) for k in kernels])
    logger.info("Probabilities for grid calculated.")

    img = np.flip(zz.reshape((res, res)), 0)
    print(img)
    plt.imshow(img)
    plt.title("Combined GP")
    plt.savefig("combined_gp2.png")
    plt.close()
    



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

        plt.figure()
        plt.title('f2 ({}:{})'.format(start, stop))
        plot_gp_grid_region(tau_grid[start:stop], f2_grid_mean[start:stop], f2_grid_var[start:stop])
        plt.savefig("f2_f3/{}/{}.png".format(trajectory_i, (i * 2) + 1))
        plt.clf()
        plt.close()

        plt.figure()
        plt.title('f3 ({}:{})'.format(start, stop))
        plot_gp_grid_region(tau_grid[start:stop], f3_grid_mean[start:stop], f3_grid_var[start:stop])
        plt.savefig("f2_f3/{}/{}.png".format(trajectory_i, (i * 2) + 2))
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


def train_f2_gp(X_tau, Y_lat):
    """GP which maps tau -> lat."""
    with gpflow.defer_build():
        m = gpflow.models.GPR(X_tau, Y_lat, kern=gpflow.kernels.Matern32(1))
        m.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m)
        logger.info("f2 GP trained.")
    return m


def train_f3_gp(X_tau, Y_lng):
    """GP which maps tau -> lng."""
    with gpflow.defer_build():
        m = gpflow.models.GPR(X_tau, Y_lng, kern=gpflow.kernels.Matern32(1))
        m.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m)
        logger.info("f3 GP trained.")
    return m


def plot_trajectories(trajectories, print_only=False):
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




if __name__ == "__main__":
    logger = setup_logging("gps_var.py", "gps_var.log")
    main(sys.argv[1:])