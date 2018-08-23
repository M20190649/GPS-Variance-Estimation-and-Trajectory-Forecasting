import ml_helper as hlp
from helpers import setup_logging
import getopt
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.stats import sem
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

def main(argv):
    traj = None
    help_line = 'usage: metrics.py -t <traj>'
    try:
        opts, args = getopt.getopt(argv,"ht:",["traj="])
    except getopt.GetoptError:
        print(help_line)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_line)
            sys.exit()
        elif opt in ("-t", "--traj"):
            traj = arg
    if traj is None:
        print(help_line)
        sys.exit(2)
    run(traj)


def run(traj):
    logger.info("Starting execution of metrics.py!")
    BASE_DIR = "forecasting/arrival_time_distributions/"

    segments = 18
    truths = []
    predicted = []
    for i in range(segments):
        result = hlp.load_array("{}{}/{}/predicted".format(BASE_DIR, traj, i))
        truths.extend(result["truth"])
        predicted.extend(result["predicted"])

    truths = np.array(truths)
    predicted = np.array(predicted)
    logger.info("RMSE: {}".format(round(np.sqrt(mean_squared_error(truths, predicted)), 2)))
    logger.info("MAE: {}".format(round(mean_absolute_error(truths, predicted), 2)))



if __name__ == "__main__":
    logger = setup_logging("file_lengths.py", "file_lengths.log")
    main(sys.argv[1:])
