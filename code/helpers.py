""" Helper class that contains useful functions """
import logging
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from decimal import Decimal

def setup_logging(logger_name, file_name):
    """
    Setup logging to file and to output.
    Clears up previous active logging.
    """
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    # Configure logging to log to file and to output.
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_name)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt='%(asctime)s:%(levelname)s:	 %(message)s', 
        datefmt='%H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Logging all set up and ready to be used for this run.
    logger.info('--------Starting a fresh run-----------')
    return logger


def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def plot_types(types, output_file="types_barplot.pdf"):
    """ Plot barplot of all types in the given Counter."""
    logging.info("Plotting types")
    df = pd.DataFrame(types)
    df = df.rename(columns={0:'event'})
    print(df.event.value_counts())
    plt.figure(figsize=(16, 5))
    sns.set_color_codes("pastel")
    ax = (df.event.value_counts(normalize=True)*100).plot(kind='bar', fontsize=12, rot=45, title="Event type distribution")
    ax.set_yscale('log')
    ax.title.set_size(24)
    for p in ax.patches:
        v = Decimal(p.get_height())
        if v > 0.1:
            ax.annotate(str(round(v, 2)) + "%", (p.get_x() * 0.99 , p.get_height() * 1.12))
    plt.ylabel('Percentage', fontsize=18)
    plt.xlabel('Event types', fontsize=18)
    plt.xticks(ha='right')
    plt.savefig(output_file, bbox_inches='tight')  


def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def str_gps_to_tuple(string_gps):
    """Helper function that takes a gps in string format (e.g., "58.xxxxx,15.xxxxx")
    and converts it to a tuple with floats (e.g., (58.xxxxx, 15.xxxxx)).
    """
    gps = string_gps.split(",")
    return (float(gps[0]), float(gps[1]))

def epochify(time_stamp):
    """Takes a human readable time_stamp and converts it into Unix epoch time.
    Example: "2016-11-17 06:25:02" => 1479363902
    """
    return int(time.mktime(time.strptime(time_stamp, '%Y-%m-%d %H:%M:%S')))