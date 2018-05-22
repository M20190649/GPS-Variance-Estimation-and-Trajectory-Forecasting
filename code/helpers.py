""" Helper class that contains useful functions """
import logging
import subprocess

def setup_logging():
    """
    Setup logging to file and to output.
    Clears up previous active logging.
    """
    # Clear potential previous mess
    logging.shutdown()

    # Configure logging to log to file and to output.
    logging.basicConfig(
        format='%(asctime)s:%(name)s:%(levelname)s:	 %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.FileHandler("main.log"), logging.StreamHandler()])

    # Logging all set up and ready to be used for this run.
    logging.info('--------Starting a fresh run-----------')

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