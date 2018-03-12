""" Helper class that contains useful functions """
import logging

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