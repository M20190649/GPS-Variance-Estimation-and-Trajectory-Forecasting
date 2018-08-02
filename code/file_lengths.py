import helpers as hlp
import getopt
import matplotlib.pyplot as plt
import sys
import os

def main(argv):
    path = None
    help_line = 'usage: file_lengths.py -p <path>'
    try:
        opts, args = getopt.getopt(argv,"hp:",["path="])
    except getopt.GetoptError:
        print(help_line)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_line)
            sys.exit()
        elif opt in ("-p", "--path"):
            path = arg
    if path is None:
        print(help_line)
        sys.exit(2)
    run(path)


def run(path):
    logger.info("Starting execution of file_lengths.py!")
    event_counts = []
    file_sizes = []
    files = [f for f in os.listdir(path) if ".log" in f]
    T = len(files)
    percent = T//100 if T >= 100 else 1
    for i, file_name in enumerate(files):
        if (i+1) % percent == 0 or (i+1) == T:
            hlp.print_progress_bar(i+1, T, length=50)
        event_counts.append(hlp.file_len(path + file_name))
        file_sizes.append(os.path.getsize(path + file_name)/1e9)
    logger.info("Total Size: %sGB", sum(file_sizes))
    
    fig = plt.figure()
    plt.subplot(121)
    plt.boxplot(event_counts, showmeans=True)
    plt.title("Number of Events")

    plt.subplot(122)
    plt.boxplot(file_sizes, showmeans=True)
    plt.title("Data Log Sizes (GB)")
    
    fig.suptitle("Measurements on {} Logs".format(len(event_counts)))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    plt.savefig("log_sizes.png")

if __name__ == "__main__":
    logger = hlp.setup_logging("file_lengths.py", "file_lengths.log")
    main(sys.argv[1:])
