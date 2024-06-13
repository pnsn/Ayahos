# Get plain vanilla python stuff
import logging, argparse, os, warnings
from logging.handlers import TimedRotatingFileHandler
# Get the Module Constructor Class 
from PULSE.module.coordinate import EWFlow

# Suppress warnings - just stick to logging
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')


# Setup argument parsing (command line arguments)
parser = argparse.ArgumentParser(description="This is an example EWFlow module for developmental purposes")
parser.add_argument('-f', action='store', dest='config', default='ewflow.ini', type=str)

# Read arguments from command line
arguments = parser.parse_args()

Logger = logging.getLogger(__name__)

# Setup module logfile 
log_path = os.environ['EW_LOG']
log_name = f'{os.path.splitext(arguments.config)[0]}.log'
log_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Rotate files, like eartworm but using only 3 backups
fh = TimedRotatingFileHandler(
    filename = os.path.join(log_path, log_name),
    when='midnight',
    interval=1,
    backupCount=3
)
# Set logging level for log file
fh.setLevel(logging.WARNING)
# Set logging format
fh.setFormatter(log_fmt)

# Set logging level for the terminal
ch = logging.StreamHandler()
ch.setFormatter(log_fmt)
ch.setLevel(logging.INFO)

# Add handler to the root logger
logging.getLogger().addHandler(fh)
logging.getLogger().addHandler(ch)
# # Set root logger logging level
logging.getLogger().setLevel(logging.DEBUG)

# main program start
if __name__ == '__main__':
    mod_ewflow = EWFlow(arguments.config)
    try:
        mod_ewflow.start()
    except KeyboardInterrupt:
        # Logger.critical('Stopping - User Hit Keyboard Interrupt')
        mod_ewflow.stop()