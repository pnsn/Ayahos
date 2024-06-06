"""
Ayahos_Test is an example Ayahos module
"""
# Get plain vanilla python stuff
import logging, argparse, os
from logging.handlers import TimedRotatingFileHandler
# Get the Module Constructor Class - 
from ayahos.modules.Ayahos_Test.ayahos_get_waves import EWModuleConstructor
# Imports required for base module initialization
from ayahos.wyrms import *
# Import required to initialize a SeisBench WaveformModel object specified in the *.ini file
import seisbench.models as sbm

# Setup argument parsing (command line arguments)
parser = argparse.ArgumentParser(description="This is an example Ayahos module for developmental purposes")
parser.add_argument('-f', action='store', dest='config', default='mod_ayahos.ini', type=str)

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
fh.setLevel(logging.DEBUG)
# Set logging format
fh.setFormatter(log_fmt)
# Add handler to the root logger
logging.getLogger().addHandler(fh)
# Set root logger logging level
logging.getLogger().setLevel(logging.DEBUG)


# main program start
if __name__ == '__main__':
    mod_ayahos = EWModuleConstructor(arguments.config)
    try:
        mod_ayahos.start()
    except KeyboardInterrupt:
        # Logger.critical('Stopping - User Hit Keyboard Interrupt')
        mod_ayahos.stop()