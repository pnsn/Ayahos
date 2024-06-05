"""
Ayahos_Pick is an example Ayahos module that conducts
"""
import logging, configparser, argparse, os
from logging.handlers import TimedRotatingFileHandler
from ayahos_pick import AyahosPick

# Setup argument parsing (command line arguments)
parser = argparse.ArgumentParser(description="This is an example ring2ring ML picking module using one model and one weight")
parser.add_argument('-f', action='store', dest='config', default='mod_ayahos_pick.d', type=str)

# Read arguments from command line
arguments = parser.parse_args()

# Setup module logfile 
log_path = os.environ['EW_LOG']
log_name = f'{os.path.spiltext(arguments.config)}.log'
log_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Rotate files, like eartworm but using only 3 backups
fh = TimedRotatingFileHandler(
    filename = os.path.join(log_path, log_name),
    when='minute',
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
    ayahos_pick = AyahosPick(arguments.ConfFile)
    try:
        ayahos_pick.start()
    except KeyboardInterrupt:
        Logger.critical('Stopping - User Hit Keyboard Interrupt')
        ayahos_pick.stop()