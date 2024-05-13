import os, logging
from wyrm.core.wyrms.heartwyrm import HeartWyrm
from wyrm.core.wyrms.methodwyrm import MethodWyrm
from wyrm.core.wyrms.earwyrm import WaveWyrm
from wyrm.core.wyrms.messagewyrm import MessageWyrm
from wyrm.core.wyrms.ringwyrm import RingWyrm


Logger = logging.getLogger(__name__)

# Get EW_HOME environmental variable
try:
    EW_HOME = os.environ['EW_HOME']
except KeyError:
    Logger.critical('Environmental variable "EW_HOME" not set')

# Initialize HeartWyrm object
heartwyrm = HeartWyrm(ew_home=EW_HOME, wait_sec=0,
                      DR_ID=1000, MOD_ID=200, INST_ID=7,
                      HB_PERIOD=15, wyrm_list = {})

# Establish connection to the WAVE ring
heartwyrm.add_connection(RING_ID=1000, RING_name='WAVE')
# Create a custom memory ring for processed data
heartwyrm.module.add_ring(9999)
heartwyrm.add_connection(RING_ID=9999, RING_name='my_custom_FILT_ring')

# Initialize WaveWyrms
iwavewyrm = WaveWyrm(module = heartwyrm.module, conn_id=0, flow_direction='to_py')
owavewyrm = WaveWyrm(module=heartwyrm.module, conn_id=1, )