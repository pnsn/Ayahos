#
#                     Configuration File for slink2ew
#
MyModuleId       MOD_SLINK2EW
RingName         WAVE_RING       # Transport ring to write data to.

HeartBeatInterval     30         # Heartbeat interval, in seconds.
LogFile               1          # 1 -> Keep log, 0 -> no log file
                                 # 2 -> write to module log but not stderr/stdout
Verbosity      0		 # Set level of verbosity.

SLhost   rtserve.iris.washington.edu                # Host address of the SeedLink server
SLport         18000             # Port number of the SeedLink server

StateFile                        # If this flag is specified (uncommented) a 
                                 # file with a list of sequence numbers is
                                 # written, during a clean module shutdown,
                                 # to the parameter directory with the name
                                 # "slink<mod id>.state". During module startup 
                                 # these sequence numbers are used to resume
                                 # data streams from the last received data.
                                 # Using this functionality is highly
                                 # recommended.

#StateFileInt   100              # This controls the interval (in packets 
	                         # received) at which the state is saved in
                                 # the state file.  Default is 100 packets,
                                 # 0 to disable.

#NetworkTimeout 600              # Network timeout, after this many idle
                                 # seconds the connection will be reset.
                                 # Default is 600 seconds, 0 to disable.

#NetworkDelay   30               # Network re-connect delay in seconds.

#KeepAlive      0                # Send keepalive packets (when idle) at this
                                 # interval in seconds.  Default is 0 (disabled).

#ForceTraceBuf1 0                # On systems that support TRACEBUF2
                                 # messages this flag will force the module
                                 # to create TRACEBUF messages instead.
                                 # Most people will never need this.

# Selectors and Stream's.  If any Stream lines are specified the connection
# to the SeedLink server will be configured in multi-station mode using
# Selectors, if any, as defaults.  If no Stream lines are specified the
# connection will be configured in uni-station mode using Selectors, if any.

#Selectors      "BH?.D"          # SeedLink selectors.  These selectors are used
                                 # for a uni-station mode connection.  If one
                                 # or more 'Stream' entries are given these are
                                 # used as default selectors for multi-station
                                 # mode data streams.  See description of
                                 # SeedLink selectors below.  Multiple selectors
                                 # must be enclosed in quotes.


# List each data stream (a network and station code pair) that you
# wish to request from the server with a "Stream" command.  If one or
# more Stream commands are given the connection will be configured in
# multi-station mode (multiple station data streams over a single
# network connection).  If no Stream commands are specified the
# connection will be configured in uni-station mode, optionally using
# any specified "Selectors".  A Stream command should be followed by a
# stream key, a network code followed by a station code separated by
# an underscore (i.e. IU_KONO).  SeedLink selectors for a specific
# stream may optionally be specified after the stream key.  Multiple
# selectors must be enclosed in quotes.  Any selectors specified with
# the Selectors command above are used as defaults when no selectors
# are specified for a given stream.

#Stream  GE_DSB   "BH?.D HH?.D"
#Stream  II_KONO  00BH?.D

# Some SeedLink servers support extended selection capability and
# allow wildcars (either '*' or '?') for both the network and station
# fields, for example to request all stations from the TA network:

# Washington Broadbands
Stream UW_* HH?.D
# Washgington Strong Motion
Stream UO_* HH?.D
# Mount St. Helens Stations
# CVO Stations
#Stream CC_EDM BH?.D
#Stream CC_HOA BH?.D
#Stream CC_HSR BH?.D
#Stream CC_LOO BH?.D
#Stream CC_REM BH?.D
#Stream CC_SEP BH?.D
#Stream CC_SHW BH?.D
#Stream CC_STD BH?.D
#Stream CC_SUG BH?.D
#Stream CC_USRF BH?.D
#Stream CC_VALT BH?.D
# Plate Boundary Observatory Stations
#Stream PB_B201 EH?.D
#Stream PB_B202 EH?.D
#Stream PB_B203 EH?.D
#Stream PB_B204 EH?.D
# UW 1C Stations
#Stream UW_ELK EHZ.D
# UW 6C Stations
#Stream UW_TOUT "HH?.D EN?.D"
#Stream UW_YACT "HH?.D EN?.D"
#Stream UW_ASR2 "HH?.D EN?.D"
#Stream UW_CINE "HH?.D EN?.D"
# UW 4C Stations
#Stream UW_PUPY "EHZ.D EN?.D"

#(notes regarding "selectors" from a SeedLink configuration file)
#
#   The "selectors" parameter tells to request packets that match given
#   selectors. This helps to reduce network traffic. A packet is sent to
#   client if it matches any positive selector (without leading "!") and
#   doesn't match any negative selectors (with "!"). General format of
#   selectors is LLSSS.T, where LL is location, SSS is channel, and T is
#   type (one of DECOTL for data, event, calibration, blockette, timing,
#   and log records). "LL", ".T", and "LLSSS." can be omitted, meaning
#   "any". It is also possible to use "?" in place of L and S.
#
#   Some examples:
#   BH?            - BHZ, BHN, BHE (all record types)
#   00BH?          - BHZ, BHN, BHE with location code '00' (all record types)
#   BH?.D          - BHZ, BHN, BHE (data records)
#   BH? !E         - BHZ, BHN, BHE (excluding detection records)
#   BH? E          - BHZ, BHN, BHE plus detection records of all channels
#   !LCQ !LEP      - exclude LCQ and LEP channels
#   !L !T          - exclude log and timing records
#
#
# For slink2ew no record types except data records will be written to
# the ring.  In other words, requesting any records in addition to
# data records is a waste.
