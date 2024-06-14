
#   ew2file_generic configuration file
#
#
 MyModuleId     MOD_EW2FILE         # module id for this program
 InRing         HYPO_RING             # transport ring to use for input
 HeartBeatInt   30                  # EW internal heartbeat interval (sec)
 LogFile        1                   # If 0, don't write logfile
                                    # if 2, write to module log but not to
                                    # stderr/stdout .
#
# Logos of messages to export to client systems
# Each message of a given logo will be written to a file with the specified
# File Suffix at the end of the file name folloing `.'.
# Do NOT include a `.' in the suffix.
# Use as many GetMsgLogo commands as you want.
#              Installation       Module       Message Type        File Suffix
 GetMsgLogo    INST_WILDCARD        MOD_WILDCARD   TYPE_HYP2000ARC     arc
 
 MaxMsgSize        10000         # maximum size (bytes) for input msgs
 QueueSize         1000        # number of msgs to buffer

 TimeBasedFilename 1           # set non-zero if you want a time string in the
                               # form of "yyyymmdd-hhmmss." at the beginning of
                               # each output file. The timestring is generated from
                               # msg contents, currently possible for only msgtypes:
                               # TYPE_PICK2K, TYPE_PICK_SCNL, TYPE_HYP2000ARC
       
# Optional heartbeat file commands. 
#----------------------------------
# Default HeartFileInt is 0 => send no heartbeat files.
# Default HeartFileMsg is "Alive".
# Default HeartFileSuffix is "hrt"
 HeartFileInt     0           # heartbeat file interval; optional
 HeartFileMsg     Alive        # string for heartbeat file; optional  
 HeartFileSuffix  hrtUCB       # suffix for heartbeat files; do not include `.'

# SerialKeeper: File name where the serial number of the output files is 
# stored. This should be a unique file name, in the params directory.
# If the file does not exist, ew2file will attempt to create it.
SerialKeeper  ${EW_DATA_DIR}/tmp/ew2file_keeper

# TempDir: Directory in which files are created. This directory must be in the
# same filesystem as all the SendDir directories. Exactly one TempDir is required.
TempDir      ${EW_DATA_DIR}/tmp

# SendDir: Directory to which output files will be moved after they have 
# been created. 
#  Use as many SendDir commands as you need.
# Be sure the final line is terminated with a newline.
SendDir      ${EW_DATA_DIR}/arc
