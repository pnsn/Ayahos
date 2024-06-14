#                 Startstop Configuration File for Windows NT
#
#    <nRing> is the number of transport rings to create.
#    <Ring> specifies the name of a ring followed by it's size
#    in kilobytes, eg        Ring    WAVE_RING 1024
#    The maximum size of a ring is 1024 kilobytes.
#    Ring names are listed in file earthworm.h.
#
  nRing               4
  Ring   WAVE_RING   1024
#  Ring    SCN_RING   500
  Ring   PICK_RING   128
#  Ring   HYPO_RING   128
  Ring   TEST_RING   128
  Ring STATUS_RING   128

 MyModuleId       MOD_STARTSTOP  # Module Id for this program
 HeartbeatInt     15             # Heartbeat interval in seconds
# the next is for windows only
# MyPriorityClass  Normal         # For startstop
# these next 2  are for unix
 MyClassName   OTHER             # For this program
 MyPriority     0             # For this program
 LogFile           1             # 1=write a log file to disk, 0=don't, 
				         # 2=write to module log but not stderr/stdout
 KillDelay        5             # number of seconds to wait on shutdown
                                 #  for a child process to self-terminate
                                 #  before killing it
 HardKillDelay    5             # wait this many more secs for procs to die before really killing them

 # statmgrDelay		2        # Uncomment to specify the number of seconds
					   # to wait after starting statmgr 
					   # default is 1 second

#
#    PriorityClass values:
#       Idle            4
#       Normal          9 forground, 7 background
#       High            13
#       RealTime        24
#
#    ThreadPriority values:
#       Lowest          PriorityClass - 2
#       BelowNormal     PriorityClass - 1
#       Normal          PriorityClass
#       AboveNormal     PriorityClass + 1
#       Highest         PriorityClass + 2
#       TimeCritical    31 if PriorityClass is RealTime; 15 otherwise
#       Idle            16 if PriorityClass is RealTime; 1 otherwise
#
#    Display can be either NewConsole, NoNewConsole, or MinimizedConsole.
#
#    If the command string required to start a process contains
#    embedded blanks, it must be enclosed in double-quotes.
#    Processes may be disabled by commenting them out.
#    To comment out a line, preceed the line by #.
#
#
  Process          "copystatus WAVE_RING STATUS_RING"
  Class/Priority    OTHER 0
#
#  Process          "copystatus SCN_RING STATUS_RING"
#  Class/Priority    OTHER 0
#
  Process          "copystatus PICK_RING STATUS_RING"
  Class/Priority    OTHER 0
#
#  Process          "copystatus HYPO_RING STATUS_RING"
#  Class/Priority    OTHER 0
#
  Process          "copystatus TEST_RING STATUS_RING"
  Class/Priority    OTHER 0
#
 Process          "statmgr statmgr.d"
 Class/Priority    OTHER 0
#
 Process          "slink2ew slink2ew.d"
 Class/Priority    OTHER 0

# Process          "wave_serverV wave_serverV.d"
# Class/Priority    OTHER 0

# Process          "pick_ew pick_ew.d"
# Class/Priority    OTHER 0

# Process          "binder_ew binder_ew.d"
# Class/Priority    OTHER 0

# Process          "eqproc eqproc.d"
# Class/Priority    OTHER 0

# Process          "scn2scnl scn2scnl.d"
# Class/Priority    OTHER 0

# Process          "tankplayer tankplayer.d"
# Class/Priority    OTHER 0

# Process          "ew2file ew2file.d"
# Class/Priority    OTHER 0

# Process          "carlstatrig carlstatrig.d"
# Class/Priority    OTHER 0

# Process          "carlsubtrig carlsubtrig_nm.d"
# Class/Priority    OTHER 0

# Process	  "localmag localmag.d"
# Class/Priority    OTHER 0

