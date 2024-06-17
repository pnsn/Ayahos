

# tankplayer config file for playing waveform files

RingName       SCN_RING        # play waveforms into this ring
MyModuleId    MOD_TANKPLAYER     # as this module id
PlayMsgType   TYPE_TRACEBUF    # msg type to read from file
LogFile       1                # 0=no log; 1=keep log file
HeartBeatInt  30               # seconds between heartbeats
Pause         10               # seconds to pause between wavefiles
StartUpDelay  10               # seconds to wait before playing 1st file
ScreenMsg     1                # (optional) if non-zero, informational messages will be
                               #   written to the screen as data is played
SendLate      10.0           # (optional) if present, packets will be
                               #   timestamped this many seconds before
                               #   current time;
                               # if absent, packets will have original time
                               #   stamps
                               #
Debug         1                # for verbosity

# List of files to play (up to 50 allowed):
WaveFile      ${EW_DATA_DIR}/tankplayer/20060907_1351.tnk

# IgnoreTBVersionNumbers -
# Prevents tankplayer from objecting to tracebuf2 packets that don't have 
# the correct version field.  Not recommended.
IgnoreTBVersionNumbers 0
