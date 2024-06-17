# tankplayer config file for playing TYPE_TRAVEBUF2 waveforms

RingName        WAVE_RING       # play waveforms into this ring
MyModuleId      MOD_TANKPLAYER  # as this module id
PlayMsgType     TYPE_TRACEBUF2  # msg type to read from file
LogFile         1               # 0=no log; 1=keep log file
HeartBeatInt    30              # seconds between heartbeats
Pause           0               # seconds to pause between wavefiles
StartUpDelay    10              # seconds to wait before playing first wavefile
# SendLate        10.0            # (optional) if present packets are timestamped
ScreenMsg       1               # (optional) information written to screen if non-zero
Debug           1               # (for verbosity)
# List of files to play (up to 50 allowed):
WaveFile        /Volumes/TheWall/PNSN_Tank_Player_Benchmarks/Standard_Operation/tnk/BK.tnk

