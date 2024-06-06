#
#       Startstop Configuration File
#
#   <nRing> is the number of transport rings to create
#   <Ring> specifies the name fo a ring followed by it's size
#   in kilobytes

 nRing                2
 Ring    WAVE_RING 1024
 Ring    PICK_RING 1024
#
 MyModuleId     MOD_STARTSTOP
 HeartbeatInt   50
 MyClassName    OTHER
 MyPriority     0
 LogFile        1
 KillDelay      30
 HardKillDelay  5

 Process            "statmgr statmgr.d"
 Class/Priority     OTHER 0
#
 Process            "ayahos_pick_mod.py -f ayahos_pick_mod.d"
 Class/Priority     OTHER 0