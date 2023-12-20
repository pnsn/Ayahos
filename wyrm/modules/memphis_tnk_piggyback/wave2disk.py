"""
:module: wyrm/modules/memphis_tnk_piggyback/ring2disk.py
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    Provide a lightweight demonstration of Wyrm's functionality
    wherein waveform data are collected from the WAVE ring of 
    a running instance of the Memphis Tankplayer example and 
    written to disk as MSEED files using the Wyrm API
"""
from wyrm.core.sequential import HeartWyrm
from wyrm.core.io import Wave2PyWyrm, DiskOWyrm
import pandas as pd
import os

# Get SNCL information for trigger stations
EW_HOME = os.path.join("/usr", "local", "earthworm")
STA = os.path.join(EW_HOME, "memphis", "params", "memphis_trig.sta")
# Use trig.sta file to get station list & parse with Pandas
df = pd.read_table(
    STA, delim_whitespace=True, names=["type", "sta", "cha", "net", "loc", "X"]
)
# Get Unique station/network names
stanet = df[["sta", "net"]].value_counts().index.values

# Initialize HeartWyrm
heart = HeartWyrm(
    pulse_rate=1, DR_ID=1000, MOD_ID=200, INST_ID=6, HB_PERIOD=30, debug=False
)
# Initialize EW Module with user-input check enabled
heart.initialize_module(user_check=True)
# Add connection to WAVE ring
heart.add_connection(RING_ID=1000, RING_Name="WAVE_RING")
# Alias WAVE RING info
wr_info = heart.conn_info[heart.conn_info["RING_Name"] == "WAVE_RING"].T

# Initialize Elements going into the HeartWyrm
## INPUT WYRM
wave2py = Wave2PyWyrm(module=heart.module, conn_index=wr_info.RING_ID)

## OUTPUT WYRM
disko = DiskOWyrm()



heart._append(wave2py)
