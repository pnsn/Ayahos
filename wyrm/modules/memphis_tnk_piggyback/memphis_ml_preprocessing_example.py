"""
:module: wyrm/modules/memphis_tnk_piggyback/ml_preprocessing_example.py
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module runs a ring-to-ring type PyEarthworm module that
    ingests wave messages from the WAVE ring, conducts station-wise
    buffering and processing of data to produce windowed data ready
    for prediction purposes in PhaseNet,
        i.e., 3000 sample numpy arrays with mean of ~0, range \in[-1, 1]
              and no NaN-/inf-/None-type entries (no gaps)
    The module then writes out LOGO (SNCL) keyed wave messages to an
    "AI_RING"

    This module is meant to run alongside a sligt adaptation of the 
    Memphis Earthworm TankPlayer example referenced in the Earthworm 
    GitLab repository. Interested users will need to:
     1)  include this module in <FILL IN POINTER HERE> with necessary path
         definitions to 


Schematic
         | < buff, window >|
{ O-v^-> | <      ...     >| -v^->O }
         | < buff, window >|
"""

from wyrm.core.base_wyrms import TubeWyrm, CanWyrm
from wyrm.core.processes import BuffWyrm, WindowWyrm
from wyrm.core.io import Wave2PyWyrm, Py2WaveWyrm
from wyrm.core.heartwyrm import HeartWyrm
from collections import deque

import pandas as pd
import os


EW_HOME = os.path.join('/usr', 'local', 'earthworm')
STA = os.path.join(EW_HOME, 'memphis', 'params', 'memphis_trig.sta')
# Use trig.sta file to get station list & parse with Pandas
df = pd.read_table(STA,
                   delim_whitespace=True,
                   names=['type', 'sta', 'cha', 'net', 'loc', 'X'])
# Get Unique station/network names
stanet = df[['sta', 'net']].value_counts().index.values

# Initialize HeartWyrm
heart = HeartWyrm(pulse_rate=1, DR_ID=1000, MOD_ID=200, INST_ID=6, HB_PERIOD=30, debug=False)
# Initialize EW Module with user-input check enabled
heart.initialize_module(user_check=True)
# Add connection to WAVE ring
heart.add_connection(RING_ID=1000, RING_Name='WAVE_RING')
# Add connection to AI ring
heart.add_connection(RING_ID=1100, RING_Name='AI_RING')
# Alias WAVE RING info
wr_info = heart.conn_info[heart.conn_info['RING_Name'] == 'WAVE_RING'].T
ai_info = heart.conn_info[heart.conn_info['RING_Name'] == 'AI_RING'].T


# Initialize Elements going into the HeartWyrm
wave2py = Wave2PyWyrm(module=heart.module, conn_index=wr_info.RING_ID)
heart._append(wave2py)

# Initialize canwyrm
can = CanWyrm(wyrm_queue=deque([]), output_type=deque)


# Iterate across stations to create a tubewyrm for each station

for _sta, _net in stanet:
    # initialize ith tubewyrm
    tube = TubeWyrm(wyrm_queue=deque([]))
    # initialize ith buffwyrm
    buff = BuffWyrm(station=_sta, network=_net)
    # Append buffwyrm to tubewyrm
    tube._append(buff)
    # initialize ith windowwyrm
    wind = WindowWyrm(station=_sta, network=_net,
                      wlength=3000, stride=1500,
                      taper_length=6
                      obspy_evals=['filter("bandpass", freqmin=1, freqmax=45)',
                                   'resample(100)',
                                   '***trim_window',
                                   'taper',
                                   'merge()',
                                   'detrend("linear")',
                                   'detrend("demean")',
                                   'normalize()'])
    # # initialize ith babbelwyrm
    # babbel = BabbelWyrm(input=Trace, output=WaveMsg)
    # # Append babbelwyrm to tubewyrm
    # tube._append(babbel)

    ### ASSEMBLY ###
    # Append windowwyrm to tubewyrm
    tube._append(wind)
    # append ith tubewyrm to canwyrm
    can._append(tube)
# Append canwyrm to heartwyrm
heart._append(can)

# Initialize output wyrm
py2wave = Py2WaveWyrm(module=heart.module, conn_index=ai_info.RING_ID)
# Append to heartwyrm
heart._append(py2wave)

## TODO: Create HeartWyrm.validate_pipeline() class-method
heart.validate_pipeline()
### ALLOW USER EXPLORATION
breakpoint()

### LAUNCH MODULE ###
heart.run()