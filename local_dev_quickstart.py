from obspy import read, UTCDateTime
from importlib import reload
import seisbench.models as sbm
from wyrm.wyrms.seisbenchwyrm import WaveformModelWyrm
from wyrm.wyrms.window import WindowWyrm
from wyrm.structures.rtinststream import RtInstStream
from tqdm import tqdm
import wyrm.structures.rtpredbuff as rtp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Initialize model
print('loading SeisBench Model')
model = sbm.PhaseNet()
wgt = 'diting'
# load data from M4.3 near Port Townsend, 2023
print('loading data')
st = read('example/uw61965081/bulk.mseed')
# Convert stream into inststream
print('converting to InstStream')
rtis = RtInstStream(max_length=300).append(st)
# Initialize wyrms
print('initializing wyrms')
windwyrm = WindowWyrm(fill_value=0.).set_windowing_from_seisbench(model)
sbwyrm = WaveformModelWyrm(model, [wgt], devicetype='mps', max_pulse_size=10000)
# Pulse windwyrm
print('windowing data')
y = windwyrm.pulse(rtis)
# preprocess windows
print('preprocessing data')
for _y in tqdm(y):
    _y._preproc_example()
y_backup = deepcopy(y)
# Pulse seisbenchwyrm
print('running predictions')
z = sbwyrm.pulse(y)

# Get example data for stacking
zz = z['CC.ARAT..BH']
mm = zz['metadata']
pp = zz[wgt]