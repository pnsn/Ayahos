from obspy import read, UTCDateTime
from importlib import reload
import seisbench.models as sbm
from wyrm.wyrms.seisbenchwyrm import WaveformModelWyrm
from wyrm.wyrms.windowwyrm import WindowWyrm
from wyrm.structures.rtinststream import RtInstStream
from tqdm import tqdm
import wyrm.structures.rtpredbuff as rtp
import numpy as np
import matplotlib.pyplot as plt

# Initialize model
model = sbm.EQTransformer()
# load data from M4.3 near Port Townsend, 2023
print('loading data')
st = read('example/uw61965081/bulk.mseed')
# Convert stream into inststream
rtis = RtInstStream(max_length=300).append(st)
# Initialize wyrms
windwyrm = WindowWyrm(fill_value=0.).set_windowing_from_seisbench(model)
sbwyrm = WaveformModelWyrm(model, ['pnw'], devicetype='mps', max_pulse_size=10000)
# Pulse windwyrm
print('windowing data')
y = windwyrm.pulse(rtis)
# preprocess windows
print('preprocessing data')
for _y in tqdm(y):
    _y._preproc_example()
# Pulse seisbenchwyrm
print('running predictions')
z = sbwyrm.pulse(y)

# Get example data for stacking
zz = z['CC.ARAT..BH']
mm = zz['metadata']
pp = zz['pnw']