import time, tqdm
quicklog = {'start': time.time()}
import numpy as np
import obspy, os, sys, pandas
sys.path.append(os.path.join('..','..'))
import seisbench.models as sbm
import wyrm.data.dictstream as ds
import wyrm.data.componentstream as cs
import wyrm.core.coordinate as coor
import wyrm.core.process as proc 
from wyrm.util.time import unix_to_UTCDateTime
from collections import deque
import matplotlib.pyplot as plt

TICK = time.time()


# reference_sampling_rate 
RSR = 100.
#reference_channel_fill_rule
RCFR= 'zeros'
# Reference component
RCOMP = 'Z'
# Initialize Standard Processing Elements
treat_gap_kwargs = {} # see ComponentStream.treat_gaps() and MLTrace.treat_gaps() for defaults
                      # Essentially, filter 1-45 Hz, linear detrend, resample to 100 sps
# Initialize main pre-processing MethodWyrm objects (these can be cloned for multiple tubes)



# Get model architectures loaded and set lists of pretrained weights
common_pt = ['stead','instance','iquique','lendb']
EQT = sbm.EQTransformer()
EQT_list = common_pt + ['pnw']
EQT_aliases = {'Z':'Z3','N':'N1','E':'E2'}
PN = sbm.PhaseNet()
PN_aliases = {'Z':'Z3','N':'N1','E':'E2'}
PN_list = common_pt + ['diting']

## INITIALIZE WYRMS

## GENERIC PROCESSING STEPS
# For treating gappy data
mwyrm_gaps = proc.MethodWyrm(
    pclass=cs.ComponentStream,
    pmethod='treat_gaps',
    pkwargs={})

# For synchronizing temporal sampling and windowing
mwyrm_sync = proc.MethodWyrm(
    pclass=cs.ComponentStream,
    pmethod='sync_to_reference',
    pkwargs={'fill_value': 0})

# For filling data out to 3-C from non-3-C data (either missing channels, or 1C instruments)
mwyrm_fill = proc.MethodWyrm(
    pclass=cs.ComponentStream,
    pmethod='apply_fill_rule',
    pkwargs={'rule': RCFR, 'ref_component': RCOMP})

## EQTransformer SPECIFIC WYRMS

# Initialize model specific normalization MethodWyrms
mwyrm_normEQT = proc.MethodWyrm(
    pclass=cs.ComponentStream,
    pmethod='normalize_traces',
    pkwargs={'norm_type': 'peak'}
)

# Initialize WindowWyrm elements
windwyrmEQT = proc.WindowWyrm(
    component_aliases=EQT_aliases,
    model_name='EQTransformer',
    reference_sampling_rate=RSR,
    reference_npts=6000,
    reference_overlap=1800,
    max_pulse_size=1)

quicklog.update({'compose non-prediction proc wyrms': time.time()})

# Initialize PredictionWyrm elements
predwyrmEQT = proc.PredictionWyrm(
    model=EQT,
    weight_names=EQT_list,
    devicetype='mps',
    compiled=False,
    max_pulse_size=10000,
    debug=True)

# Compose EQT processing TubeWyrm
tubewyrmEQT = coor.TubeWyrm(
    max_pulse_size=1,
    debug=True,
    wyrm_dict= {'window': windwyrmEQT,
                'gaps': mwyrm_gaps.copy(),
                'sync': mwyrm_sync.copy(),
                'norm': mwyrm_normEQT,
                'fill': mwyrm_fill.copy(),
                'predict': predwyrmEQT})
    
## PHASENET SPECIFIC WYRMS

windwyrmPN = proc.WindowWyrm(
    component_aliases=PN_aliases,
    model_name='PhaseNet',
    reference_sampling_rate=100.,
    reference_npts=3001,
    reference_overlap=900,
    max_pulse_size=1)


mwyrm_normPN = proc.MethodWyrm(
    pclass=cs.ComponentStream,
    pmethod='normalize_traces',
    pkwargs={'norm_type': 'std'}
)

predwyrmPN = proc.PredictionWyrm(
    model=PN,
    weight_names=PN_list,
    devicetype='mps',
    compiled=False,
    max_pulse_size=10000,
    debug=True)

# Copy/Update to create PhaseNet processing TubeWyrm
tubewyrmPN = tubewyrmEQT.copy().update({'window': windwyrmPN,
                                        'norm': mwyrm_normPN,
                                        'predict': predwyrmPN})
tubewyrmPN.max_pulse_size=10

# # Compose CanWyrm to host multiple processing lines
canwyrm = coor.CanWyrm(wyrm_dict={'EQTransformer': tubewyrmEQT,
                             'PhaseNet': tubewyrmPN},
                  wait_sec=0,
                  max_pulse_size=10,
                  debug=False)




ROOT = os.path.join('..','..','example')
HDD_ROOT = os.path.join('/Volumes','TheWall','PNSN_miniDB','data')
WF_IN_ROOT = os.path.join(HDD_ROOT,'waveforms','uw{EVID}')
WF_OUT_ROOT = os.path.join(HDD_ROOT,'avg_stack_preds','uw{EVID}')
# Load catalog picks
pick_df = pandas.read_csv(os.path.join(ROOT,'AQMS_event_mag_phase_query_output.csv'))
# Correct times from UNIX to UTC
pick_df.arrdatetime = pick_df.arrdatetime.apply(lambda x :unix_to_UTCDateTime(x))

evid_list = pick_df.evid.value_counts().index[:20]
# Hard set event ID list to use
# evid_list = [61965081]
# runtime_tracker = []
for _i, evid in enumerate(evid_list):
    print(f'Running event uw{evid} ({_i+1}/{len(evid_list)})')
    start = time.time()
    # Subset pick_dataframe to this EVID
    pidf = pick_df[pick_df.evid == evid][['evid','orid','arid','net','sta','arrdatetime','iphase','timeres','qual','arrquality']]
    # Convert UNIX times, correcting for leap seconds
    picked_netsta = list(pidf[['net','sta']].value_counts().index)

    # Load Waveform Data
    st = obspy.read(os.path.join(WF_IN_ROOT.format(EVID=evid),'bulk.mseed'))
    # Subset waveforms
    ist = obspy.Stream()
    for tr in st:
        if (tr.stats.network, tr.stats.station) in picked_netsta:
            ist += tr
    # Convert into DictStream
    dst = ds.DictStream(ist)
    # Execute processing
    print('running CanWyrm')
    can_wyrm_out = canwyrm.pulse(dst)
    DST = ds.DictStream()
    # Merge results
    print('merging results')
    for mn, queue in can_wyrm_out.items():
        print(mn)
        for _dst in queue:
            for mltr in _dst:
                if mn == 'EQTransformer':
                    mltr.apply_blinding(blinding=(500,500))
                elif mn == 'PhaseNet':
                    mltr.apply_blinding(blinding=(0,0))
                DST.__add__(mltr, key_attr='id', method=3, fill_value=0)
    DST.write(base_path=WF_OUT_ROOT.format(EVID=evid), path_structure='{site}')

    

