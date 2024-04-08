import obspy, os, sys, glob, time
import numpy as np
sys.path.append(os.path.join('..','..'))
import seisbench.models as sbm
import wyrm.data.dictstream as ds
import wyrm.data.componentstream as cs
import wyrm.core.coordinate as coor
import wyrm.core.process as proc 
from wyrm.util.time import unix_to_UTCDateTime


# ROOT = os.path.join('..','..','example')
ROOT = os.path.join('/Users','nates','Documents','Conferences','2024_SSA','PNSN')
IN_ROOT = os.path.join(ROOT,'data','waveforms')
OUT_ROOT = os.path.join(ROOT,'processed_data','avg_pred_stacks')
EVID_DIRS =glob.glob(os.path.join(IN_ROOT,'uw*'))
EVID_DIRS.sort()


common_pt = ['stead','instance','iquique','lendb']
EQT = sbm.EQTransformer()
EQT_list = common_pt + ['pnw']
EQT_aliases = {'Z':'Z3','N':'N1','E':'E2'}
PN = sbm.PhaseNet()
PN_aliases = {'Z':'Z3','N':'N1','E':'E2'}
PN_list = common_pt + ['diting']
## (ADDITIONAL) DATA SAMPLING HYPERPARAMETERS ##
# reference_sampling_rate 
RSR = 100.
#reference_channel_fill_rule
RCFR= 'zeros'
# Reference component
RCOMP = 'Z'

# Initialize WindowWyrm elements
# PARAM NOTE: Set these to 'network' pulses to generate all windows up front
windwyrmEQT = proc.WindowWyrm(
    component_aliases=EQT_aliases,
    model_name='EQTransformer',
    reference_sampling_rate=RSR,
    reference_npts=6000,
    reference_overlap=1800,
    pulse_type='network',
    max_pulse_size=10,
    debug=False)

windwyrmPN = proc.WindowWyrm(
    component_aliases=PN_aliases,
    model_name='PhaseNet',
    reference_sampling_rate=RSR,
    reference_npts=3001,
    reference_overlap=900,
    pulse_type='network',
    max_pulse_size=20,
    debug=False)


# Initialize main pre-processing MethodWyrm objects (these can be cloned for multiple tubes)
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
    pkwargs={'rule': RCFR})

# Initialize model specific normalization MethodWyrms
mwyrm_normEQT = proc.MethodWyrm(
    pclass=cs.ComponentStream,
    pmethod='normalize_traces',
    pkwargs={'norm_type': 'peak'}
)

mwyrm_normPN = proc.MethodWyrm(
    pclass=cs.ComponentStream,
    pmethod='normalize_traces',
    pkwargs={'norm_type': 'std'}
)

# Initialize PredictionWyrm elements
predwyrmEQT = proc.PredictionWyrm(
    model=EQT,
    weight_names=EQT_list,
    devicetype='mps',
    compiled=False,
    max_pulse_size=1000,
    debug=False)

predwyrmPN = proc.PredictionWyrm(
    model=PN,
    weight_names=PN_list,
    devicetype='mps',
    compiled=False,
    max_pulse_size=1000,
    debug=False)

# Initialize Prediction BufferWyrm elements
pbuffwyrmEQT = coor.BufferWyrm(max_length=200,
                              restrict_past_append=True,
                              blinding=(500,500),
                              method=3,
                              max_pulse_size=10000,
                              debug=False)

pbuffwyrmPN = coor.BufferWyrm(max_length=200,
                             restrict_past_append=True,
                             blinding=(250,250),
                             method=3,
                             max_pulse_size=10000,
                             debug=False)


## GROUP INDIVIDUAL WYRMS
# Compose EQT processing TubeWyrm
tubewyrmEQT = coor.TubeWyrm(
    max_pulse_size=1,
    debug=True,
    wyrm_dict= {'window': windwyrmEQT,
                'gaps': mwyrm_gaps.copy(),
                'sync': mwyrm_sync.copy(),
                'norm': mwyrm_normEQT,
                'fill': mwyrm_fill.copy(),
                'predict': predwyrmEQT,
                'buffer': pbuffwyrmEQT})
    
# Copy/Update to create PhaseNet processing TubeWyrm
tubewyrmPN = tubewyrmEQT.copy().update({'window': windwyrmPN,
                                        'norm': mwyrm_normPN,
                                        'predict': predwyrmPN,
                                        'buffer': pbuffwyrmPN})

## GROUP TUBES
canwyrm = coor.CanWyrm(wyrm_dict={'EQTransformer': tubewyrmEQT,
                                  'PhaseNet': tubewyrmPN},
                       wait_sec=0,
                       max_pulse_size=30,
                       debug=False)

for evid_dir in [EVID_DIRS[50]]:
    print(f'=== STARTING {evid_dir} ===')
    tick = time.time()
    ## INIT ##
    # Create copy of the processing line as a full reset for each dataset
    iter_canwyrm = canwyrm.copy()

    ## PATH ##
    # Set event-specific paths & generate output directory
    _, dir = os.path.split(evid_dir)
    out_dir = os.path.join(OUT_ROOT, dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ## LOAD ##
    wffile = os.path.join(evid_dir, 'bulk.mseed')
    st = obspy.read(wffile, fmt='MSEED')
    for tr in st:
        tr.data = tr.data.astype(np.float32)
        if tr.stats.sampling_rate != round(tr.stats.sampling_rate):
            tr.resample(round(tr.stats.sampling_rate))
    # Do slight sampling rate adjustments to floating point sampling rates
    # Mainly analog stations and OBSs
    # for _tr in st:
    #     if _tr.stats.sampling_rate != round(_tr.stats.sampling_rate):
    #         _tr.resample(round(_tr.stats.sampling_rate))
    # Merge data
    # breakpoint()
    st.merge()
    # Convert to dictstream
    dst = ds.DictStream(traces=st)
    
    ## RUN ##
    can_wyrm_out = iter_canwyrm.pulse(dst)
    tock = time.time()
    print(f'processing for {evid_dir} took {tock - tick:.2f} sec')

    ## SAVE ##
for model, ml_dst_buffer in can_wyrm_out.items():
    print(f'saving {model}')
    ml_dst_buffer.write(base_path=os.path.join(out_dir, model),
                path_structure='{weight}/{site}')
    print(f'saving took {time.time() - tock: .3f}sec')


