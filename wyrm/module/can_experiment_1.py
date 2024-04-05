import obspy, os, sys, pandas, glob
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
windwyrmEQT = proc.WindowWyrm(
    component_aliases=EQT_aliases,
    model_name='EQTransformer',
    reference_sampling_rate=RSR,
    reference_npts=6000,
    reference_overlap=1800,
    pulse_type='site',
    max_pulse_size=200,
    debug=False)

windwyrmPN = proc.WindowWyrm(
    component_aliases=PN_aliases,
    model_name='PhaseNet',
    reference_sampling_rate=RSR,
    reference_npts=3001,
    reference_overlap=900,
    pulse_type='site',
    max_pulse_size=200,
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
    max_pulse_size=10000,
    debug=False)

predwyrmPN = proc.PredictionWyrm(
    model=PN,
    weight_names=PN_list,
    devicetype='mps',
    compiled=False,
    max_pulse_size=10000,
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
                       debug=True)


for evid_dir in EVID_DIRS:
    # Create copy of the processing line
    iter_canwyrm = canwyrm.copy()
    # Set event-specific paths & generate output directory
    _, dir = os.path.split(evid_dir)
    out_dir = os.path.join(OUT_ROOT, dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Load waveform data
    wffile = os.path.join(evid_dir, 'bulk.mseed')
    st = obspy.read(wffile, fmt='MSEED')
    dst = ds.DictStream(traces=st)
    # Run 
    can_wyrm_out = canwyrm.pulse(dst)
    breakpoint()
    for _k, _v in can_wyrm_out.items():
        _v.write(base_path=os.path.join(out_dir, _k),
                 path_structure='{weight}/{site}')


