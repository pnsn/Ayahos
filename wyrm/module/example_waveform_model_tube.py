import os, sys, logging
import seisbench.models as sbm

sys.path.append(os.path.join('..','..'))
import wyrm.core.coordinate as coo
import wyrm.core.process as pro
from wyrm.core.data import InstrumentWindow, BufferTree, TraceBuffer


## MORE OR LESS DIRECTLY FROM THE PYTHON 3.12 LOGGING COOKBOOK ##
# Initialize Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create file handler for detailed debugging information
fh = logging.FileHandler('waveform_model_tube.log')
# Create console handler for error and higher priority messages
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# Create formatter and add to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add handlers to logger
logger.addHandler(fh)
logger.addHandler(ch)


# Initailize model
model = sbm.EQTransformer()
logger.info(f'created sbm.{model.name} object')
# List desired weights
weight_names = ['pnw','instance','stead','ethz','geofon','iquique','scedc','neic']
# weight_names = model.list_pretrained()
# Ensure all models are loaded
passed_wn = []
for _wn in weight_names:
    logger.debug(f'attempting to load model weight {_wn}')
    try:
        model.from_pretrained(_wn)
        passed_wn.append(_wn)
        logger.debug('successful weight load')
    except ValueError:
        logger.warning(f'model weight {_wn} not available')

# Do a little updating to define windowing
t_sr = 100.     # target samplign rate [sps]
t_over = 1000   # target overap [samples]
t_blind = 500   # target blinding (symmetrical) [samples]

if t_over < t_blind *2:
    logger.warning('selected target overlap and blinding will result in gaps')
    # raise ValueError('selected target overlap and blinding will result in gaps - invalid')

nblind = model._annotate_args['blinding'][1]
if nblind != (t_blind, t_blind):
    model._annotate_args['blinding'][1] = (t_blind, t_blind)

if model._annotate_args['overlap'][1] != t_over:
    model._annotate_args['overlap'] = (model._annotate_args['overlap'][0], t_over)

if model.sampling_rate != t_sr:
    model.samping_rate = t_sr


####################
# Initialize Wyrms #
####################
# print('INITIALIZING WYRMS')
logging.info('initializing component wyrms')
DEBUG = False
live = True
# windowwyrm - window_ð - wind_d
wind_d = pro.WindowWyrm(
    code_map={'Z': 'Z3', 'N': 'N1', 'E': 'E2'},
    completeness={'Z': 0.95, 'N': 0.95, 'E' :0.95},
    missing_component_rule='clonehz',
    model_name=model.name,
    target_samprate = t_sr,
    target_overlap=t_over,
    target_blinding=t_blind,
    target_order=model.component_order,
    max_pulse_size=5,
    debug = DEBUG
)
logging.info(f'WindowWyrm initialized with debugging set as {wind_d.debug}')

# procwyrm - proc_ð - wind_d
proc_d = pro.ProcWyrm(
    class_type=InstrumentWindow,
    class_method_list=['.split_window()',
                       '.detrend("demean")',
                       f'.resample({t_sr})',
                       '.taper(None, max_length=0.06, side="both")',
                       '.merge_window()',
                       '.sync_window_timing()',
                       '.trim_window()',
                       '.fill_window_gaps(fill_value=0.)',
                       '.normalize_window()',
                       '.to_pwind(ascopy=False)'],
    max_pulse_size=10000,
    debug=DEBUG
)

logging.info(f'ProcWyrm initialized with debugging set as {proc_d.debug}')

# wfpredictionwyrm - wfml_ð - wfml_d
wfm_d = pro.WaveformModelWyrm(
    model=model,
    weight_names=weight_names,
    devicetype='cpu',
    max_samples=model.in_samples*3,
    stacking_method='avg',
    max_pulse_size=1000,
    debug=DEBUG
)
logging.info(f'ProcWyrm initialized with debugging set as {wfm_d.debug}')

# print('WaveformModel_ð')
# tubewyrm - tube_ð - tube_d
tube_d = coo.TubeWyrm(
    {'window': wind_d,
     'process': proc_d,
     'predict': wfm_d},
    wait_sec=0.,
    max_pulse_size=1,
    debug=True
)
print('Tube_ð')

def pulse_tubewyrm(x):
    """
    Conduct a pulse of this tubewyrm on an input BufferTree with TraceBuffer buds

    :: INPUTS ::
    :param x: [wyrm.core.data.BufferTree] tree terminating (budding) in 
                wyrm.core.data.TraceBuffer objects
    """
    if not isinstance(x, BufferTree):
        raise TypeError('input x is not type wyrm.core.data.BufferTree')
    elif x.buff_class != TraceBuffer:
        raise TypeError('input x does not bud (terminate) in TraceBuffer objects')
    else:
        y = tube_d.pulse(x)
        return y


# EXAMPLE
from obspy import read
print('loading example data from M4.3 near Port Townsend, WA')
# Load data from mSEED
st = read('../../example/uw61965081/bulk.mseed')
# Initialize BufferTree
tree = BufferTree(buff_class=TraceBuffer, max_length=180)
# Append a chunk of stream to buffer tree
tree.append_stream(st)

# DISSECTED TUBEWYRM VERSION
if not live:
    print('starting dissected version')
    x = wind_d.pulse(tree)
    print('windoing done')
    y = proc_d.pulse(x.copy())
    print('processing done')
    z = wfm_d.pulse(y.copy())
    print('prediciton done')
    print('DISSECTED VERSION COMPLETE')
else:
    print('STARTING LIVE TUBEWYRM VERSION')
    # LIVING TUBEWYRM VERSION
    tube_out = pulse_tubewyrm(tree)
    print("LIVE VERSION COMPLETE")
