import os
import sys
sys.path.append(os.path.join('..','..'))
import wyrm.core.coordinate as coo
import wyrm.core.process as pro
from wyrm.core.data import InstrumentWindow, BufferTree, TraceBuffer
import seisbench.models as sbm

print('INITIALIZING SEISBENCH WAVEFORMMODEL')
# Initailize model
model = sbm.EQTransformer()
# List desired weights
weight_names = ['pnw','instance','stead']
# Do a little updating to define windowing
t_sr = 100.     # target samplign rate [sps]
t_over = 1000   # target overap [samples]
t_blind = 500   # target blinding (symmetrical) [samples]

if t_over < t_blind *2:
    raise ValueError('selected target overlap and blinding will result in gaps - invalid')

nblind = model._annotate_args['blinding'][1]
if nblind != (t_blind, t_blind):
    model._annotate_args['blinding'][1] = (t_blind, t_blind)

if model._annotate_args['overlap'][1] != t_over:
    model._annotate_args['overlap'] = (model._annotate_args['overlap'][0], t_over)

if model.sampling_rate != t_sr:
    model.samping_rate = t_sr

print('CHECKING PRETRAINED MODEL WEIGHT VALIDITY')
# Get pretrained model names for cross reference (the list_pretrained() method takes a bit of time)
# pretrains = model.list_pretrained()
# if not all(_wn in pretrains for _wn in weight_names):
#     raise ValueError(f'Not all provided weight names {weight_names} are in the pretrained model weights for {model.name}: \n {pretrains}')

# Load each model weight to ensure the relevant files are saved locally
for _wn in weight_names:
    print(f'Loading {_wn}')
    model.from_pretrained(_wn)


####################
# Initialize Wyrms #
####################
print('INITIALIZING WYRMS')
DEBUG = False

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
    max_pulse_size=1,
    debug = DEBUG
)
print('Window_ð')

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
print('Processing_ð')
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
print('WaveformModel_ð')
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
# Append stream to buffer tree
tree.append_stream(st[:60])

x = wind_d.pulse(tree)
y = proc_d.pulse(x)
z = wfm_d.pulse(y)
# # RUN PULSE
# y = pulse_tubewyrm(tree)

# # Show status after
# print(wfml_d.__repr__())

