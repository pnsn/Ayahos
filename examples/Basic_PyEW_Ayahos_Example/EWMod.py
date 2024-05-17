import logging, torch, os
import seisbench.models as sbm
from ayahos.core.ayahos import Ayahos
from ayahos.wyrms.ringwyrm import RingWyrm
from ayahos.wyrms.bufferwyrm import BufferWyrm
from ayahos.wyrms.windowwyrm import WindowWyrm
from ayahos.wyrms.methodwyrm import MethodWyrm
from ayahos.wyrms.sbmwyrm import SBMWyrm
from ayahos.core.windowstream import WindowStream


# Set Up Logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger('EWMod')
# Limit CPU resources a little
torch.set_num_threads(10)

ew_env_file = '/usr/local/earthworm/memphis/environment/ew_macox.bash'
# Ensure environment is sourced prior to initializing wyrms
try:
    os.environ['EW_HOME']
except KeyError:
    os.system(f'source {ew_env_file}')

# Initialize HeartWyrm (and the PyEW.EWModule therein) but don't start the module yet
ayahos = Ayahos(ew_env_file=ew_env_file,
                      default_ring_id=1000,
                      module_id=200,
                      installation_id=6,
                      heartbeat_period=15,
                      conn_dict={'PICK': 1005},
                      wait_sec=0.001)

# Initialize a EW->Py RingWyrm for waves
iringwyrm = RingWyrm(module = ayahos.module,
                     conn_id = ayahos.connections['DEFAULT'][0],
                     pulse_method='get_wave',
                     msg_type=19,
                     max_pulse_size=1000)

# Initialize a Py->EW RingWyrm for waves (currently unused)
oringwyrm = RingWyrm(module = ayahos.module,
                     conn_id = ayahos.connections['PICK'][0],
                     pulse_method='put_wave',
                     msg_type=19,
                     max_pulse_size=1000)

# Initialize a default BufferWyrm
buffwyrm = BufferWyrm()

# Initialize a WindowWyrm for PhaseNet input windows
windwyrm = WindowWyrm(model_name="PhaseNet",
                          reference_npts=3001,
                          reference_overlap=500,
                          reference_sampling_rate=100.,
                          max_pulse_size=1)

# Initialize MethodWyrms for pre-processing steps
# Handle incomplete data (compound method: filtering, resampling, tapering, gap filling/padding)
mwyrm_gaps = MethodWyrm(
    pclass=WindowStream,
    pmethod='treat_gaps',
    pkwargs={}
)
# Synchronize sampling (align data points of all traces)
mwyrm_sync = MethodWyrm(
    pclass=WindowStream,
    pmethod='sync_to_reference',
    pkwargs={'fill_value': 0}
)
# Fill insufficiently full/missing channels
mwyrm_fill = MethodWyrm(
    pclass=WindowStream,
    pmethod='apply_fill_rule',
    pkwargs={'rule': 'zeros'}
)
# Normalize trace data
mwyrm_norm = MethodWyrm(
    pclass=WindowStream,
    pmethod='normalize_traces',
    pkwargs={'norm_type': 'std'}
)

# Initialize predicting Wyrm submodule
mldetwyrm = MLDetectWyrm(
    model=sbm.PhaseNet(),
    weight_names=['instance','stead'],
    devicetype='cpu',
    compiled=False,
    max_pulse_size=256
)

# string together sub-modules in ayahos
ayahos.update({'iring': iringwyrm,
                  'buffer': buffwyrm,
                  'window': windwyrm,
                  'gaps': mwyrm_gaps,
                  'sync': mwyrm_sync,
                  'fill': mwyrm_fill,
                  'norm': mwyrm_norm,
                  'detect': mldetwyrm})

# Run module!
ayahos.run()