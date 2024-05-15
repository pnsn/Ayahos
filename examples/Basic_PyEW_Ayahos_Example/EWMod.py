import logging, torch
import seisbench.models as sbm
from ayahos.core.wyrms.heartwyrm import HeartWyrm
from ayahos.core.wyrms.ringwyrm import RingWyrm
from ayahos.core.wyrms.bufferwyrm import BufferWyrm
from ayahos.core.wyrms.windowwyrm import WindowWyrm
from ayahos.core.wyrms.methodwyrm import MethodWyrm
from ayahos.core.wyrms.mldetectwyrm import MLDetectWyrm
from ayahos.core.stream.windowstream import WindowStream


# Set Up Logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger('EWMod')
# Limit CPU resources a little
torch.set_num_threads(10)

heartwyrm = HeartWyrm(ew_env_file='/usr/local/earthworm/memphis/environment/ew_macox.bash',
                      default_ring_id=1000,
                      module_id=200,
                      installation_id=6,
                      heartbeat_period=15,
                      conn_dict={'PICK': 1005},
                      wait_sec=0.001)

iringwyrm = RingWyrm(module = heartwyrm.module,
                     conn_id = heartwyrm.connections['DEFAULT'][0],
                     pulse_method='get_wave',
                     msg_type=19,
                     max_pulse_size=1000)

oringwyrm = RingWyrm(module = heartwyrm.module,
                     conn_id = heartwyrm.connections['PICK'][0],
                     pulse_method='put_wave',
                     msg_type=19,
                     max_pulse_size=1000)

buffwyrm = BufferWyrm()
windwyrm = WindowWyrm(model_name="PhaseNet",
                          reference_npts=3001,
                          reference_overlap=500,
                          max_pulse_size=1)
mwyrm_gaps = MethodWyrm(
    pclass=WindowStream,
    pmethod='treat_gaps',
    pkwargs={}
)
mwyrm_sync = MethodWyrm(
    pclass=WindowStream,
    pmethod='sync_to_reference',
    pkwargs={'fill_value': 0}
)
mwyrm_fill = MethodWyrm(
    pclass=WindowStream,
    pmethod='apply_fill_rule',
    pkwargs={'rule': 'zeros'}
)

mwyrm_norm = MethodWyrm(
    pclass=WindowStream,
    pmethod='normalize_traces',
    pkwargs={'norm_type': 'std'}
)

mldetwyrm = MLDetectWyrm(
    model=sbm.PhaseNet(),
    weight_names=['instance','stead'],
    devicetype='mps',
    compiled=False,
    max_pulse_size=256
)
# string together input and output rings in heartwyrm
heartwyrm.update({'iring': iringwyrm,
                  'buffer': buffwyrm,
                  'window': windwyrm,
                  'gaps': mwyrm_gaps,
                  'sync': mwyrm_sync,
                  'fill': mwyrm_fill,
                  'norm': mwyrm_norm,
                  'detect': mldetwyrm})
# Run module
heartwyrm.run()