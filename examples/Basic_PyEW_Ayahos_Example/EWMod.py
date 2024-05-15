import seisbench.models as sbm
from ayahos.core.wyrms.heartwyrm import HeartWyrm
from ayahos.core.wyrms.ringwyrm import RingWyrm
from ayahos.core.wyrms.bufferwyrm import BufferWyrm
from ayahos.core.wyrms.windowwyrm import WindowWyrm
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger('EWMod')

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

buffwyrm = BufferWyrm()
windwyrm = WindowWyrm(model_name="PhaseNet",
                          reference_npts=3001,
                          reference_overlap=500)
# we
# string together input and output rings in heartwyrm
heartwyrm.update({'iring': iringwyrm,
                  'buffer': buffwyrm,
                  'window': windwyrm})
# Run module
heartwyrm.run()