from ayahos.core.wyrms.heartwyrm import HeartWyrm
from ayahos.core.wyrms.ringwyrm import RingWyrm
import logging

# logging.basicConfig(level=logging.DEBUG,
#                     format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
# Logger = logging.getLogger('EWMod')

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
# string together input and output rings in heartwyrm
heartwyrm.update({'iring': iringwyrm, 'oring': oringwyrm})
# Run module
heartwyrm.run()