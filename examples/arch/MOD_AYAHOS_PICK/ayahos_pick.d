[Common]
INST_ID: 2
DEFAULT_RING: 1000
PICK_RING: 1005
TMP_RING: 

[ayahos]
CLASS: ayahos.core.wyrms.heartwyrm.HeartWyrm
DEFAULT_RING_ID: ${Common: DEFAULT_RING}
MODULE_ID: 200
INSTALLATION_ID: ${Common: INST_ID}
HEARTBEAT_PERIOD: 30
CONN_DICT: {"PICK":%(Common: PICK_RING), "TMP":1010}
WAIT_SEC: 0.

[iringwyrm]
CLASS: ayahos.core.wyrms.ringwyrm.RingWyrm
MODULE: ayahos.module
CONN_ID: 0
PULSE_METHOD: get_wave
MSG_TYPE: 19
MAX_PULSE_SIZE: 1000

[oringwyrm]
CLASS: ayahos.core.wyrms.ringwyrm.RingWyrm
MODULE: ayahos.module
CONN_ID: 1
PULSE_METHOD: put_wave
MSG_TYPE: 19
MAX_PULSE_SIZE: 1000


[wyrm_dict]

