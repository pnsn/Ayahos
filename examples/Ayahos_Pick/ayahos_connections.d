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
conn_dict: {"PICK":%(Common: PICK_RING), "TMP":1010}

[iringwyrm]
CLASS: ayahos.core.wyrms.ringwyrm.ringwyrm




[wyrm_dict]

