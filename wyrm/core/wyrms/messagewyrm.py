from wyrm.core.wyrms.ringwyrm import RingWyrm
from wyrm.core.
from collections import deque


class MessageWyrm(RingWyrm):

    def __init__(
            self,
            module=None,
            conn_id=0,
            msg_type=1,
            flow_type='put'
            max_pulse_size=1e6,
    ):
        super().__init(self, max_pulse_size=max_pulse_size, module=module, )