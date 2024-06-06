from ayahos.wyrms.wyrm import Wyrm
import logging

Logger = logging.getLogger(__name__)

class EchoWyrm(Wyrm):

    def __init__(self, max_pulse_size=10000, mute_pulse_logging=True):
        super().__init__(max_pulse_size=max_pulse_size, mute_pulse_logging=mute_pulse_logging)
    
    def _unit_process(self, unit_input):
        Logger.info(f'{unit_input}')
        unit_output = unit_output
        return unit_output