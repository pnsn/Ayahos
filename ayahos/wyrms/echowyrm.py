from ayahos.wyrms.wyrm import Wyrm
import logging

Logger = logging.getLogger(__name__)

class EchoWyrm(Wyrm):

    def __init__(self, max_pulse_size=10, max_output_size=100, meta_memory=3600, report_period=False):
        super().__init__(max_pulse_size=max_pulse_size,
                         max_output_size=max_output_size,
                         meta_memory=meta_memory,
                         report_period=report_period)


    
    def _capture_unit_output(self, unit_output):
        self.output.append(unit_output)
        while len(self.output) > self.max_output_size:
            obj  = self.output.popleft()
            Logger.info(f'\n\n{obj}\n')
            self.output.popleft()