from PULSED.module._base import _BaseMod

class EchoMod(_BaseMod):
    def __init__(self, max_pulse_size=10, meta_memory=3600, max_output_size=1e10, report_period=False):
        super().__init__(max_pulse_size=max_pulse_size,
                         max_output_size=max_output_size,
                         meta_memory=meta_memory,
                         report_period=False)
        
    def _unit_process(self, unit_input):
        self.Logger.warning(f'Object Summary\n{unit_input}')
        return unit_input