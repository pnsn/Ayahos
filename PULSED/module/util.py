from PULSED.module._base import _BaseMod

class EchoMod(_BaseMod):
    """
    A module that sends the __repr__ of unit_input objects
    to logging 
    """    
    def __init__(self, max_pulse_size=1, meta_memory=3600, max_output_size=1e10, report_period=False):
        super().__init__(max_pulse_size=max_pulse_size,
                         max_output_size=max_output_size,
                         meta_memory=meta_memory,
                         report_period=False)
        
    def _unit_process(self, unit_input):
        msg = f'\nObject Summary\n{unit_input.__repr__()}'
        # msg += f'\n'
        self.Logger.warning(msg)
        return unit_input
    

class BreakpointMod(_BaseMod):
    """
    A module that calls a breakpoint for each _unit_process call
    within :meth:`~PULSED.module._base._BaseModule.pulse` 
    """    
    def __init__(self, max_pulse_size=1, meta_memory=3600, max_output_size=1e10, report_period=False):
        super().__init__(max_pulse_size=max_pulse_size,
                         max_output_size=max_output_size,
                         meta_memory=meta_memory,
                         report_period=False)
        
    def _unit_process(self, unit_input):
        breakpoint()
        return unit_input