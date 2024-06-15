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
    A module that calls breakpoints within :meth:`~PULSED.module.util.BreakpointMod.pulse`
    at the :meth:`~PULSED.module.util.BreakpointMod._mesaure_input_size` subroutine call 
    (before the iteration loop in pulse),
    
    This is intended to provide inspection points within an operational
    PulsedMod_EW instance for debugging and development purposes.
    """    
    def __init__(self, max_pulse_size=1, meta_memory=3600, max_output_size=1e10, report_period=False):
        super().__init__(max_pulse_size=max_pulse_size,
                         max_output_size=max_output_size,
                         meta_memory=meta_memory,
                         report_period=False)
        
    def _measure_input_size(self, input):
        input_size = super()._measure_input_size(input)
        if input_size > 0:
            breakpoint()
        return input_size
    