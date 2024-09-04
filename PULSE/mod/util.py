from PULSE.mod.base import BaseMod

class EchoMod(BaseMod):
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
    

class BreakpointMod(BaseMod):
    """
    A module that calls breakpoints within :meth:`~PULSE.mod.util.BreakpointMod.pulse`
    at the :meth:`~PULSE.mod.util.BreakpointMod._mesaure_input_size` subroutine call 
    (before the iteration loop in pulse),
    
    This is intended to provide inspection points within an operational
    PulsedMod_EW instance for debugging and development purposes.
    """    
    def __init__(self, max_pulse_size=1, meta_memory=3600, max_output_size=1e10, report_period=False):
        super().__init__(max_pulse_size=max_pulse_size,
                         max_output_size=max_output_size,
                         meta_memory=meta_memory,
                         report_period=report_period)
        
    def _measure_input_size(self, input):
        input_size = super()._measure_input_size(input)
        if input_size > 0:
            breakpoint()
        return input_size
    

class LogGateMod(BaseMod):
    def __init__(
            self,
            max_pulse_size=1000,
            min_pulse_size=10,
            log_base=2,
            meta_memory=60,
            max_output_size=10000,
            report_period=False):
        super().__init__(max_pulse_size=max_pulse_size,
                         max_output_size=max_output_size,
                         meta_memory=meta_memory,
                         report_period=report_period)
        self.bounding_max = self.max_pulse_size


# class RCGateMod(BaseMod):
#     """Use a regularized coulomb relationship for dynamically scaling
#     the max_pulse_size of this Mod

#     See Joughin, Smith, and Schoof (2019, GRL)
#     replace :math:`\\tau_b` with max_pulse_size
#     replace :math:`u_b` with len(self.output) = :math:`O`

#     max_pulse_size = -C * \left(\frac{|O|}{|O| + O_0}\right)^{1/m} \frac{O}{|O|}

#     Parameters to assign
#     C - prefactor
#     m - exponent
#     O_0 - reference output size

#     :param BaseMod: _description_
#     :type BaseMod: _type_
#     """
#     def __init__(self):


    