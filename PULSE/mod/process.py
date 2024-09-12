"""
:module: PULSE.mod.unit.process
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This contains class definitions for data processing unit modules that either
    conduct in-place alterations to data objects or capture outputs of data objects'
    class-method outputs

# TODO: Need to write pulse subroutines with simplified structure


Classes
-------
:class:`~PULSE.mod.process.InPlaceMod`
:class:`~PULSE.mod.process.OutputMod`
"""

import logging, sys
from collections import deque
from obspy import UTCDateTime
from PULSE.data.mltrace import MLTrace
from PULSE.data.dictstream import DictStream
from PULSE.data.window import Window
from PULSE.mod.base import BaseMod

Logger = logging.getLogger(__name__)

class InPlaceMod(BaseMod):
    """
    A submodule for applying a class method with specified key-word arguments to objects
    sourced from an input deque and passed to an output deque (self.output) following processing.

    Initialization Notes
    - sanity checks are applied to ensure that pmethod is in the attributes and methods associated with pclass
    - the only sanity check applied is that pkwargs is type dict. 
    Users should refer to the documentation of their intended pclass.pmethod() to ensure keys and values are compatable.
    
    
    """
    def __init__(
        self,
        pclass,
        pmethod,
        pkwargs,
        max_pulse_size=10000,
        meta_memory=3600,
        report_period=False,
        max_output_size=1e9
        ):
        """
        Initialize a InPlaceMod object

        :: INPUTS ::
        :param pclass: full import path and name of class this InPlaceMod will operate on
        :type pclass: str, e.g., "PULSE.data.window.Window"
        :param pmethod: name of class method to apply to unit_input objects
        :type pmethod: str, e.g., "filter"
        :param pkwargs: key-word arguments (and positional arguments stated as key-word arguments) for pclass.pmethod(**pkwargs)
        :type pkwargs: dict, e.g., {"type": "bandpass", "freqmin": 1, "freqmax": 45}
        :param max_pulse_size: maximum number of iterations to conduct in a pulse, defaults to 10000.
        :type max_pulse_size: int, optional
        """

        # Initialize/inherit from BaseMod
        super().__init__(max_pulse_size=max_pulse_size,
                         meta_memory=meta_memory,
                         report_period=report_period,
                         max_output_size=max_output_size)

        # pclass compatability checks
        self.pclass = self.import_class(pclass)
        # pmethod compatability checks
        if pmethod not in [func for func in dir(self.pclass) if callable(getattr(self.pclass, func))]:
            raise ValueError(f'pmethod "{pmethod}" is not defined in {self.pclass} properties or methods')
        else:
            self.pmethod = pmethod
        # pkwargs compatability checks
        if isinstance(pkwargs, dict):
            self.pkwargs = pkwargs
        else:
            raise TypeError
        # initialize output queue
        self.queue = deque()

    # def import_class(self, class_path_str):
    #     self.pclass = super().import_class(class_path_str)
        
    # Inherited from BaseMod
    # def _continue_iteration()
    # def _capture_unit_out()
        
    def _unit_input_from_input(self, input):
        # Use checks from BaseMod on input
        unit_input = super()._unit_input_from_input(input)
        # Then apply checks from pclass
        if isinstance(unit_input, self.pclass):
            return unit_input
        else:
            self.Logger.critical(f'object popped from input mismatch {self.pclass} != {type(unit_input)}')
            raise TypeError
        
    def _unit_process(self, unit_input):
        """unit_process for InPlaceMod

        Check if the input deque and iteration number
        meet iteration continuation criteria inherited from BaseMod

        Check if the next object popleft'd off `x` is type self.pclass
        
            Mismatch: send object back to `x` with append()

            Match: Execute the in-place processing and append to InPlaceMod.output

        :param unit_input: object to be modified with self.pmethod(**self.pkwargs)
        :type unit_input: self.pclass
        :returns:
         - **unit_output** (*self.pclass*) -- modified object
        """ 
        try:
            getattr(unit_input, self.pmethod)(**self.pkwargs)
            unit_output = unit_input
        except:
            self.Logger.warning(f'{self.pmethod} did not work on unit input - skipping')
            unit_output = None
            return unit_output
        if self.pclass in [MLTrace, DictStream, Window]:
            unit_input.stats.processing.append([self.__name__(full=False), self.pmethod, UTCDateTime()])
        return unit_output
    

class OutputMod(InPlaceMod):
    """A child class of InPlaceMod that orchestrates execution of a class method for
    input data objects and captures their outputs in the OutputMod.output attribute

    A simple example for creating copies of DictStreams at a rate of <= 20 per pulse

    owyrm_copy = OutputMod(
        pclass=DictStream,
        oclass=DictStream,
        pmethod='copy',
        pkwargs={},
        max_pulse_size=20)
    
    :param pclass: processing class expected for input objects
    :type pclass: type
    :param oclass: output class type expected for objects appended to the output attribute
    :type oclass: type
    :param pmethod: name of class method for `pclass` to execute, defaults to 'prediction_trigger_report'
    :type pmethod: str
    :param pkwargs: key word arguments to pass as `pclass.pmethod(**pkwargs)`, defaults to {}.
    :type pkwargs: dict, optional
    :param max_pulse_size: maximum number of input objects to process per pulse, defaults to 10000
    :type max_pulse_size: int, optional 
    """    
    def __init__(
            self,
            pclass,
            oclass,
            pmethod,
            pkwargs={},
            max_pulse_size=10000,
            meta_memory=3600,
            report_period=False,
            max_output_size=1e9
            ):
        """Initialize an OutputMod object
        
        :param pclass: processing class expected for input objects
        :type pclass: type
        :param oclass: output class type expected for objects appended to the output attribute
        :type oclass: type
        :param pmethod: name of class method for `pclass` to execute, defaults to 'prediction_trigger_report'
        :type pmethod: str
        :param pkwargs: key word arguments to pass as `pclass.pmethod(**pkwargs)`, defaults to {}.
        :type pkwargs: dict, optional
        :param max_pulse_size: maximum number of input objects to process per pulse, defaults to 10000
        :type max_pulse_size: int, optional
        """        
        super().__init__(
            pclass=pclass,
            pmethod=pmethod,
            pkwargs=pkwargs,
            max_pulse_size=max_pulse_size,
            meta_memory=meta_memory,
            report_period=report_period,
            max_output_size=max_output_size)
        
        if not isinstance(oclass, str):
            self.Logger.critical(f'oclass must be type =str. Not {type(oclass)}')
        else:
            if oclass in ['str','int','float']:
                self.oclass = eval(oclass)
            else:
                self.oclass = self.import_class(oclass)
    
    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last update with :class:`~PULSE.mod.process.OutputMod`

        Run the specified class method (and kwargs) on the unit input
        and return the output of that class method

        unit_output = getattr(unit_input, self.pmethod)(**self.pkwargs) 


        :param unit_input: input unit_inputect to act upon
        :type unit_input: varies
        :returns:
            - **unit_out** (*self.oclass*)
        :rtype: varies, must match self.oclass
        """
        try:
            unit_output = getattr(unit_input, self.pmethod)(**self.pkwargs)
        except:
            self.Logger.warning(f'{self.pmethod} was unsuccessful - skipping')
            unit_output = None
        return unit_output
    
    def _capture_unit_output(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.OutputMod.OutputMod`

        Use inherited :meth:`~ayahos.wyrms.wyrm.Wyrm._capture_unit_output` to append
        the output to self.output, if and only if unit_output's type matches self.oclass.

        :param unit_output: unit_input.pmethod(**pkwargs) output
        :type unit_output: rtype of unit_input.pmethod(**pkwargs) output
        """        
        if isinstance(unit_output, self.oclass):
            status = super()._capture_unit_output(unit_output)
            return status
        else:
            self.Logger.critical(f'unit_output type mismatch {self.oclass} != {type(unit_output)}')
            sys.exit(1)
        



    def __str__(self):
        rstr = f'{self.__class__.__name__}\n'
        rstr += f'{self.pclass.__name__}.{self.pmethod} --> {self.oclass.__name__}'
        return rstr