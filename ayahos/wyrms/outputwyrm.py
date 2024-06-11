"""
:module: wyrm.processing.inplace
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module hosts class definitions for a Wyrm class that facilitates pulsed
    execution of class methods of objects that conduct in-place alterations
    to those objects' contents.

    MethodWyrm - a submodule for applying a single class method to objects presented to the
                MethodWyrm
                    PULSE
                        input: deque of objects
                        output: deque of objects
    
"""
import logging, sys
from ayahos.wyrms.methodwyrm import MethodWyrm

Logger = logging.getLogger(__name__)

class OutputWyrm(MethodWyrm):
    """A child class of MethodWyrm that orchestrates execution of a class method for
    input data objects and captures their outputs in the OutputWyrm.output attribute

    A simple example for creating copies of DictStreams at a rate of <= 20 per pulse

    owyrm_copy = OutputWyrm(
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
            report_period=None
            ):
        """Initialize an OutputWyrm object
        
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
            report_period=report_period)
        
        if not isinstance(oclass, type):
            raise ValueError
        else:
            self.oclass = oclass
    
    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last update with :class:`~ayahos.wyrms.outputwyrm.OutputWyrm`

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
            Logger.critical('Specified operation did not work - exiting')
            sys.exit(1)
        return unit_output
    
    def _capture_unit_output(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.outputwyrm.OutputWyrm`

        Use inherited :meth:`~ayahos.wyrms.wyrm.Wyrm._capture_unit_output` to append
        the output to self.output, if and only if unit_output's type matches self.oclass.

        :param unit_output: unit_input.pmethod(**pkwargs) output
        :type unit_output: rtype of unit_input.pmethod(**pkwargs) output
        """        
        if isinstance(unit_output, self.oclass):
            status = super()._capture_unit_output(unit_output)
            return status
        else:
            Logger.critical(f'unit_output type mismatch {self.oclass} != {type(unit_output)}')
            sys.exit(1)
        



    def __str__(self):
        rstr = f'{self.__class__.__name__}\n'
        rstr += f'{self.pclass.__name__}.{self.pmethod} --> {self.oclass.__name__}'
        return rstr