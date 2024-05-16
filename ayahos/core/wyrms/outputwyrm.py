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
import logging
import pandas as pd
from ayahos.core.wyrms.methodwyrm import MethodWyrm, add_class_name_to_docstring
from ayahos.core.stream.dictstream import DictStream

Logger = logging.getLogger(__name__)

###################################################################################
# METHOD WYRM CLASS DEFINITION - FOR EXECUTING CLASS METHODS IN A PULSED MANNER ###
###################################################################################

# @add_class_name_to_docstring
class OutputWyrm(MethodWyrm):
    """A child class of MethodWyrm that orchestrates execution of a class method for
    input data objects and captures their standard output in the OutputWyrm.output
    attribute
    """    
    def __init__(
            self,
            pclass=DictStream,
            oclass=pd.DataFrame,
            pmethod='prediction_trigger_report',
            pkwargs={'thresh': 0.1, 'blinding': (500,500),
                     'stats_pad': 20,
                     'extra_quantiles':[0.025, 0.159, 0.25, 0.75, 0.841, 0.975]},
            max_pulse_size=10000
            ):
        """Create an ayahos.core.wyrms.outputwyrm.OutputWyrm object

        Default parameters are provided for conducting threshold-based triggering and statistical
        assessments of triggering features from ML prediction windows. 
        
        A simpler example for creating copies of DictStreams at a rate of <= 20 per pulse

        owyrm_copy = OutputWyrm(
            pclass=DictStream,
            oclass=DictStream,
            pmethod='copy',
            pkwargs={},
            max_pulse_size=20)
        
        :param pclass: processing class expected for input objects, defaults to DictStream
        :type pclass: type, optional
        :param oclass: output class type expected for objects appended to the output attribute, defaults to pd.DataFrame
        :type oclass: type, optional
        :param pmethod: name of class method for `pclass` to execute, defaults to 'prediction_trigger_report'
        :type pmethod: str, optional
        :param pkwargs: key word arguments to pass as `pclass.pmethod(**kwargs)`, defaults to {'thresh': 0.1, 'blinding': (500,500), 'stats_pad': 20, 'extra_quantiles':[0.025, 0.159, 0.25, 0.75, 0.841, 0.975]}
        :type pkwargs: dict, optional
        :param max_pulse_size: maximum number of input objects to process per pulse, defaults to 10000
        :type max_pulse_size: int, optional
        """        
        super().__init__(pclass=pclass, pmethod=pmethod, pkwargs=pkwargs, max_pulse_size=max_pulse_size)
        if not isinstance(oclass, type):
            raise ValueError
        else:
            self.oclass = oclass
    
    def _unit_process(self, obj):
        """_unit_process for OutputWyrm

        Serves as a polymorphic method call in the inherited Wyrm().pulse() method

        This method appends the stdout of the pclass.pmethod(**kwargs) to
        the output attribute, as opposed to the core_process

        :param obj: input object to act upon
        :type obj: varies
        :return unit_out: output of obj.pmethod(**pkwargs)
        :rtype: varies, must match self.oclass
        """
        unit_out = getattr(obj, self.pmethod)(**self.pkwargs)
        return unit_out
    
    def _capture_unit_out(self, unit_out):
        """_capture_unit_out for OutputWyrm

        Use MethodWyrm's _capture_unit_out method to capture
        unit_out if unit_out's type matches self.oclass

        :param unit_out: obj.pmethod(**pkwargs) output
        :type unit_out: rtype of obj.pmethod(**pkwargs) output
        :return status: continue iterating in pulse?
        :rtype status: bool
        :raises TypeError: type(unit_out) != self.oclass
        """        
        if isinstance(unit_out, self.oclass):
            status = super()._capture_unit_out(unit_out)
            return status
        else:
            Logger.critical(f'unit_out type mismatch {self.oclass} != {type(unit_out)}')
            raise TypeError
        

    def __str__(self):
        rstr = f'{self.__class__.__name__}\n'
        rstr += f'{self.pclass.__name__}.{self.pmethod} --> {self.oclass.__name__}'
        return rstr