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

import pandas as pd
from ayahos.core.wyrms.methodwyrm import MethodWyrm
from ayahos.core.stream.dictstream import DictStream

###################################################################################
# METHOD WYRM CLASS DEFINITION - FOR EXECUTING CLASS METHODS IN A PULSED MANNER ###
###################################################################################


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
    
    def unit_process(self, x, i_):
        """unit_process for OutputWyrm

        Serves as a polymorphic method call in the inherited Wyrm().pulse() method

        This method appends the stdout of the pclass.pmethod(**kwargs) to
        the output attribute, as opposed to the core_process

        :param x: input collection of objects
        :type x: collections.deque of pclass-type objects
        :param i_: iteration number
        :type i_: int
        :return status: should process continue to next iteration?
        :rtype: bool
        """        
        if super()._continue_iteration(x, i_):
            _x = x.popleft()
            if isinstance(_x, self.pclass):
                _y = getattr(_x, self.pmethod)(**self.pkwargs)
                self.output.append(_y)
            else:
                x.append(_x)
            status = True
        else:
            status = False
        return status
    
    def __str__(self):
        rstr = f'{self.__class__.__name__}\n'
        rstr += f'{self.pclass.__name__}.{self.pmethod} --> {self.oclass.__name__}'
        return rstr