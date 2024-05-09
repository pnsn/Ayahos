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

import time
from collections import deque
import pandas as pd
from wyrm.core.wyrm import Wyrm
from wyrm.core.wyrmstream import WyrmStream
from wyrm.streaming.windowstream import WindowStream

###################################################################################
# METHOD WYRM CLASS DEFINITION - FOR EXECUTING CLASS METHODS IN A PULSED MANNER ###
###################################################################################

class MethodWyrm(Wyrm):
    """
    A submodule for applying a class method with specified key-word arguments to objects
    sourced from an input deque and passed to an output deque (self.queue) following processing.

    NOTE: This submodule assumes that applied class methods apply in-place alterations on data

    """
    def __init__(
        self,
        pclass=WindowStream,
        pmethod="filter",
        pkwargs={'type': 'bandpass',
                 'freqmin': 1,
                 'freqmax': 45},
        timestamp = False,
        timestamp_method=None,
        max_pulse_size=10000,
        debug=False,
        ):
        """
        Initialize a MethodWyrm object

        :: INPUTS ::
        :param pclass: [type] class defining object (ComponentStream, MLStream, etc.)
        :param pmethod: [str] name of class method to apply 
                            NOTE: sanity checks are applied to ensure that pmethod is in the
                                attributes and methods associated with pclass
        :param pkwargs: [dict] dictionary of key-word arguments (and positional arguments
                                stated as key-word arguments) for pclass.pmethod(**pkwargs)
                            NOTE: only sanity check applied is that pkwargs is type dict.
                                Users should refer to the documentation of their intended
                                pclass.pmethod() to ensure keys and values are compatable.
        :param max_pulse_size: [int] positive valued maximum number of objects to process
                                during a call of MethodWyrm.pulse()
        :param debug: [bool] should this Wyrm be run in debug mode?

        """

        # Initialize/inherit from Wyrm
        super().__init__(timestamp=timestamp, timestamp_method=timestamp_method, max_pulse_size=max_pulse_size, debug=debug)
        # pclass compatability checks
        if not isinstance(pclass,type):
            raise TypeError('pclass must be a class defining object (type "type")')
        else:
            self.pclass = pclass
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

    def pulse(self, x):
        """
        Execute a pulse wherein items are popleft'd off input deque `x`,
        checked if they are type `pclass`, have `pmethod(**pkwargs)` applied,
        and are appended to deque `self.queue`. Items popped off `x` that
        are not type pclass are reappended to `x`.

        Early stopping is triggered if `x` reaches 0 elements or the number of
        iterations equals the initial len(x)

        :: INPUT ::
        :param x: [deque] of [pclass (ideally)]

        :: OUTPUT ::
        :return y: [deque] access to the objects in self.queue
        """
        if not isinstance(x, deque):
            raise TypeError
        qlen = len(x)
        for _i in range(self.max_pulse_size):
            # Early stopping if all items have been assessed
            if _i - 1 > qlen:
                break
            # Early stopping if input deque is exhausted
            if len(x) == 0:
                break
            
            _x = x.popleft()
            if not isinstance(_x, self.pclass):
                x.append(_x)
            else:
                if self._timestamp:
                    _x.stats.processing.append(['MethodWyrm',self.pmethod, 'start', time.time()])
                getattr(_x, self.pmethod)(**self.pkwargs);
                # For objects with a stats.processing attribute, append processing info
                # if 'stats' in dir(_x):
                #     if 'processing' in dir(_x.stats):
                #         _x.stats.processing.append(
                #             [time.time(),
                #              'Wyrm 0.0.0',
                #              'MethodWyrm',
                #              self.pmethod,
                #              f'({self.pkwargs})'])
                if self._timestamp:
                    _x.stats.processing.append(['MethodWyrm',self.pmethod, 'end', time.time()])
                self.queue.append(_x)
        y = self.queue
        return y
    
    def __str__(self):
        rstr = f'wyrm.core.process.MethodWyrm(pclass={self.pclass}, '
        rstr += f'pmethod={self.pmethod}, pkwargs={self.pkwargs}, '
        rstr += f'max_pulse_size={self.max_pulse_size}, '
        rstr += f'debug={self.debug})'
        return rstr


class OutputWyrm(MethodWyrm):
    def __init__(
            self,
            pclass=WyrmStream,
            oclass=pd.DataFrame,
            pmethod='prediction_trigger_report',
            pkwargs={'thresh': 0.1, 'blinding': (500,500),
                     'stats_pad': 20,
                     'extra_quantiles':[0.025, 0.159, 0.25, 0.75, 0.841, 0.975]},
            max_pulse_size=10000,
            debug=False):
        super().__init__(pclass=pclass, pmethod=pmethod, pkwargs=pkwargs, max_pulse_size=max_pulse_size, debug=debug)
        if not isinstance(oclass, type):
            raise ValueError
        else:
            self.oclass = oclass
    
    def pulse(self, x):
        if not isinstance(x, deque):
            raise TypeError
        qlen = len(x)
        for _i in range(self.max_pulse_size):
            if _i - 1 > qlen:
                break
            if len(x) == 0:
                break
            _x = x.popleft()
            if not isinstance(_x, self.pclass):
                x.append(_x)
            else:
                _y = getattr(_x, self.pmethod)(**self.pkwargs)
                if isinstance(_y, self.oclass):
                    self.queue.append(_y)
        y = self.queue
        return y

    def __str__(self):
        rstr = f'wyrm.core.process.OutputWyrm(pclass={self.pclass}, '
        rstr += f'oclass={self.oclass}, '
        rstr += f'pmethod={self.pmethod}, pkwargs={self.pkwargs}, '
        rstr += f'max_pulse_size={self.max_pulse_size}, '
        rstr += f'debug={self.debug})'
        return rstr