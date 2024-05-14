"""
:submodule: wyrm.core.wyrm
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module hosts the fundamental class definition for all
    other *Wyrm classes -- "Wyrm" -- and serves as a template
    for the minimum required methods of each successor class. 
"""
from copy import deepcopy
from collections import deque
from time import time
import logging

Logger = logging.getLogger(__name__)

class Wyrm(object):
    """Fundamental base class template all other *Wyrm classes

    Has a deque attribute `output`    
    
    """

    def __init__(self, max_pulse_size=1e9):
        """Initialize a Wyrm object

        :param max_pulse_size: maximum , defaults to None
        :type max_pulse_size: _type_, optional
        :raises ValueError: for non-positive number argument for max_pulse_size
        :raises TypeError: for non-int-like or non-NoneType argument for max_pulse_size
        """
        Logger.debug('Initializing a Wyrm object')
        # Compatability check for max_pulse_size
        if max_pulse_size is None:
            self.max_pulse_size = None
        elif isinstance(max_pulse_size, (int, float)):
            if 1 <= max_pulse_size:
                self.max_pulse_size = int(max_pulse_size)
            else:
                raise ValueError('max_pulse_size must be g.e. 1 ')
        else:
            raise TypeError('max_pulse_size must be NoneType, or positive int-like')
        self.output = deque()


    def __repr__(self):
        """
        Provide a string representation string of essential user data for this Wyrm
        """
        # rstr = "~~wyrm~~\nFundamental Base Class\n...I got no legs...\n"
        rstr = f"Max Pulse Size: {self.max_pulse_size} | debug: {self.debug}"
        return rstr

    def __str__(self):
        """
        Provide a string representation of how to recreate this Wyrm
        """
        rstr = f'wyrm.wyrms.base.Wyrm(max_pulse_size={self.max_pulse_size}, debug={self.debug})'
        return rstr

    def copy(self):
        """
        Return a deepcopy of this wyrm
        """
        return deepcopy(self)

    def _core_process(self, x):
        """The core unit process for this particular wyrm
        ayahos.core.wyrms.wyrm.Wyrm

        1) popleft() an item in x
        2) set _y = _x
        3) append(_y) to self.output

        :param x: input data object source
        :type x: collections.deque of objects
        """        
        _x = x.popleft()
        _y = _x
        self.output.append(_y)
        y = self.output
        return y

    def _early_stopping(self, x, i_):
        """Early stopping criteria for this Wyrm
        ayahos.core.wyrms.wyrm.Wyrm

        Criteria:
         - len(x) = 0
         - i_ > len(x)
        
        :param x: input data object collection
        :type x: collections.deque
        :param i_: iteration number
        :type i_: int
        :return: early stopping criteria met?
        :rtype: bool
        """        
        if len(x) == 0:
            return True
        elif i_ > len(x):
            return True
        else:
            return False
        
    def pulse(self, x=deque()):
        """Run a pulse for this Wyrm for up to self.max_pulse_size items in `x`

        :param x: collection of objects to assess
        :type x: collection.deque of objects
        :return: collection of outputs in this Wyrm's self.output attribute
        :rtype: collection.deque of objects
        """        
        Logger.debug('initiating pulse')
        for i_ in range(self.max_pulse_size):
            if self._early_stopping(x, i_):
                break
            else:
                y = self._core_process(x)
        Logger.debug('concluding pulse')
        return y
