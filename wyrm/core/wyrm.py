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
from time import time
import logging

Logger = logging.getLogger(__name__)

class Wyrm(object):
    """Fundamental base class for all other *Wyrm classes

    :return: Wyrm object
    :rtype: wyrm.core.wyrm.Wyrm
    """

    def __init__(self, max_pulse_size=None):
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
        elif isinstance(max_pulse_size, int, float):
            if 1 <= max_pulse_size:
                self.max_pulse_size = int(max_pulse_size)
            else:
                raise ValueError('max_pulse_size must be g.e. 1 ')
        else:
            raise TypeError('max_pulse_size must be NoneType, or positive int-like')


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

    def pulse(self, x=None):
        """Run a pulse where the input equals the output

        :param x: input value, defaults to None
        :type x: any, optional
        :return: output value
        :rtype: same as x
        """        
        Logger.debug('initiating pulse')
        y = x
        Logger.debug('concluding pulse')
        return y
