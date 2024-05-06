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
import wyrm.util.input as wcc
from copy import deepcopy
from time import time
import logging

logger = logging.getLogger(__name__)

class Wyrm:
    """
    Fundamental Base Class for all *Wyrm classes in this module that are
    defined by having the y = *wyrm.pulse(x) class method.

    The Wyrm base class produces an object with no attributes and placeholders
    for fundamental class-methods common to all Wyrm class objects:

    + __init__
    + __str__
    + __repr__
    + pulse

    And attributes:
    @ max_pulse_size - reference integer for maximum number of iterations for
                        the outer loop of a given pulse() method
    @ debug - bool switch for running the wyrm in debug mode (in development)
    """

    def __init__(self, timestamp=False, max_pulse_size=None, debug=False):
        """
        Initialize a Wyrm object
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info('creating an instance of Wyrm')
        # Compatability check for debug
        if not isinstance(debug, bool):
            raise TypeError("debug must be type bool")
        else:
            self.debug = debug
        
        if isinstance(timestamp, bool):
            self._timestamp = timestamp
        else:
            raise TypeError('timestamp must be type bool')

        # Compatability check for max_pulse_size
        if max_pulse_size is None:
            self.max_pulse_size = None
        elif isinstance(max_pulse_size, int, float):
            if 1 <= max_pulse_size <= 1e6:
                self.max_pulse_size = int(max_pulse_size)
            else:
                raise ValueError('max_pulse_size must be between 1 and 1000000')
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
        """
        Run a pulse with input argument and return that argument
        with check that input x is the expected _in_type
        :: INPUT ::
        :param x: [type] or [NoneType] input object x
        :param options: [kwargs] collector for addtional key word arguments
                        to pass to internal processes
        :: OUTPUT ::
        :return y: [type] or [NoneType] alias of input x
        """
        y = x
        return y
