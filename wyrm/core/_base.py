"""
:module: wyrm.core._base
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module hosts the fundamental class definition for all
    other *Wyrm classes -- "Wyrm" -- and serves as a template
    for the minimum required methods of each successor class. 
"""
import wyrm.util.compatability as wcc
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

    def __init__(self, timestamp=False, timestamp_method=None, max_pulse_size=None, debug=False):
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

        if isinstance(timestamp_method, (type(None), str)):
            self._timestamp_method=timestamp_method
        else:
            raise TypeError('timestamp_method must be NoneType or str')

        # Compatability check for max_pulse_size
        if max_pulse_size is None:
            self.max_pulse_size = None
        else:
            self.max_pulse_size = wcc.bounded_intlike(max_pulse_size, name='max_pulse_size', minimum=1)

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

    def timecheck(self, fmt=float):
        if not isinstance(fmt, type):
            raise TypeError('fmt must be type "type"')
        try:
            return fmt(time())
        except TypeError:
            raise TypeError(f'fmt {fmt} must be compatable with a single float argument input')

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
        self.logger.debug('pulse initiated')
        y = x
        self.logger.debug('pulse concluded')
        return y
