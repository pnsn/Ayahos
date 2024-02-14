"""
:module: wyrm.core.wyrm
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module hosts the fundamental class definition for all
    other *Wyrm classes -- "Wyrm" -- and serves as a template
    for the minimum required methods of each successor class. 
"""
import wyrm.util.input_compatability_checks as icc


class Wyrm(object):
    """
    Fundamental Base Class for all *Wyrm classes in this module that are
    defined by having the y = *wyrm.pulse(x) class method.

    The Wyrm base class produces an object with no attributes and placeholders
    for the 4 fundamental class-methods common to all Wyrm class objects:

    + __init__
    + __str__
    + __repr__
    + pulse
    """

    def __init__(self, max_pulse_size=None, debug=False):
        """
        Initialize a Wyrm object
        """
        # Compatability check for debug
        if not isinstance(debug, bool):
            raise TypeError("debug must be type bool")
        else:
            self.debug = debug
        
        # Compatability check for max_pulse_size
        if max_pulse_size is None:
            self.max_pulse_size = None
        else:
            self.max_pulse_size = icc.bounded_intlike(max_pulse_size, name='max_pulse_size', minimum=1)

    def __str__(self):
        """
        Provide a string representation string of essential user data for this Wyrm
        """
        # rstr = "~~wyrm~~\nFundamental Base Class\n...I got no legs...\n"
        rstr = f"Max Pulse Size: {self.max_pulse_size} | debug: {self.debug}"
        return rstr

    def __repr__(self):
        """
        Provide a string representation of how to recreate this Wyrm
        """
        rstr = f'wyrm.wyrms.base.Wyrm(max_pulse_size={self.max_pulse_size}, debug={self.debug})'
        return rstr
    
    def pulse(self, x=None):
        """
        ~~~ POLYMORPHIC METHOD ~~~
        Run a pulse with input argument and return that argument
        """
        y = x
        return y
