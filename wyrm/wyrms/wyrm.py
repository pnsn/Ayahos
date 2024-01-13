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

class Wyrm:
    """
    Fundamental Base Class for all *Wyrm classes in this module that are
    defined by having the y = *wyrm.pulse(x) class method.

    The Wyrm base class produces an object with no attributes and placeholders
    for the 3 fundamental class-methods common to all Wyrm class objects:

    + __init__
    + __repr__
    + pulse
    """

    def __init__(self, max_pulse_size=None, debug=False):
        """
        Initialize a Wyrm object
        """
        if not isinstance(debug, bool):
            raise TypeError("debug must be type bool")
        else:
            self.debug = debug

        # Inherit input compatability check methods
        self.bounded_intlike = icc.bounded_intlike
        self.bounded_floatlike = icc.bounded_floatlike
        self.iterable_characters = icc.iterable_characters
        self.none_str = icc.none_str
        self.iscamelcase_str = icc.iscamelcase_str
        self.isiterable = icc.isiterable
        self.validate_seisbench_model_name = icc.validate_seisbench_model_name

        if max_pulse_size is None:
            self.max_pulse_size = None
        else:
            self.max_pulse_size = self.bounded_intlike(max_pulse_size, name='max_pulse_size', minimum=1)

    def __str__(self):
        """
        Provide a representation string of a Wyrm
        """
        # rstr = "~~wyrm~~\nFundamental Base Class\n...I got no legs...\n"
        rstr = f"Max Pulse Size: {self.max_pulse_size} | debug: {self.debug}"
        return rstr

    def __repr__(self):
        rstr = self.__str__()
        return rstr
    
    def pulse(self, x=None):
        """
        ~~~ POLYMORPHIC METHOD ~~~
        Run a pulse with input argument and return that argument
        """
        y = x
        return y
