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

    def __init__(self):
        """
        Initialize a Wyrm object
        """
        return None

    def __repr__(self):
        """
        Provide a representation string of a Wyrm
        """
        rstr = "~~wyrm~~\nFundamental Base Class\n...I got no legs...\n"
        return rstr

    def pulse(self, x=None):
        """
        ~~~ POLYMORPHIC METHOD ~~~
        Run a pulse with input argument and return that argument
        """
        y = x
        return y
