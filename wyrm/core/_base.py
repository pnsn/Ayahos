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

class Wyrm(object):
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
            self.max_pulse_size = wcc.bounded_intlike(max_pulse_size, name='max_pulse_size', minimum=1)
        # # input and output type for pulse
        # self._in_type = (int, str, float, bool, type(None), type)
        # self._out_type = (int, str, float, bool, type(None), type)

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
    
    # def _update_io_types(self, itype=None, otype=None):
    #     """
    #     --- PRIVATE METHOD ---
    #     Update the reference type(s) for the input and output of self.pulse()

    #     :: INPUTS ::
    #     :param itype: [type], [NoneType], or [tuple] thereof
    #                     None input results in no change to _in_type
    #     :param otype: [type], [NoneType], or [tuple] thereof
    #                     None input results in no change to _out_type
    #     :: OUTPUT ::
    #     :return self: [Wyrm] return self to enable cascading
    #     """
    #     if itype is not None and isinstance(itype, type):
    #         self._in_type = itype
    #     elif isinstance(itype, tuple):
    #         if all(isinstance(_it, (type, type(None))) for _it in itype):
    #             self._in_type = itype
    #     if otype is not None and isinstance(otype, type):
    #         self._out_type = otype
    #     elif isinstance(otype, tuple):
    #         if all(isinstance(_it, (type, type(None))) for _it in otype):
    #             self._out_type = otype
    #     return self

    # def _matches_itype(self, arg, raise_error=False):
    #     if not isinstance(arg, self._in_type):
    #         if raise_error:
    #             emsg = f'arg type {type(arg)} does not match expected type(s): {self._in_type}'
    #             raise TypeError(emsg)
    #         else:
    #             return False
    #     else:
    #         if raise_error:
    #             pass
    #         else:
    #             return True
    
    # def _matches_otype(self, arg, raise_error=False):
    #     if not isinstance(arg, self._out_type):
    #         if raise_error:
    #             emsg = f'arg type {type(arg)} does not match expected type(s): {self._out_type}'
    #             raise TypeError(emsg)
    #         else:
    #             return False
    #     else:
    #         if raise_error:
    #             pass
    #         else:
    #             return True

    def copy(self):
        """
        Return a deepcopy of this wyrm
        """
        return deepcopy(self)

    def pulse(self, x=None, **options):
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
        self._matches_itype(x)
        y = x
        self._matches_otype(y)
        return y
