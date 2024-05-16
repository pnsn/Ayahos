"""
:module: wyrm.core.coordinate
:auth: Nathan T Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains class defining submodules for coordinating sets
    of wyrms to form parts of fully operational Wyrm modules, including:

    CloneWyrm - a submodule class for producing cloned deques of objects in
                a pulsed manner.
"""
import copy, logging
from collections import deque
from ayahos.core.wyrms.wyrm import Wyrm

Logger = logging.getLogger(__name__)

class ForkWyrm(Wyrm):
    """
    Submodule class that provides a pulsed method for producing (multiple) copies
    of an arbitrary set of items in an input deque into a dictionary of output deques
    """
    def __init__(self, queue_names=['A','B'], max_pulse_size=1000000, debug=False):
        """
        Initialize a ForkWyrm object
        :: INPUTS ::
        :param queue_names: [list-like] of values to assign as keys (names) to 
                            output deques held in self.queues
        :p
        :param max_pulse_size: [int] maximum number of elements to pull from 
                            an input deque in a single pulse
        :param debug: [bool] - should this be run in debug mode?
        """
        super().__init__(max_pulse_size=max_pulse_size)
        self.nqueues = len(queue_names)
        self.output = {}
        for _k in queue_names:
            self.output.update({_k: deque()})
    
    # Inherit from Wyrm()
    # def _continue_iteration - input is deque, non-empty,and iterno + 1 <= len(stdin)
    # def _get_obj_from_input - use popleft()
    
    def _unit_process(self, obj):
        """_unit_process for ForkWyrm

        appends a copy of obj to each deque in output

        :param obj: object to copy
        :type obj: any
        :return unit_out: placeholder unit_out
        :rtype: None
        """        
        for _v in self.output.values():
            _v.append(copy.deepcopy(obj))
        unit_out = None
        return unit_out
    
    def _capture_unit_out(self, unit_out):
        """_capture_unit_out for ForkWyrm

        pass - placeholder

        Capture is handled in _unit_process

        :param unit_out: placeholder, unused
        :type unit_out: any
        :return status: unconditional True (keep iterating in pulse)
        :rtype status: bool
        """        
        status = True
        return status
