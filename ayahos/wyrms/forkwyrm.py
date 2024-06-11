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
from ayahos.wyrms.wyrm import Wyrm

Logger = logging.getLogger(__name__)

class ForkWyrm(Wyrm):
    """
    Wyrm-type class submodule that creates deepcopy's of inputs to its
    :meth:`~ayahos.wyrm.forkwyrm.ForkWyrm.pulse` method and presents
    them as a dictionary tied to its self.output attribute. 

    :param output_names: names to assign to each deepcopy in self.output
        Notes:
        - This parameter also sets the number of copies that will be produced.
        - Providing multiple identical names will result in only one output of that name
            also see :meth:`~dict.update`
    :type output_names: list
    :param max_copy_per_pulse: maximum number of items to copy from 
    """
    def __init__(
            self,
            output_names=['A','B'],
            max_pulse_size=1000000,
            meta_memory=3600,
            report_period=None):
        """
        Initialize a ForkWyrm object

        :param output_names: [list-like] of values to assign as keys (names) to 
                            output deques held in self.queues
        :p
        :param max_pulse_size: maximum
        :param debug: [bool] - should this be run in debug mode?
        """
        super().__init__(max_pulse_size=max_pulse_size,
                         meta_memory=meta_memory,
                         report_period=report_period)
        self.output = {}
        for _k in output_names:
            self.output.update({_k: deque()})
    
    # Inherit from Wyrm()
    # def _continue_iteration - input is deque, non-empty,and iterno + 1 <= len(stdin)
    # def _get_obj_from_input - use popleft()
    def _unit_input_from_input(self, input):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.forkwyrm.ForkWyrm`

        :param input: generalized input value
        :type input: deque
        :returns:
            - **unit_input** (*varies*) - an object removed from **input** using :meth:`~collections.deque.popleft`
        """        
        unit_input = input.popleft()
        return unit_input
    
    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.forkwyrm.ForkWyrm`

        Appends a deepcopy of unit input from 


        :param unit_input: unit_inputect to copy
        :type unit_input: any
        :return unit_output: placeholder unit_output
        :rtype: None
        """        
        for _k in self.output.keys():
            self.output[_k].append(copy.deepcopy(unit_input))
        unit_output = None
        return unit_output
    
    def _capture_unit_out(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.forkwyrm.ForkWyrm`

        Unconditionally returns True to continue iterations.

        Capture is handled in _unit_process

        :param unit_output: placeholder, unused
        :type unit_output: None
        :return status: unconditional True (keep iterating in pulse)
        :rtype status: bool
        """ 
        if unit_output is not None:
            Logger.warning('_unit_process is producing non-NoneType unit_output values')     
        
        status = True
        return status

    def _should_next_iteration_run(self, unit_output):
        if unit_output is not None:
            Logger.warning('_unit_process is producing non-NoneType outputs')
        status = True
        return status