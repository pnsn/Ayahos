"""
:submodule: wyrm.unit.wyrm
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
import logging

# Logger = logging.getLogger(__name__)

def add_class_name_to_docstring(cls):
    for name, method in vars(cls).items():
        if callable(method) and hasattr(method, "__doc__"):
            method.__doc__ = method.__doc__.format(class_name_lower=cls.__name__.lower(),
                                                   class_name_camel=cls.__name__,
                                                   class_name_upper=cls.__name__.upper())
    return cls

Logger = logging.getLogger(__name__)

# @add_class_name_to_docstring
class Wyrm(object):
    """Fundamental base class template all other *Wyrm classes with the following
    template methods

    Wyrm().pulse() - performs multiple iterations of the following polymorphic subroutines
        Wyrm()._should_this_iteration_run() - check if this iteration in pulse should be run
        Wyrm()._unit_input_from_input() - get the object input for _unit_process()
        Wyrm()._unit_process() - execute a single operation on the input object
        Wyrm()._capture_unit_output() - do something to save the output of _unit_process
        Wyrm()._should_next_iteration_run() - decide if the next iteration should be run

    Wyrm.output - collector of outputs from core_process(). Should be modified along with
                Wyrm().core_process() to meet your needs.

    :param max_pulse_size: maximum number of iterations to run for wyrm.pulse(), defaults to 10
    :type max_pulse_size: int, optional
    
    """

    def __init__(self, max_pulse_size=10):
        """Initialize a {class_name_camel} object

        :param max_pulse_size: maximum , defaults to 10
        :type max_pulse_size: int, optional
        :raises ValueError: for non-positive number argument for max_pulse_size
        :raises TypeError: for non-int-like or non-NoneType argument for max_pulse_size
        """
        # Logger.debug('Initializing a Wyrm object')
        # Compatability check for max_pulse_size
        if isinstance(max_pulse_size, (int, float)):
            if 1 <= max_pulse_size:
                self.max_pulse_size = int(max_pulse_size)
            else:
                raise ValueError('max_pulse_size must be g.e. 1 ')
        else:
            raise TypeError('max_pulse_size must be positive int-like')
        self.output = deque()

    def __name__(self):
        """Return the camel-case name of this class without
        the submodule extension

        alias of self.__class__.__name__
        :return: class name
        :rtype: str
        """        
        return self.__class__.__name__
    
    def __repr__(self):
        """
        Provide a string representation string of essential user data for this {class_name_camel}
        """
        # rstr = "~~wyrm~~\nFundamental Base Class\n...I got no legs...\n"
        rstr = f'{self.__class__}\n'
        rstr += f"Max Pulse Size: {self.max_pulse_size}\nOutput: {len(self.output)} (type {type(self.output)})"
        return rstr

    def __str__(self):
        """
        Return self.__class__ for this wyrm
        """
        rstr = self.__class__ 
        #(max_pulse_size={self.max_pulse_size})'
        return rstr
    
    def copy(self):
        """
        Return a deepcopy of this Wyrm-like object

        :return: copy of this {class_name_camel} object
        :rtype: ayahos.core.wyrms.{class_name_lower}.{class_name_camel}
        """
        return deepcopy(self)

    def pulse(self, input, mute_logging=False):
        """
        TEMPLATE METHOD
         - Last altered with :class: `~ayahos.wyrms.wyrm.Wyrm`

        Run up to max_pulse_size iterations of _unit_process()
        for ayahos.core.wyrms.{class_name_lower}.{class_name_camel}

        NOTE: Houses the following polymorphic methods that can be modified for decendent Wyrm-like classes
            _should_this_iteration_run: check if iteration should continue (early stopping opportunity)
            _unit_input_from_input: get input object for _unit_process
            _unit_process: run core process
            _capture_unit_output: attach _unit_process output to {class_name_camel}.output
            _should_next_iteration_run: check if the next iteration should occur (early stopping opportunity)

        :param input: standard input
        :type input: collections.deque
            see ayahos.core.wyrms.wyrm.Wyrm._unit_input_from_input()
        :return output: aliased access to {class_name_camel}.output
        :rtype output: collections.deque of objects
        """ 
        # Iterate across
        if not mute_logging:
            Logger.debug(f'{self.__name__} pulse firing')
        input_size = self._measure_input_size(input)
        nproc = 0
        for iter_number in range(self.max_pulse_size):
            # Check if this iteration should proceed
            if self._should_this_iteration_run(input, input_size, iter_number):
                pass
            else:
                break
            # get single object for unit_process
            unit_input = self._unit_input_from_input(input)
            # Execute unit process
            unit_output = self._unit_process(unit_input)
            # Increase process counter
            nproc += 1
            # Capture output
            self._capture_unit_output(unit_output)
            #  Check if early stopping should occur at the end of this iteration
            if self._should_next_iteration_run(unit_output):
                pass
            else:
                break
        if not mute_logging:
            Logger.debug(f'{self.__name__} {nproc} processes run (MAX: {self.max_pulse_size})')
        # Get alias of self.output as output
        output = self.output
        return output, nproc

    def _measure_input_size(self, input):
        """
        POLYMORPHIC
        Last updated with :class: `~ayahos.wyrms.wyrm.Wyrm`

        take a reference measurement for the input before starting
        iterations for :meth: `~ayahos.core.wyrm.Wyrm.pulse(input)` prior
        to starting the pulse.

        This version measures the length of input.

        :param input: standard input
        :type input: varies, deque here
        :return input_size: representative measure of input
        :rtype: int-like
        """        
        if input is None:
            input_size = self.max_pulse_size
        else:
            input_size = len(input)
        return input_size

    def _should_this_iteration_run(self, input, input_size, iter_number):
        """
        POLYMORPHIC - last updated with :class: `~ayahos.wyrms.wyrm.Wyrm`

        Should this iteration in :meth: `~ayahos.wyrms.wyrm.Wyrm.pulse()` be run?
        
        Criteria:
         - input is type collections.deque
         - len(input) > 0
         - iter_number + 1 <= len(input)
        
        :param input: input data object collection
        :type input: collections.deque
        :param iter_number: iteration number
        :type iter_number: int
        :return status: continue to next iteration?
        :rtype status: bool
        """
        status = False
        # if input is deque
        if isinstance(input, deque):
            # and input is non-empty
            if len(input) > 0:
                # and iter_number +1 is l.e. length of input
                if iter_number + 1 <= input_size:
                    status = True
        return status
    
    def _unit_input_from_input(self, input):
        """
        POLYMORPHIC
        Last updated with :class: `~ayahos.wyrms.wyrm.Wyrm`

        Get the input object for this Wyrm's _unit_process

        :param input: standard input object
        :type input: collections.deque
        :return unit_input: unit_input popleft'd from input
        :rtype unit_input: any
        
        :raises TypeError: if input is not expected type
        
        """        
        if isinstance(input, deque):
            unit_input = input.popleft()
            return unit_input
        else:
            Logger.error(f'input object was incorrect type')
            raise TypeError
        

    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last updated with :class: `~ayahos.wyrms.wyrm.Wyrm`

        return unit_output = unit_input

        :param obj: input object
        :type obj: any
        :return unit_output: output object
        :rtype unit_output: any
        """
        unit_output = unit_input
        return unit_output        
    
    def _capture_unit_output(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class: `~ayahos.wyrms.wyrm.Wyrm`

        Append unit_output to self.output

        run Wyrm().output.append(unit_output)

        :param unit_output: standard output object from unit_process
        :type unit_output: any
        :return: None
        :rtype: None
        """        
        self.output.append(unit_output)
        return None
        
    def _should_next_iteration_run(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class: `~ayahos.wyrms.wyrm.Wyrm`

        check if the next iteration should be run based on unit_output

        Returns status = True unconditionally

        :param unit_out: output _unit_process
        :type unit_out: object
        :return status: should the next iteration be run?
        :rtype: bool
        """
        status = True
        return status
