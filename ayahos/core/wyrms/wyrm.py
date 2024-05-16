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

# @add_class_name_to_docstring
class Wyrm(object):
    """Fundamental base class template all other *Wyrm classes with the following
    template methods

    Wyrm().pulse() - performs a for-loop execution of many Wyrm().core_process() calls
    Wyrm().core_process() - template method for polymorphic inheritance

    Wyrm.output - collector of outputs from core_process(). Should be modified along with
                Wyrm().core_process() to meet your needs.
    
    """

    def __init__(self, max_pulse_size=10):
        """Initialize a {class_name_camel} object

        :param max_pulse_size: maximum , defaults to 10
        :type max_pulse_size: int, optional
        :raises ValueError: for non-positive number argument for max_pulse_size
        :raises TypeError: for non-int-like or non-NoneType argument for max_pulse_size
        """
        self.logger = logging.getLogger(__name__)
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
        Return a deepcopy of this wyrm

        :return: copy of this {class_name_camel} object
        :rtype: ayahos.core.wyrms.{class_name_lower}.{class_name_camel}
        """
        return deepcopy(self)

    def pulse(self, stdin):
        """
        Run up to max_pulse_size iterations of _unit_process()
        for ayahos.core.wyrms.{class_name_lower}.{class_name_camel}

        NOTE: Houses the following polymorphic methods
            _continue_iteration: check if iteration should continue
            _get_obj_from_input: get input object for _unit_process
            _unit_process: run core process
            _capture_stdout: attach _unit_process output to {class_name_camel}.output

        :param stdin: standard input
        :type stdin: collections.deque
            see ayahos.core.wyrms.wyrm.Wyrm._get_obj_from_input()
        :return stdout: aliased access to {class_name_camel}.output
        :rtype stdout: collections.deque of objects
        """ 
        # Iterate across 
        self.logger.debug(f'{self.__class__.__name__} pulse firing')
        stdin_measure = self._measure_stdin(stdin)
        for iterno in range(self.max_pulse_size):
            # Check if iterations should continue
            status1 = self._continue_iteration(stdin, stdin_measure, iterno)
            # If iterations should continue
            if status1:
                # get single object for unit_process
                obj = self._get_obj_from_input(stdin)
                # Execute unit process
                unit_out = self._unit_process(obj)
                # Attach
                try:
                    status2 = self._capture_unit_out(unit_out)
                except:
                    breakpoint()
                if status2 is False:
                    break
            # If iterations should not continue
            else:
                # end iterations
                break
        # Get alias of self.output as stdout
        stdout = self.output
        return stdout

    def _measure_stdin(self, stdin):
        """reference measurement for stdin

        :param stdin: standard input
        :type stdin: varies, deque here
        :return stdin_measure: representative measure of stdin
        :rtype: int-like
        """        
        if stdin is None:
            stdin_measure = self.max_pulse_size
        else:
            stdin_measure = len(stdin)
        return stdin_measure

    def _continue_iteration(self, stdin, stdin_measure, iterno):
        """Iteration continuation criteria for {class_name_camel}
        POLYMORPHIC

        Criteria:
         - stdin is type collections.deque
         - len(stdin) > 0
         - iterno + 1 <= len(stdin)
        
        :param stdin: input data object collection
        :type stdin: collections.deque
        :param iterno: iteration number
        :type iterno: int
        :return status: continue to next iteration?
        :rtype status: bool
        """
        status = False
        # if stdin is deque
        if isinstance(stdin, deque):
            # and stdin is non-empty
            if len(stdin) > 0:
                # and iterno +1 is l.e. length of stdin
                if iterno + 1 <= stdin_measure:
                    status = True
        return status
    
    def _get_obj_from_input(self, stdin):
        """_get_obj_from_input for Wyrm
        POLYMORPHIC

        Get the input object for this Wyrm's _unit_process

        :param stdin: standard input object
        :type stdin: collections.deque
        :return obj: object popleft'd from stdin
        :rtype obj: any
        
        :raises TypeError: if stdin is not expected type
        
        """        
        if isinstance(stdin, deque):
            obj = stdin.popleft()
            return obj
        else:
            self.logger.error(f'stdin object was incorrect type')
            raise TypeError
        

    def _unit_process(self, obj):
        """unit_process for ayahos.core.wyrms.wyrm.Wyrm
        POLYMORPHIC

        return unit_out = obj

        :param obj: input object
        :type obj: any
        :return unit_out: output object
        :rtype unit_out: any
        """
        unit_out = obj
        return unit_out        
    
    def _capture_unit_out(self, unit_out):
        """Append the unit_out to Wyrm().output
        POLYMORPHIC

        run Wyrm().output.append(unit_out)

        :param unit_out: standard output object from unit_process
        :type unit_out: any
        """        
        self.output.append(unit_out)
        status = True
        return status
