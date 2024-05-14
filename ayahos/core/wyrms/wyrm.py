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

Logger = logging.getLogger(__name__)

def add_class_name_to_docstring(cls):
    for name, method in vars(cls).items():
        if callable(method) and hasattr(method, "__doc__"):
            method.__doc__ = method.__doc__.format(class_name_lower=cls.__name__.lower(),
                                                   class_name_camel=cls.__name__,
                                                   class_name_upper=cls.__name__.upper())
    return cls

@add_class_name_to_docstring
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
        Logger.debug('Initializing a Wyrm object')
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

    def pulse(self, x):
        """
        Run up to max_pulse_size iterations of unit_process()
        for ayahos.core.wyrms.{class_name_lower}.{class_name_camel}

        see {class_name_camel}.unit_process() for details

        :param x: input iterable
        :type x: any
        :return: aliased access to {class_name_camel}.output
        :rtype: collection.deque of objects
        """ 
        # Iterate across 
        Logger.debug(f'{self.__class__} pulse firing')
        for i_ in range(self.max_pulse_size):
            status = self.unit_process(x, i_)
            if status is False:
                break
        y = self.output
        return y

    def unit_process(self, x, i_):
        """The unit process for this particular wyrm
        ayahos.core.wyrms.{class_name_lower}.{class_name_camel}
        0) assess if _early_stopping criteria are met
        if not:
            1) popleft() an item in x
            2) set _y = _x
            3) append(_y) to self.output

        :param x: input data object source
        :type x: collections.deque of objects
        :param i_: iteration number
        :type i_: int
        :return status: early stopping flag. True = stop early, False = continue
        :rtype status: bool
        """
        if isinstance(x, deque):
            if self._continue_iteration(x, i_):
                _x = x.popleft()
                _y = _x
                self.output.append(_y)
                status = True
            else:
                status = False
        else:
            raise TypeError('x must be type collections.deque')

    def _continue_iteration(self, x, i_):
        """Iteration continuation criteria for {class_name_camel}

        Criteria:
         - len(x) > 0
         - i_ < len(x)
        
        :param x: input data object collection
        :type x: collections.deque
        :param i_: iteration number
        :type i_: int
        :return: continue to next iteration?
        :rtype: bool
        """        
        if len(x) == 0:
            return False
        elif i_ + 1 > len(x):
            return False
        else:
            return True
        
