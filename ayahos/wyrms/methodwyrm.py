"""
:module: wyrm.processing.inplace
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module hosts class definitions for a Wyrm class that facilitates pulsed
    execution of class methods of objects that conduct in-place alterations
    to those objects' contents.

    MethodWyrm - a submodule for applying a single class method to objects presented to the
                MethodWyrm
                    PULSE
                        input: deque of objects
                        output: deque of objects
    
"""
import logging
from collections import deque
import pandas as pd
from ayahos.wyrms.wyrm import Wyrm, add_class_name_to_docstring
from ayahos.core.dictstream import DictStream
from ayahos.core.windowstream import WindowStream

###################################################################################
# METHOD WYRM CLASS DEFINITION - FOR EXECUTING CLASS METHODS IN A PULSED MANNER ###
###################################################################################

Logger = logging.getLogger(__name__)

# @add_class_name_to_docstring
class MethodWyrm(Wyrm):
    """
    A submodule for applying a class method with specified key-word arguments to objects
    sourced from an input deque and passed to an output deque (self.output) following processing.

    Initialization Notes
    - sanity checks are applied to ensure that pmethod is in the attributes and methods associated with pclass
    - the only sanity check applied is that pkwargs is type dict. Users should refer to the documentation of their intended pclass.pmethod() to ensure keys and values are compatable.
    
    
    """
    def __init__(
        self,
        pclass,
        pmethod,
        pkwargs,
        max_pulse_size=10000,
        ):
        """
        Initialize a MethodWyrm object

        :: INPUTS ::
        :param pclass: expected class of unit_input objects passed to :meth:`~ayahos.wyrms.methodwyrm.MethodWyrm._unit_process`
        :type pclass: type, e.g., WindowStream
        :param pmethod: name of class method to apply to unit_input objects
        :type pmethod: str, e.g., "filter"
        :param pkwargs: key-word arguments (and positional arguments stated as key-word arguments) for pclass.pmethod(**pkwargs)
        :type pkwargs: dict, e.g., {"type": "bandpass", "freqmin": 1, "freqmax": 45}
        :param max_pulse_size: maximum number of iterations to conduct in a pulse, defaults to 10000.
        :type max_pulse_size: int, optional
        """

        # Initialize/inherit from Wyrm
        super().__init__(max_pulse_size=max_pulse_size)

        # pclass compatability checks
        if not isinstance(pclass,type):
            raise TypeError('pclass must be a class defining object (type "type")')
        else:
            self.pclass = pclass
        # pmethod compatability checks
        if pmethod not in [func for func in dir(self.pclass) if callable(getattr(self.pclass, func))]:
            raise ValueError(f'pmethod "{pmethod}" is not defined in {self.pclass} properties or methods')
        else:
            self.pmethod = pmethod
        # pkwargs compatability checks
        if isinstance(pkwargs, dict):
            self.pkwargs = pkwargs
        else:
            raise TypeError
        # initialize output queue
        self.queue = deque()

    # Inherited from Wyrm
    # def _continue_iteration()
    # def _capture_unit_out()
        
    def _unit_input_from_input(self, input):
        # Use checks from Wyrm on input
        unit_input = super()._unit_input_from_input(input)
        # Then apply checks from pclass
        if isinstance(unit_input, self.pclass):
            return unit_input
        else:
            Logger.critical(f'object popped from input mismatch {self.pclass} != {type(obj)}')
            raise TypeError
        
    def _unit_process(self, unit_input):
        """unit_process for MethodWyrm

        Check if the input deque and iteration number
        meet iteration continuation criteria inherited from Wyrm

        Check if the next object popleft'd off `x` is type self.pclass
        
            Mismatch: send object back to `x` with append()

            Match: Execute the in-place processing and append to MethodWyrm.output

        :param unit_input: object to be modified with self.pmethod(**self.pkwargs)
        :type unit_input: self.pclass
        :returns:
         - **unit_output** (*self.pclass*) -- modified object
        """ 
        getattr(unit_input, self.pmethod)(**self.pkwargs)
        unit_output = unit_input
        return unit_output
    


        # # Use inherited Wyrm()._continue_iteration() check to see if input deque has unassessed items
        # status = super()._continue_iteration(x, i_)
        # # if there are items in `x` to assess
        # if status:
        #     # Detach the next item
        #     _x = x.popleft()
        #     # Check that it matches pclass
        #     if isinstance(_x, self.pclass):
        #         # Run inplace modification
        #         getattr(_x, self.pmethod)(**self.pkwargs)
        #         # Attach modified object to output
        #         self.output.append(_x)
        #     # If the object isnt type pclass, re-attach it to x
        #     else:
        #         x.append(_x)
        # # Return status for use in Wyrm().pulse() iteration continuation assessment
        # return status

    
    def __str__(self):
        rstr = f'{self.__class__}(pclass={self.pclass}, '
        rstr += f'pmethod={self.pmethod}, pkwargs={self.pkwargs}, '
        rstr += f'max_pulse_size={self.max_pulse_size}, '
        return rstr




    # def pulse(self, x):
    #     """
    #     Execute a pulse wherein items are popleft'd off input deque `x`,
    #     checked if they are type `pclass`, have `pmethod(**pkwargs)` applied,
    #     and are appended to deque `self.queue`. Items popped off `x` that
    #     are not type pclass are reappended to `x`.

    #     Early stopping is triggered if `x` reaches 0 elements or the number of
    #     iterations equals the initial len(x)

    #     :: INPUT ::
    #     :param x: [deque] of [pclass (ideally)]

    #     :: OUTPUT ::
    #     :return y: [deque] access to the objects in self.queue
    #     """
    #     if not isinstance(x, deque):
    #         raise TypeError
    #     qlen = len(x)
    #     for _i in range(self.max_pulse_size):
    #         # Early stopping if all items have been assessed
    #         if _i - 1 > qlen:
    #             break
    #         # Early stopping if input deque is exhausted
    #         if len(x) == 0:
    #             break
            
    #         _x = x.popleft()
    #         if not isinstance(_x, self.pclass):
    #             x.append(_x)
    #         else:
    #             if self._timestamp:
    #                 _x.stats.processing.append(['MethodWyrm',self.pmethod, 'start', time.time()])
    #             getattr(_x, self.pmethod)(**self.pkwargs);
    #             # For objects with a stats.processing attribute, append processing info
    #             # if 'stats' in dir(_x):
    #             #     if 'processing' in dir(_x.stats):
    #             #         _x.stats.processing.append(
    #             #             [time.time(),
    #             #              'Wyrm 0.0.0',
    #             #              'MethodWyrm',
    #             #              self.pmethod,
    #             #              f'({self.pkwargs})'])
    #             if self._timestamp:
    #                 _x.stats.processing.append(['MethodWyrm',self.pmethod, 'end', time.time()])
    #             self.queue.append(_x)
    #     y = self.queue
    #     return y