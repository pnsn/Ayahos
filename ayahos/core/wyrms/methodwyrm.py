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
from ayahos.core.wyrms.wyrm import Wyrm, add_class_name_to_docstring
from ayahos.core.stream.dictstream import DictStream
from ayahos.core.stream.windowstream import WindowStream

###################################################################################
# METHOD WYRM CLASS DEFINITION - FOR EXECUTING CLASS METHODS IN A PULSED MANNER ###
###################################################################################

Logger = logging.getLogger(__name__)

# @add_class_name_to_docstring
class MethodWyrm(Wyrm):
    """
    A submodule for applying a class method with specified key-word arguments to objects
    sourced from an input deque and passed to an output deque (self.queue) following processing.

    NOTE: This submodule assumes that applied class methods apply in-place alterations on data

    """
    def __init__(
        self,
        pclass=WindowStream,
        pmethod="filter",
        pkwargs={'type': 'bandpass',
                 'freqmin': 1,
                 'freqmax': 45},
        max_pulse_size=10000,
        ):
        """
        Initialize a MethodWyrm object

        :: INPUTS ::
        :param pclass: [type] class defining object (ComponentStream, MLStream, etc.)
        :param pmethod: [str] name of class method to apply 
                            NOTE: sanity checks are applied to ensure that pmethod is in the
                                attributes and methods associated with pclass
        :param pkwargs: [dict] dictionary of key-word arguments (and positional arguments
                                stated as key-word arguments) for pclass.pmethod(**pkwargs)
                            NOTE: only sanity check applied is that pkwargs is type dict.
                                Users should refer to the documentation of their intended
                                pclass.pmethod() to ensure keys and values are compatable.
        :param max_pulse_size: [int] positive valued maximum number of objects to process
                                during a call of MethodWyrm.pulse()
        :param debug: [bool] should this Wyrm be run in debug mode?

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
        
    def _get_obj_from_input(self, stdin):
        # Use checks from Wyrm on stdin
        obj = super()._get_obj_from_input(stdin)
        # Then apply checks from pclass
        if isinstance(obj, self.pclass):
            return obj
        else:
            Logger.critical(f'object popped from stdin mismatch {self.pclass} != {type(obj)}')
            raise TypeError
        
    def _unit_process(self, obj):
        """unit_process for MethodWyrm

        Check if the input deque and iteration number
        meet iteration continuation criteria inherited from Wyrm

        Check if the next object popleft'd off `x` is type self.pclass
        
            Mismatch: send object back to `x` with append()

            Match: Execute the in-place processing and append to MethodWyrm.output

        :param obj: input object to be modified
        :type obj: self.pclass

        :return unit_out: modified obj
        :rtype: self.pclass
        """ 
        getattr(obj, self.pmethod)(**self.pkwargs)
        unit_out = obj
        return unit_out
    


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