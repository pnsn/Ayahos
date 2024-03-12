"""
:module: wyrm.core.coordinate
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module houses classes used to coordinate operation and execution
    of parallel and serial sets of wyrms and operate the core PyEW.EWModule
    object that defines this collection of wyrms as an Earthworm module.

    Classes:
        TubeWyrm - submodule for executing sequential sets of wyrms wherein
                the output of one wyrm's pulse() method is taken as the input
                to the next wyrm's pulse() method. Final outputs are housed
                in a queue (collections.deque)

                deque(x) -> TubeWyrm -> deque(y)
                    with y = wyrm3.pulse(wyrm2.pulse(wyrm1.pulse(wyrm0.pulse(x))))
        
        CanWyrm - TubeWyrm child class - submodule for executing parallel sets of wyrms
                where the input to all wyrms is the same and the output is written to a
                dictionary keyed by a name given to each wyrm
                              / Wyrm0.pulse(x) -> {wyrm0: y0}
                x -> CanWyrm {  Wyrm1.pulse(x) -> {wyrm1: y1}
                              \ WyrmN.pulse(x) -> {wyrmN: yN}
        
        HeartWyrm - TubeWyrm child class - submodule housing the PyEW.EWModule instance
                that facilitates communication between the Earthworm Message Transport
                System and Python

        ProcWyrm - submodule for coordinating chained class method execution on deques
                of initialized objects. Uses eval() functions to form eval statements
                of the form f'obj{_method_str}' where _method_str has the minimum
                form '.method(args)'.

TODO:
    - Update HeartWyrm with wcc
    - Finish cleanup of CanWyrm                                              
"""
from wyrm.core._base import Wyrm
from wyrm.core.data import TraceBuffer, BufferTree, Trace
import wyrm.util.compatability as wcc
from collections import deque
from threading import Thread
from time import sleep
import pandas as pd
import numpy as np
# import PyEW

class TreeWyrm(Wyrm):
    """
    Wyrm submodule that takes the items presented from the left end of a
    double ended queue (collections.deque) and distributes them to a BufferTree 
    via the BufferTree.append() method.
    """
    def __init__(self,
                 input_class=Trace,
                 buff_class=TraceBuffer,
                 tk0_eval="'.'.join(_x.id.split('.')[:-1])",
                 tk1_eval="_x.stats.channel[:-1]",
                 tk2_eval="_x.stats.channel[-1]",
                 max_pulse_size=10000,
                 debug=False,
                 **buff_class_options
                 ):
        """
        Initialize a TreeWyrm Object
        """
        # Inherit from Wyrm
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)
        # Initialize Buffer
        self.buffer = BufferTree(buff_class=buff_class, **buff_class_options)
        # input_class compat checks
        if isinstance(input_class, (list, tuple)):
            if all(isinstance(_e, type) for _e in input_class):
                self.input_class = input_class
            else:
                raise TypeError('not all elements in list-like input_class are type "type"')
        elif isinstance(input_class, type):
            self.input_class = input_class
        else:
            raise TypeError('input_class must be type "type" or a list-like thereof')
        
        if not isinstance(tk0_eval, str):
            raise TypeError
        elif '_x' not in tk0_eval:
            raise SyntaxError
        else:
            self.tk0_eval = tk0_eval
    
        if not isinstance(tk1_eval, str):
            raise TypeError
        elif '_x' not in tk1_eval:
            raise SyntaxError
        else:
            self.tk1_eval = tk1_eval

        if not isinstance(tk2_eval, str):
            raise TypeError
        elif '_x' not in tk2_eval:
            raise SyntaxError
        else:
            self.tk2_eval = tk2_eval


    def pulse(self, x, **options):
        """
        Conduct a pulse that can accept either a deque or an self.input_class object
        and appends the contents (or input_class object) to self.buffer.
        """
        if not isinstance(x, (deque, self.input_class)):
            raise TypeError(f'input x must be type deque or {self.input_class}')
        elif isinstance(x, self.input_class):
            _x = x
            tk0 = eval(self.tk0_eval)
            tk1 = eval(self.tk1_eval)
            tk2 = eval(self.tk2_eval)
            self.buffer.append(_x, TK0=tk0, TK1=tk1, TK2=tk2, **options)
        else:
            qlen = len(x)
            if qlen < self.max_pulse_size:
                iterations = qlen
            else:
                iterations = self.max_pulse_size
            
            for _ in range(iterations):
                _x = x.popleft()
                if not isinstance(_x, self.input_class):
                    x.append(x)
                else:
                    tk0 = eval(self.tk0_eval)
                    tk1 = eval(self.tk1_eval)
                    tk2 = eval(self.tk2_eval)
                    self.buffer.append(_x, TK0=tk0, TK1=tk1, TK2=tk2, **options)

        y = self.buffer
        return y


class TubeWyrm(Wyrm):
    """
    Wyrm child-class facilitating chained execution of pulse(x) class methods
    for a sequence wyrm objects, with each wyrm.pulse(x) taking the prior
    member's pulse(x) output as its input.

    The pulse method operates as follows
    for wyrm_dict = {key0: <wyrm0>, key2: <wyrm2> , key1: <wyrm1>}
        tubewyrm.pulse(x) = wyrm1.pulse(wyrm2.pulse(wyrm0.pulse(x)))

        Note that the dictionary ordering dictates the execution order!
    """

    def __init__(self, wyrm_dict={}, wait_sec=0.0, max_pulse_size=1, debug=False):
        """
        Create a tubewyrm object
        :: INPUT ::
        :param wyrm_dict: [dict], [list], or [deque] of [Wyrm-type] objects
                        that are executed in their provided order. dict-type
                        entries allow for naming of component wyrms for 
                        user-friendly assessment. list, deque, and Wyrm-type
                        inputs use a 0-indexed naming system.
        :param wait_sec: [float] seconds to wait between execution of each
                        Wyrm in wyrm_dict
        :param max_pulse_size: [int] number of times to run the sequence of pulses
                        Generally, this should be 1.
        :param debug: [bool] run in debug mode?

        :: OUTPUT ::
        Initialized TubeWyrm object
        """
        # Inherit from Wyrm
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)

        # wyrm_dict compat. checks
        if isinstance(wyrm_dict, Wyrm):
            self.wyrm_dict = {'_tmp': wyrm_dict}
        elif isinstance(wyrm_dict, dict):
            if all(isinstance(_w, Wyrm) for _w in wyrm_dict.values()):
                self.wyrm_dict = wyrm_dict
            else:
                raise TypeError('All elements in wyrm_dict must be type Wyrm')
        # Handle case where a list or deque are provided
        elif isinstance(wyrm_dict, (list, deque)):
            if all(isinstance(_w, Wyrm) for _w in wyrm_dict):
                # Default keys are the sequence index of a given wyrm
                self.wyrm_dict = {_k: _v for _k, _v in enumerate(wyrm_dict)}
            else:
                raise TypeError('All elements in wyrm_dict must be type Wyrm')
        else:
            raise TypeError('wyrm_dict must be a single Wyrm-type object or a list or dictionary thereof')
        
        # wait_sec compat. checks
        self.wait_sec = wcc.bounded_floatlike(
            wait_sec,
            name='wait_sec',
            minimum=0,
            maximum=None,
            inclusive=True
        )
        # Create a list representation of keys
        self.names = list(wyrm_dict.keys())

        # Enforce debug setting on all subsequent wyrms
        for _w in self.wyrm_dict.values():
            _w.debug = self.debug
        
        # # Get input and output types from wyrm_dict
        # self._in_type = self.wyrm_dict[self.names[0]]._in_type
        # self._out_type = self.wyrm_dict[self.names[0]]._out_type

    def update(self, new_dict):
        """
        Apply update to wyrm_dict using the 
        dict.update(new_dict) builtin_function_or_method
        and then update relevant attributes of this TubeWyrm

        This will update existing keyed entries and append new
        at the end of self.wyrm_dict (same behavior as dict.update)

        :: INPUT ::
        :param new_dict: [dict] of [Wyrm] objects

        :: OUTPUT ::
        :return self: [TubeWyrm] enables cascading
        """
        if not isinstance(new_dict, dict):
            raise TypeError('new_dict must be type dict')
        elif not all(isinstance(_w, Wyrm) for _w in new_dict.values()):
            raise TypeError('new_dict can only have values of type Wyrm')
        else:
            pass
        # Run updates on wyrm_dict, enforce debug, and update names list
        self.wyrm_dict.update(new_dict)
        for _w in self.wyrm_dict.values():
            _w.debug = self.debug
        self.names = list(self.wyrm_dict.keys())
        return self
    
    def remove(self, key):
        """
        Convenience wrapper of the dict.pop() method
        to remove an element from self.wyrm_dict and
        associated attributes

        :: INPUT ::
        :param key: [object] valid key in self.wyrm_dict.keys()

        :: RETURN ::
        :return popped_item: [tuple] popped (key, value)
        """
        if key not in self.wyrm_dict.keys():
            raise KeyError(f'key {key} is not in self.wyrm_dict.keys()')
        
        val = self.wyrm_dict.pop(key)
        self.names = list(self.wyrm_dict.keys())
        return (key, val)

    def reorder(self, reorder_list):
        """
        Reorder the current contents of wyrm_dict using either
        an ordered list of wyrm_dict

        :: INPUT ::
        :reorder_list: [list] unique list of keys from self.wyrm
        """
        # Ensure reorder_list is a list
        if not isinstance(reorder_list, list):
            raise TypeError('reorder_list must be type list')

        # Ensure reorder_list is a unique set
        tmp_in = []
        for _e in reorder_list:
            if _e not in tmp_in:
                tmp_in.append(_e)
        if tmp_in != reorder_list:
            raise ValueError('reorder_list has repeat entries - all entries must be unique')

        # Conduct reordering if checks are passed
        # Handle input (re)ordered wyrm_dict key list
        if all(_e in self.wyrm_dict.keys() for _e in reorder_list):
            tmp = {_e: self.wyrm_dict[_e] for _e in reorder_list}
        # Handle input (re)ordered index list
        elif all(_e in np.arange(0, len(reorder_list)) for _e in reorder_list):
            tmp_keys = list(self.wyrm_dict.keys())
            tmp = {_k: self.wyrm_dict[_k] for _k in tmp_keys}

        # Run updates
        self.wyrm_dict = tmp
        self.names = list(tmp.keys())
        return self


    def __str__(self, extended=False):
        rstr = super().__str__()
        rstr = "(wait: {self.wait_sec} sec)\n"
        for _i, (_k, _v) in enumerate(self.wyrm_dict.items()):
            # Provide index number
            rstr += f'({_i:<2}) '
            # Provide labeling of order
            if _i == 0:
                rstr += "(head) "
            elif _i == len(self.wyrm_dict) - 1:
                rstr += "(tail) "
            else:
                rstr += "  ||   "
            rstr += f"{_k} | "
            if extended:
                rstr += f'{_v.__str__()}\n'
            else:
                rstr += f'{type(_v)}\n'
        return rstr


    def pulse(self, x):
        """
        Initiate a chained pulse for elements of wyrm_queue.

        E.g.,
        tubewyrm.wyrm_dict = {name0:<wyrm0>,
                              name1:<wyrm1>,
                              name2:<wyrm3>}
        y = tubewyrm.pulse(x)
            is equivalent to
        y = wyrmw.pulse(wyrmq.pulse(wyrm0.pulse(x)))

        Between each successive wyrm in the wyrm_dict there
        is a pause of self.wait_sec seconds.

        :: INPUT ::
        :param x: Input `x` for the first Wyrm object in wyrm_dict

        :: OUTPUT ::
        :param y: Output `y` from the last Wyrm object in wyrm_dict
        """
        for _i, _wyrm in enumerate(self.wyrm_dict.values()):
            x = _wyrm.pulse(x)
            # if not last step, wait specified wait_sec
            if _i + 1 < len(self.wyrm_dict):
                sleep(self.wait_sec)
        y = x
        return y


class CanWyrm(TubeWyrm):
    """
    Child class of TubeWyrm.
    It's pulse(x) method runs the dict of *wyrm_n.pulse(x)'s
    sourcing inputs from a common input `x` and creating a dict
    of each wyrm_n.pulse(x)'s output `y_n`.

    NOTE: This class runs items in serial, but with some modification
    this would be a good candidate class for orchestraing multiprocessing.
    """

    def __init__(self,
                 wyrm_dict={},
                 wait_sec=0.,
                 max_pulse_size=1,
                 debug=False):
        """
        
        """
        # Handle some basic indexing/formatting
        if not isinstance(wyrm_dict, dict):
            if isinstance(wyrm_dict, (list, tuple)):
                wyrm_dict = dict(zip(range(len(wyrm_dict)), wyrm_dict))
            elif isinstance(wyrm_dict, Wyrm):
                wyrm_dict = {0: wyrm_dict}
            else:
                raise TypeError('wyrm_dict must be type dict, list, tuple, or Wyrm')
        
        # Initialize from Wyrm inheritance
        super().__init__(wyrm_dict=wyrm_dict, wait_sec=wait_sec, debug=debug, max_pulse_size=max_pulse_size)

        self.dict = {_k: None for _k in self.names}


    def pulse(self, x, **options):
        """
        Triggers the wyrm.pulse(x) method for each wyrm in wyrm_dict, sharing
        the same inputs, and writing outputs to self.dict[wyrmname] via the __iadd__
        method. I.e.,

            self.dict[wyrmname] += wyrm.pulse(x, **options)

        :: INPUTS ::
        :param x: [object] common input for all wyrms in wyrm_dict
        :param options: [kwargs] key word arguments passed to each wyrm's
                    wyrm.pulse(x, **options) method

        :: OUTPUT ::
        :return y: [dict] access to self.dict
        """
        for _k, _v in self.wyrm_dict.items():
            _y = _v.pulse(x, **options)
            # If this wyrm output has not been mapped to self.dict
            if self.dict[_k] is None:
                self.dict.update({_k: _y})
        y = self.dict
        return y




