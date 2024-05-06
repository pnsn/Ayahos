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
import time, threading, copy
#import PyEW
import numpy as np
import pandas as pd
from collections import deque
from wyrm.core.wyrm import Wyrm
from wyrm.util.input import bounded_floatlike
from wyrm.core.mltrace import MLTrace
from wyrm.streaming.mltracebuffer import MLTraceBuffer
from wyrm.core.wyrmstream import WyrmStream

class CloneWyrm(Wyrm):
    """
    Submodule class that provides a pulsed method for producing (multiple) copies
    of an arbitrary set of items in an input deque into a dictionary of output deques
    """
    def __init__(self, queue_names=['A','B'], max_pulse_size=1000000, debug=False):
        """
        Initialize a CloneWyrm object
        :: INPUTS ::
        :param queue_names: [list-like] of values to assign as keys (names) to 
                            output deques held in self.queues
        :param max_pulse_size: [int] maximum number of elements to pull from 
                            an input deque in a single pulse
        :param debug: [bool] - should this be run in debug mode?
        """
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)
        self.nqueues = len(queue_names)
        self.queues = {}
        for _k in queue_names:
            self.queues.update({_k: deque()})
    
    def pulse(self, x):
        """
        Run a pulse on deque x where items are popleft'd out of `x`,
        deepcopy'd for each deque in self.queues and appended to 
        each deque in self.deques. The original object popped off
        of `x` is then deleted from memory as a clean-up step.

        Stopping occurs if `x` is empty (len = 0) or self.max_pulse_size
        items are cloned from `x` 

        :: INPUT ::
        :param x: [collections.deque] double ended queue containing N objects

        :: OUTPUT ::
        :return y: [dict] of [collections.deque] clones of contents popped from `x`
                with dictionary keys corresponding to queue_names elements.
        """
        if not isinstance(x, deque):
            raise TypeError('x must be type deque')
        for _ in range(self.max_pulse_size):
            if len(x) == 0:
                break
            _x = x.popleft()
            for _q in self.queues.values():
                _q.append(copy.deepcopy(_x))
            del _x
        y = self.queues
        return y