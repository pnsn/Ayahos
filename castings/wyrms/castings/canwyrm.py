"""
:module: wyrm.wyrms.canwyrm
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose: 
    Child class of TubeWyrm.
    It's pulse(x) method runs the queue of *wyrm_n.pulse(x)'s
    sourcing inputs from a common input `x` and creating a queue
    of each wyrm_n.pulse(x)'s output `y_n`.

NOTE: This class runs items in serial, but with some modification
this would be a good candidate class for orchestraing multiprocessing.
"""
from wyrm.wyrms.sequence import TubeWyrm
from collections import deque
from time import sleep


