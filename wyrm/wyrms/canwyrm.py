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
from wyrm.wyrms.tubewyrm import TubeWyrm
from collections import deque


class CanWyrm(TubeWyrm):
    """
    Child class of TubeWyrm.
    It's pulse(x) method runs the queue of *wyrm_n.pulse(x)'s
    sourcing inputs from a common input `x` and creating a queue
    of each wyrm_n.pulse(x)'s output `y_n`.

    NOTE: This class runs items in serial, but with some modification
    this would be a good candidate class for orchestraing multiprocessing.
    """

    # Inherits __init__ from TubeWyrm
    def __init__(self, wyrm_queue=deque([]), output_method=deque):
        super().__init__(wyrm_queue=wyrm_queue)
        self.output_format = output_method

    def __repr__(self):
        rstr = "~~~ CanWyrm ~~~"
        rstr += super().__repr__()
        return rstr

    def pulse(self, x):
        """
        Iterate across wyrms in wyrm_queue that all feed
        from the same input variable x and gather each
        iteration's output y in a deque assembled with an
        appendleft() at the end of each iteration

        :: INPUT ::
        :param x: [variable] Single variable that every
                    Wyrm (or the head-wyrm in a tubewyrm)
                    in the wyrm_queue can accept as a
                    pulse(x) input.

        :: OUTPUT ::
        :return y: [deque] or [self.output_method]
                    Serially assembled outputs of each Wyrm
                    (or the tail-wyrm in a tubewyrm)
        """
        y = self.output_format()
        for _wyrm in self.wyrm_queue:
            _y = _wyrm.pulse(x)
            _y = self.output_format(_y)
            y.appendleft(_y)
        return y
