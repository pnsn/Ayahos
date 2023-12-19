"""
:module: wyrm.core.base_wyrms
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This module contains baseclass definitions for Wyrm class objects

:attribution:
    This module builds on the PyEarthworm (C) 2018 F. Hernandez interface
    between an Earthworm Message Transport system and Python distributed
    under an AGPL-3.0 license.

"""
import torch
import numpy as np
from collections import deque
from PyEW import EWModule





class RingWyrm(Wyrm):
    """
    Base class provides the basis for a one-way data flow element
    between an Earthworm Ring and the Python side of a PyEW.EWModule.
    Ths adds the following attributes


    This base class retains the vestigial pulse(x) class method of
    Wyrm.

    :: ATTRIBUTES :: 
    :attrib module: [PyEW.EWModule]


    """

    def __init__(self, module, conn_index, max_iter=1e6):
        # Run compatability checks on module
        if isinstance(module, PyEW.EWModule):
            self.module = module
        else:
            print("module must be a PyEW.EWModule object!")
        # Compatability check for conn_index
        try:
            self.conn_index = int(conn_index)
        except TypeError:
            print(f"conn_info is not compatable")
            raise TypeError

    def __repr__(self):
        rstr = "----- EW Connection -----\n"
        rstr += f"Module: {self.module}\n"
        rstr += f"Conn ID: {self.conn_index}\n"
        rstr += f"# Buffered: {len(self.queue_dict)}"
        return rstr

    def _change_conn_index(self, new_index):
        try:
            self.conn_index = int(new_index)
        except ValueError:
            raise ValueError


class TubeWyrm(Wyrm):
    """
    Base Class facilitating chained execution of pulse(x) class methods
    for a sequence wyrm objects, with each wyrm.pulse(x) taking the prior
    member's pulse(x) output as its input. 
    This `wyrm_queue` is a double ended queue (collections.deque), 
    which provides easier append/pop syntax for editing the wyrm_queue.
    """

    def __init__(self, wyrm_queue):
        """
        Create a tubewyrm object
        :: INPUT ::
        :param wyrm_list: [deque] or [list]
                            double ended queue of Wyrm objects
                            if list is provided, will be automatically
                            converted into a deque

        :: OUTPUT ::
        Initialized TubeWyrm object
        """
        # Run compatability checks on wyrm_list
        # If given a single Wyrm, wrap it in a deque
        if isinstance(wyrm_queue, Wyrm):
            self.wyrm_queue = deque([wyrm_queue])
        # If given a list of candidate wyrms, ensure they are all of Wyrm class
        elif isinstance(wyrm_queue, (list, deque)):
            if any(not isinstance(_wyrm, Wyrm) for _wyrm in wyrm_queue):
                print('Not all entries of wyrm_queue are type <class Wyrm>')
                raise TypeError
            # If all members are Wyrms, write to attribute
            else:
                self.wyrm_queue = wyrm_queue
            # Final check that the wyrm_queue is a deque
            if isinstance(wyrm_queue, list):
                self.wyrm_queue = deque(wyrm_queue)
                
        # In any other case:
        else:
            print('Provided wyrm_list was not a list or a Wyrm')
            raise TypeError

    def __repr__(self):
        rstr = '--- Tube ---\n'
        for _i, _wyrm in enumerate(self.wyrm_queue):
            if _i == 0:
                rstr += '(head) '
            else:
                rstr += '       '
            rstr += f'{type(_wyrm)}'
            if _i == len(self.wyrm_queue) - 1:
                rstr += ' (tail)'
            rstr += '\n'

    def _append(self, object, end='right'):
        """
        Convenience method for left/right append
        to wyrm_queue

        :: INPUTS ::
        :param object: [Wyrm] candidate wyrm object
        :param end: [str] append side 'left' or 'right'

        :: OUTPUT ::
        None
        """
        if isinstance(object, Wyrm):
            if end.lower() in ['right','r']:
                self.wyrm_list.append(object)
            elif end.lower() in ['left', 'l']
                self.wyrm_queue.appendleft(object)
        
        if isinstance(object, (list, deque)):
            if all(isinstance(_x, Wyrm) for _x in object):
                if end.lower() in ['right', 'r']:
                    self.wyrm_list += deque(object)
                elif end.lower() in ['left', 'l']:
                    self.wyrm_list = deque(object) + self.wyrm_list

        
    def _pop(self, end='right'):
        """
        Convenience method for left/right pop
        from wyrm_queue

        :: INPUT ::
        :param end: [str] 'left' or 'right'
        
        :: OUTPUT ::
        :param x: [Wyrm] popped Wyrm object from
                wyrm_queue
        """
        if end.lower() in ['right', 'r']:
            x = self.wyrm_list.pop()
        elif end.lower() in ['left','l']:
            x = self.wyrm_list.popleft()
        return x
    
    def pulse(self, x):
        """
        Initiate a chained pulse for elements of 
        wyrm_queue. 

        E.g., 
        tubewyrm.wyrm_queue = [<wyrm1>, <wyrm2>, <wyrm3>]
        y = tubewyrm.pulse(x) 
            is equivalent to 
        y = wyrm3.pulse(wyrm2.pulse(wyrm1.pulse(x)))

        :: INPUT ::
        :param x: Input `x` for the first Wyrm object in wyrm_queue
        
        :: OUTPUT ::
        :param y: Output `y` from the last Wyrm object in wyrm_queue 
        """
        for _wyrm in self.wyrm_list:
            x = _wyrm.pulse(x)
        y = x
        return y


class CanWyrm(TubeWyrm):
    """
    Base class for running an iterable list of Wyrms that takes a single
    input datasource and gathers the outputs of 
    """
    # Inherits __init__ from TubeWyrm
    def __init__(self, wyrm_queue=deque([]), output_method=deque):
        super().__init__(wyrm_queue=wyrm_queue)
        self.output_format = output_method

    def __repr__(self):
        rstr = '~~~ CanWyrm ~~~'
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
        

class TorchWyrm(Wyrm):
    """
    BaseClass for generalized handling of PyTorch prediction work
    and minimal handling of
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __repr__(self):
        rstr = f"Device: {self.device}\n"
        rstr += f"Model Component Order: {self.model.component_order}\n"
        rstr += f"Model Prediction Classes: {self.model.classes}\n"
        rstr += f"Model Citation\n{self.model.citation}"
        return rstr

    def _send_model_to_device(self):
        if self.model.device != self.device:
            self.model.to(self.device)

    def _send_data_to_device(self, x):
        # If the data presented are not already in a torch.Tensor format
        # but appear that they can be, convert.
        if not isinstance(x, torch.Tensor) and isinstance(x, np.ndarray):
            x = torch.Tensor(x)
            x.to(self.device)
            return x
        # If the data are already a torch.Tensor, pass input to output
        elif isinstance(x, torch.Tensor):
            x.to(self.device)
            return x
        # For all other cases, raise TypeError
        else:
            raise TypeError

    def _run_prediction(self, x):
        """
        Execute a PyTorch model prediction

        :: INPUT ::
        :param x: [torch.Tensor] input data tensor

        :: RETURN ::
        :param y: [torch.Tensor] output prediction tensor
        """
        y = self.model(x)
        return y

    def pulse(self, x, ewmod=None, connidx=None):
        """
        Run a prediction on input data tensor.

        :: INPUT ::
        :param x: [torch.Tensor] or [numpy.ndarray]
                pre-processed data with appropriate dimensionality for specified model.
                e.g.,
                for PhaseNet: x.shape = (nwind, chans, time)
                                        (nwind, 3, 3000)
                for EQTransformer: x.shape = (time, chans, nwind)
                                             (6000, 3, nwind)
        :: OUTPUT ::
        :return y: [torch.Tensor] predicted values for specified model
            e.g.
            PhaseNet: y = (nwind, [P(tP(t)), P(tS(t)), P(Noise(t))], t)
            EQTransformer: y = (t, [P(Detection(t)), P(tP(t)), P(tS(t))], nwind)

        """
        # Ensure model is on the specified device
        self._send_model_to_device()
        # Ensure data are in Tensor format and on the specified device
        self._send_data_to_device(x)
        # Run prediction
        y = self._run_prediction(x)
        # Return raw output of prediction
        return y
