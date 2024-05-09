"""
:module: wyrm.core.wyrm.bufferwyrm
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose: This module houses the definition of a Wyrm class that facilitates
        waveform buffering in the form of a WyrmStream object containing numerous
        MLTraceBuffer objects
"""

import time
from collections import deque
from wyrm.core.wyrm import Wyrm
from wyrm.core.mltrace import MLTrace
from wyrm.streaming.mltracebuffer import MLTraceBuffer
from wyrm.core.wyrmstream import WyrmStream
from wyrm.util.input import bounded_floatlike

class BufferWyrm(Wyrm):
    """
    Class for buffering/stacking MLTrace objects into a WyrmStream of MLTraceBuffer objects with
    self-contained settings and sanity checks for changing the MLTraceBuffer.__add__ method options.
    """
    def __init__(
            self,
            buffer_key='id',
            max_length=300.,
            restrict_past_append=True,
            blinding=None,
            method=1,
            max_pulse_size=10000,
            debug=False,
            **add_kwargs):
        """
        Initialize a BufferWyrm object that contains attributes that house a WyrmStream that comprises 
        MLTraceBuffer objects keyed by trace IDs. 
        

        :: INPUTS ::
        ~~~MLTraceBuffer Initilizing Inputs~~~
        :param max_length: [float] maximum MLTraceBuffer length in seconds
        :param restrict_past_add: [bool] - enforce restrictions on appending data that temporally preceed data contained
                            within MLTraceBuffer objects? 
                            ~also see wyrm.data.mltracebuffer.MLTraceBuffer.append()
        :param blinding: [None], [int], or [tuple of int] number of samples on the left and
                            right end of appended traces to "blind" (i.e. set fold = 0)
                            ~also see wyrm.data.mltrace.MLTrace.apply_blinding() for specific behviors
        ~~~Buffering Append Behavior Influencing Attributes~~~
        :param method: [int] or [str] - method for wyrm.data.mltrace.MLTrace.__add__
                        Supported:
                            0, 'dis','discard' - discard overlapping dat
                            1, 'int','interpolate' - interploate across overlapping data
                            2, 'max', 'maximum' - conduct stacking with "max" behavior
                            3, 'avg', 'average' - condcut stacking with "avg" behavior
                        ~also see wyr.data.mltrace.MLTrace.__add__()
        :param **add_kwargs: [kwargs] key-word argument collector passed to MLTrace.__add__() to alter
                            behaviors of calls of MLTraceBuffer.append() made in the BufferWyrm.pulse() method
                            ~also see wyrm.data.mltrace.MLTrace.__add__()
                            ~also see wyrm.data.mltracebuffer.MLTraceBuffer.append()

        :param max_pulse_size: [int] maximum number of items in `x` to assess in a single call of BufferWyrm.pulse(x)
        :param debug: [bool] - run in debug mode?

        """
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)
        # Initialize Buffer Containing Object
        self.buffer = WyrmStream(key_attr=buffer_key)
        # Create holder attribute for MLTraceBuffer initialization Kwargs
        self.mltb_kwargs = {}
        # Do sanity checks on max_length
        self.mltb_kwargs.update({'max_length':
                                 bounded_floatlike(
                                        max_length,
                                        name='max_length',
                                        minimum = 0,
                                        maximum=None,
                                        inclusive=False
                                        )})
        # Do sanity checks on restrict_past_append
        if not isinstance(restrict_past_append, bool):
            raise TypeError('restrict_past_append must be type bool')
        else:
            self.mltb_kwargs.update({'restrict_past_append': restrict_past_append})
        # Do sanity checks on blinding
        if not isinstance(blinding, (type(None), bool, tuple)):
            raise TypeError
        else:
            self.mltb_kwargs.update({'blinding': blinding})
        # Create holder for add_kwargs to pass to 
        self.add_kwargs = add_kwargs
        if method in [0,'dis','discard',
                          1,'int','interpolate',
                          2,'max','maximum',
                          3,'avg','average']:
            self.add_kwargs.update({'method': method})
        else:
            raise ValueError(f'add_method {method} not supported. See wyrm.data.mltrace.MLTrace.__add__()')
        


    def add_new_buffer(self, other):
        if isinstance(other, MLTrace):
            if other.id not in self.buffer.traces.keys():
                # Initialize an MLTraceBuffer Object
                mltb = MLTraceBuffer(**self.mltb_kwargs,
                                     **self.add_kwargs)
                mltb.append(other)
                self.buffer.extend(mltb)
            

    def pulse(self, x):
        """
        Conduct a pulse on input deque of MLTrace objects (x)
        """
        # Kick error if input is not collections.deque object
        if not isinstance(x, deque):
            raise TypeError
        # Get initial deque length
        qlen = len(x)
        # Iterate up to max_pulse_size times
        for _i in range(self.max_pulse_size):
            # Early stopping if no items in queue
            if qlen == 0:
                break
            # Early stopping if next iteration would exceed
            # the number of items in queue
            elif _i + 1 > qlen:
                break
            # otherwise, popleft to get oldest item in queue
            else:
                _x = x.popleft()
            # if not an MLTrace (or child) reappend to queue (safety catch)
            if not isinstance(_x, (MLTrace, WyrmStream)):
                x.append(_x)
            #Otherwise get ID of MLTrace
            elif isinstance(_x, MLTrace):
                _x = [_x]
        
            for _xtr in _x:
                tick = time.time()
                _id = _xtr.id
                # If the MLTrace ID is not in the WyrmStream keys, create new tracebuffer
                if _id not in self.buffer.traces.keys():
                    self.add_new_buffer(_xtr)
                    tock = time.time()
                    status = 'new'
                # If the MLTrace ID is in the WyrmStream keys, use the append method
                else:
                    self.buffer[_id].append(_xtr, **self.add_kwargs)
                    tock = time.time()
                    status = 'append'
                # print(f'{_i:05} | {_id} | {status} took {tock - tick} sec')
        y = self.buffer
        return y
        
                    
    # def __repr__(self, extended=False):
    #     rstr = f'Add Style: {self.add_style}'
    #     return rstr
                    
    def __str__(self):
        rstr = f'wyrm.core.coordinate.BufferWyrm()'
        return rstr