"""
:module: ayahos.core.wyrms.bufferwyrm
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose: This module houses the definition of a Wyrm class that facilitates
        waveform buffering in the form of a WyrmStream object containing numerous
        MLTraceBuffer objects
"""

import logging
from numpy import isfinite
from ayahos.wyrms.wyrm import Wyrm, add_class_name_to_docstring
from ayahos.core.mltrace import MLTrace
from ayahos.core.mltracebuffer import MLTraceBuffer
from ayahos.core.dictstream import DictStream

Logger = logging.getLogger(__name__)

# @add_class_name_to_docstring
class BufferWyrm(Wyrm):
    """
    Class for buffering/stacking MLTrace objects into a DictStream of MLTraceBuffer objects with
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
            **add_kwargs):
        """Initialize a BufferWyrm object

        :param buffer_key: MLTrace attribute to use for the DictStream keys, defaults to 'id'
        :type buffer_key: str, optional
        :param max_length: Maximum MLTraceBuffer length in seconds, defaults to 300.
        :type max_length: positive float-like, optional
        :param restrict_past_append: Enforce MLTraceBuffer past append safeguards?, defaults to True
        :type restrict_past_append: bool, optional
        :param blinding: Apply blinding before appending MLTrace objects to MLTraceBuffer objects, defaults to None
                also see ayahos.core.trace.mltrace.MLTrace.apply_blinding
                Supported 
                    None - do not blind
                    (int, int) - left and right blinding sample counts (must be non-negative), e.g., (500, 500)
        :type blinding: None or 2-tuple of int, optional
        :param method: method to use for appending data to MLTrace-type objects in this BufferWyrm's buffer attribute, defaults to 1
                Supported values:
                    ObsPy-like
                        0, 'dis','discard' - discard overlapping dat
                        1, 'int','interpolate' - interploate between edge-points of overlapping data
                    SeisBench-like
                        2, 'max', 'maximum' - conduct stacking with "max" behavior
                        3, 'avg', 'average' - condcut stacking with "avg" behavior

                also see ayahos.core.trace.mltrace.MLTrace.__add__
                         ayahos.core.trace.mltracebuffer.MLTraceBuffer.append
        :type method: int or str, optional
        :param max_pulse_size: maximum number of items to pull from source deque (x) for a single .pulse(x) call for this BufferWyrm, defaults to 10000
        :type max_pulse_size: int, optional
        :param **add_kwargs: key word argument collector to pass to MLTraceBuffer's initialization **kwargs that in turn pass to __add__
                NOTE: add_kwargs with matching keys to pre-specified values will be ignored to prevent multiple call errors
        :type **add_kwargs: kwargs
        :raises TypeError: _description_
        :raises TypeError: _description_
        :raises ValueError: _description_
        """
        # Inherit from Wyrm
        super().__init__(max_pulse_size=max_pulse_size)

        # Initialize output of type ayahos.core.stream.dictstream.DictStream
        self.output = DictStream(key_attr=buffer_key)
        self.buffer_key = buffer_key
        # Create holder attribute for MLTraceBuffer initialization Kwargs
        self.mltb_kwargs = {}
        if isinstance(max_length, (float, int)):
            if isfinite(max_length):
                if 0 < max_length:
                    if max_length > 1e7:
                        Logger.warning(f'max_length of {max_length} > 1e7 seconds. May be memory prohibitive')
                    self.mltb_kwargs.update({'max_length': max_length})
                else:
                    raise ValueError('max_length must be non-negative')
            else:
                raise ValueError('max_length must be finite')
        else:
            raise TypeError('max_length must be type float or int')

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

        if method in [0,'dis','discard',
                          1,'int','interpolate',
                          2,'max','maximum',
                          3,'avg','average']:
            self.mltb_kwargs.update({'method': method})
        else:
            raise ValueError(f'method {method} not supported. See ayahos.trace.mltrace.MLTrace.__add__()')

        for _k, _v in add_kwargs:
            if _k not in self.mltb_kwargs:
                self.mltb_kwargs.update({_k: _v})                


    def add_new_buffer(self, other):
        """Add a new MLTraceBuffer object containing the contents of `other` to this BufferWyrm's .buffer attribute

        :param other: Trace-like object to use as the source (meta)data
        :type other: obspy.core.trace.Trace-like
        """        
        if isinstance(other, MLTrace):
            if other.id not in self.output.traces.keys():
                # Initialize an MLTraceBuffer Object with pre-specified append kwargs
                mltb = MLTraceBuffer(**self.mltb_kwargs)
                mltb.append(other)
                self.output.extend(mltb)
    
    #################################
    # PULSE POLYMORPHIC SUBROUTINES #
    #################################
                
    # Direct Inheritance from Wyrm
    # def _continue_iteration - input is non-empty deque and iterno + 1 < len(stdin)
    def pulse(self, stdin):
        stdout, nproc = super().pulse(stdin)
        if nproc > 0:
        #     Logger.info('nothing new buffered')
        # else:
            Logger.info(f'{nproc} tracebuff2 messages appended to {len(self.output)} buffers')
        return stdout, nproc

    def _get_obj_from_input(self, stdin):
        """_get_obj_from_input for BufferWyrm

        :param stdin: collection of MLTrace-like objects
        :type stdin: collections.deque of ayahos.core.trace.mltrace.MLTrace or list-like thereof
        :return obj: input object for _unit_process
        :rtype obj: list of ayahos.core.trace.mltrace.MLTrace
        """        
        obj = stdin.popleft()
        if isinstance(obj, MLTrace):
            obj = [obj]
        # if listlike, convert to list
        elif all(isinstance(x, MLTrace) for x in obj):
            obj = [x for x in obj]
        else:
            raise TypeError('stdin is not type MLTrace or a list-like thereof')
        return obj

    def _unit_process(self, obj):
        """_unit_process for BufferWyrm

        iterate across MLTraces in `obj` and either generate new
        MLTraceBuffers keyed by the MLTrace.id or append the MLTrace
        to an existing MLTraceBuffer with the same MLTrace.id.

        This method conducts output "capture" by generating new
        buffers or appending to existing buffers

        :param obj: iterable set of MLTrace objects
        :type obj: DictStream, list-like
        :return unit_output: standard output of _unit_process
        :rtype unit_output: None
        """        
        for _tr in obj:
            _key = getattr(_tr, self.buffer_key)
            if _key not in self.output.traces.keys():
                self.add_new_buffer(_tr)
            else:
                try:
                    self.output[_key].append(_tr)
                except TypeError:
                    breakpoint()
        unit_out = None
        return unit_out
    
    def _capture_unit_out(self, unit_out):
        """_capture_unit_out for BufferWyrm

        pass - output capture is handled by _unit_process for this class

        :param unit_out: unused
        :type unit_out: None
        :return status: unconditional True (do not trigger early stopping in pulse)
        :rtype status: bool
        """        
        status=True
        return status

                    
    # def __str__(self):
    #     rstr = f'ayahos.core.wyrms.bufferwyrm.BufferWyrm()'
    #     return rstr
    




    # def pulse(self, x):
    #     """Conduct a pulse of this BufferWyrm for up to self.max_pulse_size items contained in `x`. This method conducts early stopping if there are no items in `x` or all items in `x` have been assessed.

    #     items in `x` that are not consistent with expected inputs are re-appended to `x` using the `append()` method.

    #     WARNING: 
    #     As with most processing in Ayahos, this method removes items
    #     from `x` using the `popleft()` method and conducts in-place changes 
    #     on `y` (i.e., self.buffer). If you want to 

    #     :param x: deque of Trace-like objects (or Stream-like collections thereof) to append to this BufferWyrm's self.buffer attribute
    #     :type x: collections.deque of ayahos.core.trace.mltrace.MLTrace-like or ayahos.core.stream.dictstream.DictStream-like objects
    #     :return y: aliased access to this BufferWyrm's self.buffer attribute
    #     :rtype y: ayahos.core.stream.dictstream.DictStream
    #     """        
    #     # Kick error if input is not collections.deque object
    #     if not isinstance(x, deque):
    #         raise TypeError
    #     # Get initial deque length
    #     qlen = len(x)
    #     # Iterate up to max_pulse_size times
    #     for _i in range(self.max_pulse_size):
    #         # Early stopping if no items in queue
    #         if qlen == 0:
    #             break
    #         # Early stopping if next iteration would exceed
    #         # the number of items in queue
    #         elif _i + 1 > qlen:
    #             break
    #         # otherwise, popleft to get oldest item in queue
    #         else:
    #             _x = x.popleft()
    #         # if not an MLTrace (or child) reappend to queue (safety catch)
    #         if not isinstance(_x, (MLTrace, DictStream)):
    #             x.append(_x)
    #         #Otherwise get ID of MLTrace
    #         elif isinstance(_x, MLTrace):
    #             _x = [_x]
        
    #         for _xtr in _x:
    #             _id = _xtr.id
    #             # If the MLTrace ID is not in the WyrmStream keys, create new tracebuffer
    #             if _id not in self.buffer.traces.keys():
    #                 self.add_new_buffer(_xtr)
    #             # If the MLTrace ID is in the WyrmStream keys, use the append method
    #             else:
    #                 self.buffer[_id].append(_xtr, **self.add_kwargs)
    #     y = self.buffer
    #     return y
        