"""
:module: wyrm.core.wyrms.wavewyrm
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module houses class definitions for handling data Input/Output
    from the Earthworm Message Transport System (memory rings)

    Class

    WaveWyrm - primary wave-fetching submodule for getting sets of tracebuff2 messages
            from a single WAVE RING using the PyEW.EWModule class, converts python-side
            `wave` messages into MLTrace objects, and appends these traces to MLTraceBuffer
            objects contained in a DictStream object
"""
import logging
import PyEW
from wyrm.core.wyrms.ringwyrm import RingWyrm
from wyrm.util.pyew import is_wave_msg, wave2mltrace
from wyrm.util.input import bounded_floatlike, bounded_intlike
from wyrm.core.trace.mltrace import MLTrace, MLTraceBuffer
from wyrm.core.stream.dictstream import DictStream

Logger = logging.getLogger(__name__)

class WaveWyrm(RingWyrm):
    """
    Wrapper child-class of RingWyrm specifi
    WAVE Ring and populating MLTraceBuffers housed in a DictStream
    """

    def __init__(
        self,
        module=None,
        conn_id=0,
        flow_direction='to_py',
        buffer_type=DictStream,
        buffer_append_method='__add__',
        buffer_append_kwargs={}
        element_type=MLTraceBuffer,
        element_append_method='__add__',
        element_append_kwargs={}
        buffer_kwargs = {},
        element_kwarg = {},
        max_pulse_size=1e6):
        """
        Initialize a EarWyrm object with a TieredBuffer + TraceBuff.
        This is a wrapper for a read-from-wave-ring RingWyrm object

        :: INPUTS ::
        :param module: [PyEW.EWModule] active PyEarthworm module object
        :param conn_id: [int] index number for active PyEarthworm ring connection
        :param max_length: [float] maximum MLTraceBuffer length in seconds
        :param max_pulse_size: [int] maximum number of get_wave() actions to execute per
                        pulse of this RingWyrm
        :param **options: [kwargs] additional kwargs to pass to MLTraceBuffer objects'
                            __add__() method.
        """
        if flow_direction.lower() in ['to_py', 'from_ew']:
            pulse_method_str = 'get_wave'
        elif flow_direction.lower() in ['to_ew','from_py']:
            pulse_method_str = 'put_wave'

        # Inherit from RingWyrm
        super().__init__(
            module=module,
            conn_id=conn_id,
            pulse_method_str=pulse_method_str,
            msg_type=19,
            max_pulse_size=max_pulse_size,
            )
        
        if buffer_type in [deque, DictStream]:
            self.buffer = buffer_type()
        else:
            raise ValueError('buffer_type {buffer_type} not supported')
        
        # Initialize WyrmStream
        self.buffer = DictStream()
        Logger.debug('init EarWyrm')

    def pulse(self, x=None):
        """
        Execute a pulse that pulls up to self.max_pulse_size tracebuff2 messages from
        a connected WAVE RING into python as a PyEarthworm `wave` message, which is converted
        into a wyrm.core.trace.MLTrace object and appended to a wyrm.core.trace.MLTraceBuffer
        object (or seeds a new MLTraceBuffer object) contained in the EarWyrm.buffer attribute
        (a wyrm.core.WyrmStream.WyrmStream object) keyed on a given MLTrace(Buffer)'s `id`.

        Early stopping occurrs if the inherited RingWyrm.pulse() method returns a `False`
        output, signifying that a "no new messages" signal was received by the 
        PyEW.EWModule.get_wave() method around which these Wyrm classes wrap.

        :: INPUT ::
        :param x: [NoneType] - inputs to x are not use here. This is included to conform
                    with the general template for the wyrm.core._base.Wyrm base class
        
        :: OUTPUT ::
        :return y: [wyrm.core.WyrmStream.WyrmStream] access to the WyrmStream object
                    contained in this EarWyrm's self.buffer attribute.
        """
        # Iterate up to max_pulse_size
        for _i in range(self.max_pulse_size):
            # Run pulse method from get_wave() formatted RingWyrm to get single wave
            _wave = super().pulse()
            # Early stopping if RingWyrm.pulse() returns False
            if not _wave:
                Logger.debug('early stop - no new entries')
                break
            # Otherwise
            else:
                # Convert wave to MLTrace
                mlt = wave2mltrace(_wave)
                # If MLTrace id is not in the DictStream keys
                if mlt.id not in self.buffer.keys():
                    # Initialize a MLTraceBuffer
                    mltb = MLTraceBuffer(
                        max_length=self.max_length,
                        restrict_past_append=self.restrict_past_append)
                    # Append mltrace to new mltracebuffer object
                    mltb.__add__(mlt, **self.mltrace_kwargs)
                    # Update the buffer attribute with the new mltracebuffer object and its ID
                    self.buffer.__add__(mlt, key_attr='id', **self.dictstream_kwargs)
                # If MLTrace ID is in the DictStream keys
                else:
                    self.buffer.__add__(mlt, key_attr='id', **self.dictstream_kwargs)
        Logger.debug('pulse ran for {0} iterations'.format(_i))
        y = self.buffer
        return y
        
    def __repr__(self, extended=False):
        """
        Return representative, user-friendly string that details the
        contents of this EarWyrm
        
        :: INPUT ::
        :param extended: [bool] should the TieredDict
        
        """
        # Populate information from RingWyrm.__repr__
        rstr = super().__repr__()
        # Add __repr__ from WyrmStream in self.buffer
        rstr += f'\n{self.buffer.__repr__(extended=extended)}'
        return rstr

    def __str__(self):
        """
        Return descriptive string of how this EarWyrm was initialized
        """
        rstr = f'wyrm.wyrms.io.EarWyrm(module={self.module}, '
        rstr += f'conn_id={self.conn_id}, max_length={self.max_length}, '
        rstr += f'mltrace_kwargs={self.mltrace_kwargs}, dictstream_kwargs={self.dictstream_kwargs}, '
        rstr += f'max_pulse_size={self.max_pulse_size}, debug={self.debug}'
        for _k, _v in self.options.items():
            rstr += f', {_k}={_v}'
        rstr += ')'
        return rstr