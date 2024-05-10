"""
:module: wyrm.core.wyrms.pick2kwyrm
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module houses class definitions for handling data Input/Output
    from the Earthworm Message Transport System (memory rings)

    Class

    Pick2kWyrm - primary wave-fetching submodule for getting sets of tracebuff2 messages
            from a single WAVE RING using the PyEW.EWModule class, converts python-side
            `wave` messages into MLTrace objects, and appends these traces to MLTraceBuffer
            objects contained in a WyrmStream object
"""
import logging
import PyEW
from wyrm.core.wyrms.ringwyrm import RingWyrm
from wyrm.util.pyew import is_wave_msg, wave2mltrace
from wyrm.util.input import bounded_floatlike, bounded_intlike
from wyrm.core.trace.mltrace import MLTrace, MLTraceBuffer
from wyrm.core.stream.wyrmstream import WyrmStream

Logger = logging.getLogger(__name__)

class Pick2kWyrm(RingWyrm):
    """
    Wrapper child-class of RingWyrm specific to listening to an Earthworm
    WAVE Ring and populating MLTraceBuffers housed in a WyrmStream
    """ 

    def __init__(
        self,
        module=None,
        conn_id=0,
        max_length=150,
        restrict_past_append=True,
        wyrmstream_kwargs={},
        mltrace_kwargs={},
        max_pulse_size=12000,
    ):
        """_summary_

        :param module: _description_, defaults to None
        :type module: _type_, optional
        :param conn_id: _description_, defaults to 0
        :type conn_id: int, optional
        :param max_length: _description_, defaults to 150
        :type max_length: int, optional
        :param restrict_past_append: _description_, defaults to True
        :type restrict_past_append: bool, optional
        :param wyrmstream_kwargs: _description_, defaults to {}
        :type wyrmstream_kwargs: dict, optional
        :param mltrace_kwargs: _description_, defaults to {}
        :type mltrace_kwargs: dict, optional
        :param max_pulse_size: _description_, defaults to 12000
        :type max_pulse_size: int, optional
        :raises TypeError: _description_
        :raises TypeError: _description_
        :raises TypeError: _description_
        """        
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
        # Inherit from RingWyrm
        super().__init__(
            module=module,
            conn_id=conn_id,
            pulse_method_str='get_wave',
            msg_type=19,
            max_pulse_size=max_pulse_size,
            )
        Logger.info('init EarWyrm')
        # max_length compatability check (runs prior to init of a MLTraceBuffer to sanity check)
        self.max_length = bounded_floatlike(
            max_length,
            name='max_length',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        if not isinstance(restrict_past_append, bool):
            raise TypeError('restrict_past_append must be type bool')
        else:
            self.rpa = restrict_past_append

        # accept options as-is and let MLTrace.__add__ catches sort this out
        if not isinstance(mltrace_kwargs, dict):
            raise TypeError
        else:
            self.mltrace_kwargs = mltrace_kwargs
        # accept options as-is and let WyrmStream.__add__ catches sort this out
        if not isinstance(wyrmstream_kwargs, dict):
            raise TypeError
        else:
            self.wyrmstream_kwargs = wyrmstream_kwargs
        # basic check on WyrmStream.__add__ 

        # Initialize WyrmStream
        self.buffer = WyrmStream()
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
                # If MLTrace id is not in the WyrmStream keys
                if mlt.id not in self.buffer.keys():
                    # Initialize a MLTraceBuffer
                    mltb = MLTraceBuffer(
                        max_length=self.max_length,
                        restrict_past_append=self.restrict_past_append)
                    # Append 
                    mltb.__add__(mlt, **self.mltrace_kwargs)
                    self.buffer.__add__(mlt, key_attr='id', **self.wyrmstream_kwargs)
                else:
                    self.buffer.__add__(mlt, key_attr='id', **self.wyrmstream_kwargs)
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
        rstr += f'mltrace_kwargs={self.mltrace_kwargs}, wyrmstream_kwargs={self.wyrmstream_kwargs}, '
        rstr += f'max_pulse_size={self.max_pulse_size}, debug={self.debug}'
        for _k, _v in self.options.items():
            rstr += f', {_k}={_v}'
        rstr += ')'
        return rstr