"""
:module: wyrm.io.ew_ring
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module houses class definitions for handling data Input/Output
    from the Earthworm Message Transport System (memory rings)

    Classes

    RingWyrm - submodule for handling individual transactions between the Python
                and Earthworm environment, built on the PyEarthworm package
                (PyEW.EWModule). This provides access to the full range of get_
                and put_ class methods for a EWModule.

    EarWyrm - primary wave-fetching submodule for getting sets of tracebuff2 messages
            from a single WAVE RING using the PyEW.EWModule class, converts python-side
            `wave` messages into MLTrace objects, and appends these traces to MLTraceBuffer
            objects contained in a WyrmStream object
    
    TODO: UNDER DEVELOPMENT
    BookWyrm - primary message-submitting submodule for sending pick2k messages to a
            single PICK RING

"""
import logging
#import PyEW
from wyrm.core.wyrm import Wyrm
from wyrm.util.pyew import is_wave_msg, wave2mltrace
from wyrm.util.input import bounded_floatlike, bounded_intlike
from wyrm.core.mltrace import MLTrace, MLTraceBuffer
from wyrm.core.wyrmstream import WyrmStream

Logger = logging.getLogger(__name__)

class RingWyrm(Wyrm):
    """
    Wyrm that facilitates transactions between memory rings in the Earthworm
    Message Transport System and the Python environment. This wraps an active 
    PyEarthworm (PyEW) module and a single python-ring connection and provides
    an abstract RingWyrm.pulse() method that facilitates the following PyEW.EWModule
    class methods:
        + get_wave() - get TYPE_TRACEBUFF2 (msg_type 19) messages from a WAVE RING
        + put_wave() - submit a `wave` dict object to a WAVE RING (msg_type 19)
        + get_msg() - get a string-formatted message* from a RING
        + put_msg() - put a string-formatted message* onto a RING
        + get_bytes() - get a bytestring* from a RING
        + put_bytes() - put a bytestring* onto a RING

        *with appropriate msg_type code
    """
    
    def __init__(
            self,
            module=None,
            conn_id=0,
            pulse_method_str='get_wave',
            msg_type=19,
            max_pulse_size=10000
            ):
        
        Wyrm.__init__(self, max_pulse_size=max_pulse_size)
        Logger.debug('init RingWyrm')
        # Compatability checks for `module`
        if module is None:
            self.module = module
            print('No EW connection provided - for debugging/dev purposes only')
        elif not isinstance(module, PyEW.EWModule):
            raise TypeError('module must be a PyEW.EWModule object')
        else:
            self.module = module

        # Compatability checks for `conn_id`
        self.conn_id = self._bounded_intlike_check(conn_id, name='conn_id', minimum=0)

        # Compat. chekcs for pulse_method_str
        if not isinstance(pulse_method_str, str):
            raise TypeError('pulse_method_str must be type_str')
        elif pulse_method_str not in ['get_wave','put_wave','get_msg','put_msg','get_bytes','put_bytes']:
            raise ValueError(f'pulse_method_str {pulse_method_str} unsupported. See documentation')
        else:
            self.pulse_method = pulse_method_str
        
        # Compatability checks for msg_type
        if self.pulse_method in ['get_msg','put_msg','get_bytes','put_bytes']:
            self.msg_type = bounded_intlike(
                msg_type,
                name='msg_type',
                minimum=0,
                maximum=255,
                inclusive=True
            )
        # In the case of get/put_wave, default to msg_type=19 (tracebuff2)
        else:
            self.msg_type=19
        Logger.info('Ringwyrm method {0} for message type {1}'.format(self.pulse_method, self.msg_type))

    def pulse(self, x):
        """
        Conduct a single transaction between an Earthworm ring
        and the Python instance this RingWyrm is operating in
        using the PyEW.EWModule.get/put -type method and msg_type
        assigned to this RingWyrm

        :: INPUT ::
        :param x: for 'put' type pulse_method_str instances of RingWyrm
                this is a message object formatted to submit to target 
                Earthworm ring following PyEW formatting guidelines
                see
        :: OUTPUT ::
        :return msg: for 'get' type pulse_method_str instances of RingWyrm
                this is a message object produced by a single call of the
                PyEW.EWModule.get_... method specified when the RingWyrm
                was initialized. 
                
                NOTE: Empty messages return False to signal no new messages 
                were available of specified msg_type in target ring conn_id.

                for get_wave() - returns a dictionary
                for get_bytes() - returns a python bytestring
                for get_msg() - returns a python string
        """
        # If getting things from Earthworm...
        if 'get' in self.pulse_method:
            # ...if getting a TYPE_TRACEBUFF2 (code 19) message, use class method directly
            if 'wave' in self.pulse_method:
                msg = self.module.get_wave(self.conn_id)
                # Flag null result as False
                if msg == {}:
                    msg = False
            # ...if getting a string or bytestring, use eval approach for compact code
            else:
                eval_str = f'self.module.{self.pulse_method}(self.conn_id, self.msg_type)'
                msg = eval(eval_str)
                # Flag empty message results as 'False'
                if msg == '':
                    msg = False
            
            return msg
        
        # If sending things to Earthworm...
        elif 'put' in self.pulse_method:
            msg = x
            # ..if sending waves
            if 'wave' in self.pulse_method and is_wave_msg(msg):
                # Commit to ring
                self.module.put_wave(self.conn_id, msg)
            # If sending byte or string messages
            else:
                # Compose evalstr
                eval_str = f'self.module.{self.pulse_method}(self.conn_id, self.msg_type, x)'
                # Execute eval
                eval(eval_str)
            return None      

    def __repr__(self):
        """
        Provide a string representation of this RingWyrm object

        :: OUTPUT ::
        :return rstr: [str] representative string
        """
        # Print from Wyrm
        rstr = super().__repr__()
        # Add lines for RingWyrm
        rstr += f'\nModule: {self.module} | Conn ID: {self.conn_id} | '
        rstr += f'Method: {self.pulse_method} | MsgType: {self.msg_type}'
        return rstr
    
    def __str__(self):
        rstr = f'wyrm.wyrms.io.RingWyrm(module={self.module}, '
        rstr += f'conn_id={self.conn_id}, pulse_method_str={self.pulse_method}, '
        rstr += f'msg_type={self.msg_type}, '
        rstr += f'max_pulse_size={self.max_pulse_size}, debug={self.debug})'
        return rstr


class EarWyrm(RingWyrm):
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