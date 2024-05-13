"""
:module: wyrm.core.wyrms.ringwyrm
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
    
    TODO: UNDER DEVELOPMENT
    BookWyrm - primary message-submitting submodule for sending pick2k messages to a
            single PICK RING

"""
import logging
#import PyEW
from wyrm.core.wyrms.wyrm import Wyrm
from wyrm.util.pyew import is_wave_msg, wave2mltrace
from wyrm.util.input import bounded_floatlike, bounded_intlike
from wyrm.core.trace.mltrace import MLTrace
from wyrm.core.trace.mltracebuffer import MLTraceBuffer
from wyrm.core.stream.dictstream import DictStream

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