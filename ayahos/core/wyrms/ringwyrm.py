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
import PyEW
from collections import deque
from ayahos.core.wyrms.wyrm import Wyrm

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
            module,
            conn_id=0,
            pulse_method='get_wave',
            msg_type=19,
            max_pulse_size=10000
            ):
        """_summary_

        :param module: Pre-initialized PyEarthworm module
        :type module: PyEW.EWModule, optional
        :param conn_id: connection ID number for target memory ring, defaults to 0
        :type conn_id: int, optional
            also see    PyEW.EWModule
                        ayahos.core.wyrms.heartwyrm.HeartWyrm
        :param pulse_method: name of PyEW.EWModule messaging method to use, defaults to 'get_wave'
        :type pulse_method: str, optional
            Supported: 'get_wave','put_wave','get_msg','put_msg','get_bytes','put_bytes'
            also see    PyEW.EWModule
        :param msg_type: Earthworm message code, defaults to 19 (TYPE_TRACEBUFF2)
        :type msg_type: int, optional
            cross reference with your installation's `earthworm_global.d` file
        :param max_pulse_size: Maximum mumber of messages to transact in a single pulse, defaults to 10000
        :type max_pulse_size: int, optional
        """        
        Wyrm.__init__(self, max_pulse_size=max_pulse_size)
        Logger.debug('init RingWyrm')
        # Compatability checks for `module`
        if isinstance(module, PyEW.EWModule):
            self.module = module
        else:
            raise TypeError(f'module must be type PyEW.EWModule, not type {type(module)}')
        
        if isinstance(conn_id, int):
            if conn_id > 0:
                self.conn_id = conn_id
            else:
                raise ValueError('conn_id must be a non-negative iteger')
        else:
            raise TypeError('conn_id must be type int')

        # Compatability or pulse_method
        if pulse_method not in ['get_wave','put_wave','get_msg','put_msg','get_bytes','put_bytes']:
            raise ValueError(f'pulse_method {pulse_method} unsupported. See documentation')
        else:
            self.pulse_method = pulse_method
        
        # Compatability checks for msg_type
        if isinstance(msg_type, int):
            if self.pulse_method in ['get_msg','put_msg','get_bytes','put_bytes']:
                if 0 <= msg_type <= 255:
                    self.msg_type = msg_type
                else:
                    raise ValueError(f'msg_type value {msg_type} out of bounds [0, 255]')
            # Hard set TYPE_TRACEBUFF2 for get/put_wave
            else:
                self.msg_type = 19
        self._core_args = [self.conn_id, self.msg_type]

        Logger.info('Ringwyrm method {0} for message type {1}'.format(self.pulse_method, self.msg_type))

    def _core_process(self, x):
        """_core_process for ayahos.core.wyrms.ringwyrm.RingWyrm

        Conduct a single transaction based on the following RingWyrm attributes
        .pulse_method
        .msg_type
        .conn_id

        :param x: for 'put_*' methods - collection of message-like objects to submit to ring 
                  for 'get_*' methods - None
        :type x: collections.deque or None
        :return y: for 'put_*' methods - None
                   for 'get_*' methods - collection of message objects
        :rtype: None or collections.deque
        """        
        if 'put' in self.pulse_method:
            if not isinstance(x, deque):
                raise 
            _x = x.popleft()
            _y = getattr(self.module, self.pulse_method)(*self._core_args, _x)
        elif 'get' in self.pulse_method:
            _y = getattr(self.module, self.pulse_method)(*self._core_args)
        return _y

    def _get_early_stopping(self, _y):

        if self.pulse_method == 'get_msg':
            if _y == '':
                return True
            else:
                return False
        if self.pulse_method == 'get_bytes':
            if _y == (0,0):
                return True
            else:
                return False
        if self.pulse_method == 'get_wave':
            if _y == {}:
                return True
            else:
                return False

    def _put_early_stopping(self, x, i_):
        """Use _early_stopping from ayahos.core.wyrms.wyrm.Wyrm for "put" type pulse methods

        Early stopping if iteration counter exceeds length of `x`
        Early stopping if length of `x` is 0

        :param x: collection of messages being submitted
        :type x: collections.deque
        :param i_: iteration counter
        :type i_: int
        :return: status
                True = Trigger early stopping
                False = do not trigger early stopping
        :rtype: bool
        """        
        status = super()._early_stopping(x, i_)
        return status

    def pulse(self, x):
        if 'get' in self.pulse_method:
            for i_ in range(self.max_pulse_size):
                _y = self._core_process(x)
                self.output.append(_y)
                if self._get_early_stopping(x):
                    break
            y = self.output
        elif 'put' in self.pulse_method:
            for i_ in range(self.max_pulse_size):
                if self._put_early_stopping(x, i_):
                    break
                self._core_process(x)
            y = None
        return y


    # def pulse(self, x):
    #     """
    #     Conduct a single transaction between an Earthworm ring
    #     and the Python instance this RingWyrm is operating in
    #     using the PyEW.EWModule.get/put -type method and msg_type
    #     assigned to this RingWyrm

    #     :: INPUT ::
    #     :param x: for 'put' type pulse_method_str instances of RingWyrm
    #             this is a message object formatted to submit to target 
    #             Earthworm ring following PyEW formatting guidelines
    #             see
    #     :: OUTPUT ::
    #     :return msg: for 'get' type pulse_method_str instances of RingWyrm
    #             this is a message object produced by a single call of the
    #             PyEW.EWModule.get_... method specified when the RingWyrm
    #             was initialized. 
                
    #             NOTE: Empty messages return False to signal no new messages 
    #             were available of specified msg_type in target ring conn_id.

    #             for get_wave() - returns a dictionary
    #             for get_bytes() - returns a python bytestring
    #             for get_msg() - returns a python string
    #     """
    #     # If getting things from Earthworm...
    #     if 'get' in self.pulse_method:
    #         # ...if getting a TYPE_TRACEBUFF2 (code 19) message, use class method directly
    #         if 'wave' in self.pulse_method:
    #             msg = self.module.get_wave(self.conn_id)
    #             # Flag null result as False
    #             if msg == {}:
    #                 msg = False
    #         # ...if getting a string or bytestring, use eval approach for compact code
    #         else:
    #             eval_str = f'self.module.{self.pulse_method}(self.conn_id, self.msg_type)'
    #             msg = eval(eval_str)
    #             # Flag empty message results as 'False'
    #             if msg == '':
    #                 msg = False
            
    #         return msg
        
    #     # If sending things to Earthworm...
    #     elif 'put' in self.pulse_method:
    #         msg = x
    #         # ..if sending waves
    #         if 'wave' in self.pulse_method and is_wave_msg(msg):
    #             # Commit to ring
    #             self.module.put_wave(self.conn_id, msg)
    #         # If sending byte or string messages
    #         else:
    #             # Compose evalstr
    #             eval_str = f'self.module.{self.pulse_method}(self.conn_id, self.msg_type, x)'
    #             # Execute eval
    #             eval(eval_str)
    #         return None      

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
        rstr = f'ayahos.core.wyrms.ringwyrm.RingWyrm(module={self.module}, '
        rstr += f'conn_id={self.conn_id}, pulse_method={self.pulse_method}, '
        rstr += f'msg_type={self.msg_type}, '
        rstr += f'max_pulse_size={self.max_pulse_size})'
        return rstr