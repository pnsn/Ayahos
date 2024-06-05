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
from ayahos.core.ayahosewmodule import AyahosEWModule
from collections import deque
from ayahos.wyrms.wyrm import Wyrm
from ayahos.util.pyew import wave2mltrace, is_empty_message

Logger = logging.getLogger(__name__)

# @add_class_name_to_docstring
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
            conn_name,
            pulse_method='get_wave',
            msg_type=19,
            max_pulse_size=10000
            ):
        """Initialize a RingWyrm object

        :param module: Pre-initialized AyahosEWModule object with connections established
        :type module: ayahos.core.ayahosewmodule.AyahosEWModule
        :param conn_name: connection name of a valid connection in module.connection
        :type conn_name: str
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
        # Compatability checks for `module`
        if isinstance(module, AyahosEWModule):
            self.module = module
        else:
            raise TypeError(f'module must be type PyEW.EWModule, not type {type(module)}')
        
        if conn_name not in module.connections.keys():
            raise KeyError(f'conn_name {conn_name} is not a named connection in the input module')
        else:
            self.conn_name = conn_name
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

        Logger.info('RingWyrm method {0} for message type {1}'.format(self.pulse_method, self.msg_type))



    #################################
    # POLYMORPHIC METHODS FOR PULSE #
    #################################

    def _should_this_iteration_run(self, input, input_size, iter_number):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.ringwyrm.RingWyrm`

        For "put" pulse_method, use Wyrm's _continue_iteration() inherited method
        to see if there are unassessed objects in input

        For "get" pulse_method, assume iteration 0 should proceed,
        for all subsequent iterations, check if the last message appended
        to RingWyrm().output was an empty message. If so

        Inputs and outputs for "put" pulse_method:
        :param input: standard input collection of objects (put)
        :type input: collections.deque (put) None (get)
        :param iter_number: iteration number
        :type iter_number: int
        :return status: should iteration in pulse continue?
        :rtype status: bool
        """        
        # If passing messages from deque to ring, check if there are messages to pass
        if 'put' in self.pulse_method:
            # Use Wyrm._should_this_iteration_run() method
            status = super()._should_this_iteration_run(input, input_size, iter_number)
        # If passing messages from ring to deque, default to True
        elif 'get' in self.pulse_method:
            status = True
            # NOTE: This relies on the 'break' clause in _capture_stdout
        return status

    def _unit_input_from_input(self, input):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.ringwyrm.RingWyrm`

        If using a "get" pulse_method, input is unused and returns None
        
        If using a "put" pulse_method, inputs and outputs are:

        :param input: standard input object collection
        :type input: collections.deque
        :return unit_input: object retrieved from input
        :rtype: PyEW message-like object
        """
        # Input object for "get" methods is None by default        
        if 'get' in self.pulse_method:
            unit_input = None
        # Input object for "put" methods is a PyEW message-formatted object
        elif 'put' in self.pulse_method:
            unit_input = super()._get_obj_from_input(input)
        return unit_input
    
    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.ringwyrm.RingWyrm`

        "get" pulse_methods fetch a message from the specified EW RING
        "put" pulse_methods put unit_input onto the specified EW RING

        :param unit_input: message to "put" onto an Earthworm memory ring, or None for "get" methods
        :type unit_input: PyEW message-formatted object (get) or None (put)
        :return unit_output: message object output from "get" pulse_methods
        :rtype unit_output: PyEW message-formatted object (get) or None (put)
        """        
        if 'get' in self.pulse_method:
            if self.pulse_method == 'get_wave':
                unit_output = getattr(self.module, self.pulse_method)(self.conn_name)
            else:
                unit_output = getattr(self.module, self.pulse_method)(self.conn_name, self.msg_type)
        elif 'put' in self.pulse_method:
            if self.pulse_method == 'put_wave':
                getattr(self.module, self.pulse_method)(self.conn_name, unit_input)
            else:
                getattr(self.module, self.pulse_method)(self.conn_name, unit_input, self.msg_type)
            unit_output = None
        return unit_output
    
    def _capture_unit_out(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.ringwyrm.RingWyrm`

        "get" pulse_methods use Wyrm's _capture_unit_output()
        "put" pulse_methods do nothing (pass)

        :param unit_output: standard output object from _unit_process
        :type unit_output: PyEW message-formatted object or None
        :return: None
        :rtype: None
        """
        # For "get" methods, capture messages        
        if 'get' in self.pulse_method:
            # If unit_output is an empty message
            if not is_empty_message(unit_output):
                if self.pulse_method == 'get_wave':
                    unit_output = wave2mltrace(unit_output)
                # For all "get" methods, use Wyrm._capture_unit_output()
                super()._capture_unit_output(unit_output)
        return None
    
    def _should_next_iteration_run(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.ringwyrm.RingWyrm`

        Do not start next iteration if unit_output looks like an
        empty message for "get" type pulse_method

        :param unit_output: unit output from :meth: `~ayahos.wyrms.ringwyrm.RingWyrm._unit_process`
        :type unit_output: dict, tuple, or list, depends on pulse_method
        :return status: Should the next iteration be run, based on unit_output?
        :rtype status: bool
        """
        if is_empty_message(unit_output):
            if 'get' in self.pulse_method:
                status = False
            else:
                Logger.error("We shouldn't have gotten here (empty message with a 'put' method)")
                status = False
        else:
            if 'get' in self.pulse_method:
                status = True
            else:
                Logger.error("We shouldn't have gotten here (empty message with a 'put' method)")
                status = False
        return status

    # def __repr__(self):
    #     """
    #     Provide a string representation of this RingWyrm object

    #     :: OUTPUT ::
    #     :return rstr: [str] representative string
    #     """
    #     # Print from Wyrm
    #     rstr = super().__repr__()
    #     # Add lines for RingWyrm
    #     rstr += f'\nModule: {self.module} | Conn ID: {self.conn_id} | '
    #     rstr += f'Method: {self.pulse_method} | MsgType: {self.msg_type}'
    #     return rstr
    
    # def __str__(self):
    #     rstr = f'ayahos.core.wyrms.ringwyrm.RingWyrm(module={self.module}, '
    #     rstr += f'conn_id={self.conn_id}, pulse_method={self.pulse_method}, '
    #     rstr += f'msg_type={self.msg_type}, '
    #     rstr += f'max_pulse_size={self.max_pulse_size})'
    #     return rstr


    # def _unit_process(self, obj):
    #     """unit_process for ayahos.core.wyrms.ringwyrm.RingWyrm

    #     Conduct a single transaction based on the following RingWyrm attributes

    #     .pulse_method
    #     .msg_type
    #     .conn_id

    #     for "get" pulse_methods - appends new messages to self.output
    #         :param x: Unused
    #         :type x: None
    #         :param i_: Unused
    #         :type i_: int

    #     for "put" pulse_methods - discards transacted message objects
    #         :param x: deque of message-like objects
    #         :type x: collections.deque
    #         :param i_: iteration counter used for early stopping
    #         :type i_: int

    #     :return status: did this iteration raise an early stopping flag?
    #     :rtype status: bool
    #     """        
    #     if 'get' in self.pulse_method:
    #         # Get a message using the PyEW.EWModule.get_* method selected
    #         if 'wave' in self.pulse_method:
    #             _y = getattr(self.module, self.pulse_method)(self._core_args[0])
    #             status = self._get_continue_iteration(_y)
    #             if status is True:
    #                 _y = wave2mltrace(_y)
    #                 self.output.append(_y)

    #         else:
    #             _y = getattr(self.module, self.pulse_method)(*self._core_args)
    #             # Check if it is a blank message
    #             status = self._get_continue_iteration(_y)
    #             # Append if status returns True (_y was a non-empty message)
    #             if status:
    #                 self.output.append(_y)

    #     if 'put' in self.pulse_method:
    #         # Check if the input is a deque
    #         if not isinstance(x, deque):
    #             raise TypeError('input x must be type collections.deque for put-type methods')
    #         # Check for early stopping (if there are any unassessed items in the deque)
    #         status = self._put_continue_iteration(x, i_)
    #         if status is True:
    #             # Grab the leftmost (oldest) item from the deque
    #             _x = x.popleft()
    #             # Submit to rings
    #             if 'wave' in self.pulse_method:
    #                 _x = trace2wave(_x)
    #                 getattr(self.module, self.pulse_method)(self._core_args[0], _x)
    #             else:
    #                 getattr(self.module, self.pulse_method)(*self._core_args, _x)

    #     return status

    

    # def _put_continue_iteration(self, x, i_):
    #     """Use _early_stopping from ayahos.core.wyrms.wyrm.Wyrm for "put" type pulse methods

    #     Early stopping if iteration counter exceeds length of `x`
    #     Early stopping if length of `x` is 0

    #     :param x: collection of messages being submitted
    #     :type x: collections.deque
    #     :param i_: iteration counter
    #     :type i_: int
    #     :return: status
    #             True = continue iterations
    #             False = trigger early stopping
    #     :rtype: bool
    #     """        
    #     if len(x) == 0:
    #         status = False
    #     elif i_ + 1 > len(x):
    #         status = False
    #     else:
    #         status = True
    #     return status

    # PULSE IS INHERITED FROM WYRM AS-IS


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

