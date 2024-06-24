"""
:module: camper.module.transact
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This conatins the class definition for a module that facilitates data transfer
    between the Python and Earthworm Message Transport System using established connections
    

Classes
-------
:class:`~ewflow.module.transact.TransactMod`
:class:`~ewflow.module.transact.EWModule`
"""
import logging
from collections import deque
from PyEW import EWModule
from PULSE.module._base import _BaseMod
from PULSE.util.pyew import wave2mltrace, is_empty_message, is_wave_msg

Logger = logging.getLogger(__name__)

class TransactMod(_BaseMod):   
    """
    Class that facilitates transactions between memory rings in the Earthworm
    Message Transport System and the Python environment. This wraps an active 
    EWModule and a single python-ring connection and provides an abstract
    TransactMod.pulse() method that facilitates the following PyEW.EWModule
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
            max_pulse_size=10000,
            meta_memory=3600,
            report_period=None,
            max_output_size=1e9
            ):
        """Initialize a TransactMod object

        :param module: Pre-initialized EWModule object with connections established
        :type module: camper.util.pyew.EWModule
        :param conn_name: connection name of a valid connection in module.connection
        :type conn_name: str
        :param pulse_method: name of the EWModule messaging method to use, defaults to 'get_wave'
        :type pulse_method: str, optional
            Supported: 'get_wave','put_wave','get_msg','put_msg','get_bytes','put_bytes'
            also see    :class:`~camper.util.pyew.EWModule`
        :param msg_type: Earthworm message code, defaults to 19 (TYPE_TRACEBUFF2)
        :type msg_type: int, optional
            cross reference with your installation's `earthworm_global.d` file
        :param max_pulse_size: Maximum mumber of messages to transact in a single pulse, defaults to 10000
        :type max_pulse_size: int, optional
        """        
        super().__init__(
            max_pulse_size=max_pulse_size,
            meta_memory=meta_memory,
            report_period=report_period,
            max_output_size=max_output_size)
        # Compatability checks for `module`
        if isinstance(module, EWModule):
            self.module = module
        else:
            raise TypeError(f'module must be type PULSE.util.pyew.EWModule, not type {type(module)}')
        
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

        self.Logger.info('INIT TransactMod: "{0}" for type {1}'.format(self.pulse_method, self.msg_type))



    #################################
    # POLYMORPHIC METHODS FOR PULSE #
    #################################

    def _should_this_iteration_run(self, input, input_size, iter_number):
        """
        POLYMORPHIC
        Last updated with :class:`~camper.module.transact.TransactMod`

        For "put" pulse_method, use _BaseMod's _continue_iteration() inherited method
        to see if there are unassessed objects in input

        For "get" pulse_method, assume iteration 0 should proceed,
        for all subsequent iterations, check if the last message appended
        to TransactMod().output was an empty message. If so

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
            # Use _BaseMod._should_this_iteration_run() method
            status = super()._should_this_iteration_run(input, input_size, iter_number)
        # If passing messages from ring to deque, default to True
        elif 'get' in self.pulse_method:
            status = True
            # NOTE: This relies on the 'break' clause in _capture_stdout
        return status

    def _unit_input_from_input(self, input):
        """
        POLYMORPHIC
        Last updated with :class:`~camper.module.transact.TransactMod`

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
            unit_input = super()._unit_input_from_input(input)
        return unit_input
    
    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last updated with :class:`~camper.module.transact.TransactMod`

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
                getattr(self.module, self.pulse_method)(self.conn_name, self.msg_type, unit_input)
            unit_output = None
        return unit_output
    
    def _capture_unit_out(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class:`~camper.module.transact.TransactMod`

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
    
    def _should_next_iteration_run(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class:`~camper.module.transact.TransactMod`

        Do not start next iteration if unit_output looks like an
        empty message for "get" type pulse_method

        :param unit_output: unit output from :meth: `~camper.module.transact.TransactMod._unit_process`
        :type unit_output: dict, tuple, or list, depends on pulse_method
        :return status: Should the next iteration be run, based on unit_output?
        :rtype status: bool
        """
        status = True
        if 'get' in self.pulse_method:
            if is_empty_message(unit_output):
                status = False
            else:
                status = True
        elif 'put' in self.pulse_method:
            status = True
        else:
            self.Logger.error('We shouldnt have gotten here - means the pulse_method was altered')
        #     else:
        #         self.Logger.error("We shouldn't have gotten here (empty message with a 'put' method)")
        #         status = False
        # else:
        #     if 'put' in self.pulse_method:
        #         status = True
        #     else:
        #         self.Logger.error("We shouldn't have gotten here (empty message with a 'put' method)")
        #         status = False
        return status
    
    
####################
#### EWMod #####
####################
class PyEWMod(EWModule):
    """
    An extension of the PyEarthworm's :class:`~PyEW.EWModule` class that provides book-keeping utilities
    for transport ring connections and queries

    :param default_ring_id: ID of the first ring to connect to, defaults to 1000 (WAVE RING)
    :type default_ring_id: int, optional
    :param module_id: ID number for this module as seen by the Earthworm Environment. Must not be the same as any other
        modules listed in your Earthworm installation's earthworm.d parameter file, defaults to 193 (unused in PNSN Earthworm)
    :type: int, optional
    :param installation_id: installation ID number, defaults to 2 (INST_UW - Pacific Northwest Seismic Network)
    :type installation_id: int, optional
    :param heartbeat_period: how often to send heartbeat messages from this module (in seconds), defaults to 30.
    :type heartbeat_period: float, optional
    :param deep_debug: provide logging messages from within the PyEW.EWModule object? Defaults to False
    :type deep_debug: bool, optional

    :additional attributes:
        - **self.connections** (*dict*) - dictionary keeping track of unique connections to earthworm transport rings
            formatted as {Connection_Name: Ring Number}, with the position of the connection 
    """
    def __init__(self,
                 connections={'WAVE_RING':1000,
                              'PICK_RING':1005},
                 module_id=193,
                 installation_id=2,
                 heartbeat_period=30, 
                 deep_debug=False):    
        """
        Initialize an EWFlowEWModule object

        :param default_ring_id: ID of the first ring to connect to, defaults to 1000 (WAVE RING)
        :type default_ring_id: int, optional
        :param module_id: ID number for this module as seen by the Earthworm Environment. Must not be the same as any other
            modules listed in your Earthworm installation's earthworm.d parameter file, defaults to 193 (unused in PNSN Earthworm)
        :type: int, optional
        :param installation_id: installation ID number, defaults to 2 (INST_UW - Pacific Northwest Seismic Network)
        :type installation_id: int, optional
        :param heartbeat_period: how often to send heartbeat messages from this module (in seconds), defaults to 30.
        :type heartbeat_period: float, optional
        :param deep_debug: provide logging messages from within the PyEW.EWModule object? Defaults to False
        :type deep_debug: bool, optional

        :additional attributes:
            - **self.connections** (*dict*) - dictionary keeping track of unique connections to earthworm transport rings
                formatted as {Connection_Name: Ring Number}, with the position of the connection 
        """
        # Compatability checks on connections
        if isinstance(connections, dict):
            if all(isinstance(_k, str) and isinstance(_v, int) and 0 < _v < 10000 for _k, _v, in connections.items()):
                default_ring_id = list(connections.values())[0]

        # Inherit from PyEW.EWModule
        super().__init__(default_ring_id,
                         module_id,
                         installation_id,
                         heartbeat_period,
                         deep_debug)
        self.Logger = logging.getLogger('PULSED.module.transact.PyEWMod')
        # Capture input values
        self.mod_id = module_id
        self.inst_id = installation_id
        self.hb_period = heartbeat_period
        self.def_ring_id = default_ring_id
        self.debug = deep_debug

        # Create holder for connections
        self.connections = {}

        # Make connections
        for _name, _id in connections.items():
            self.add_ring(_id, _name.upper())
            self.Logger.info(f'Attached to ring {_id}, index {len(self.connections)}, alias: {_name}')

    def __repr__(self):
        rstr = 'EWFlow<->Earthworm Module\n'
        rstr += f'MOD_ID: {self.mod_id} | INST_ID: {self.inst_id} | '
        rstr += f'HB_PERIOD: {self.hb_period} | DEFAULT_RING: {self.def_ring_id}\n'
        rstr += 'Connections\n      Name      |  ID  \n'
        for _name, _id in self.connections.items():
            rstr += f'{_name:>15} | {_id:<4} \n'
        return rstr


    def add_ring(self, ring_id, conn_name):
        """
        Wraps the :meth:`~PyEW.EWModule.add_ring` method with 
        added safeguards against adding identically named connections
        to this EWFlowule. It does not prevent creating duplicate
        connections between Py<->EW with different names.

        :param ring_id: _description_
        :type ring_id: _type_
        :param conn_name: _description_
        :type conn_name: _type_
        """        
        if conn_name in self.connections.keys():
            self.Logger.warning(f'connection already exists under {conn_name} to ring_id {self.connections[conn_name]}')
            self.Logger.warning(f'Connection renaming can be done using EWFlowule.update_conn_name')
        else:
            super().add_ring(ring_id)
            self.connections.update({conn_name: ring_id})

    def update_conn_name(self, oldname, newname):
        """update the name of an entry in self.connections with a new,
        unique name

        :param oldname: name of connection to change name of
        :type oldname: str
        :param newname: new name to assign to that connection
        :type newname: str
        """
        if newname in self.connections.keys():
            self.Logger.error('newname already in connections names. Cannot apply update')
        elif oldname not in self.connections.keys():
            self.Logger.error('oldname is not in connections names. Cannot apply update')
        else:
            tmp = {}
            for _k, _v in self.connections.items():
                if _k == oldname:
                    _k = newname
                tmp.update({_k: _v})
            self.connections = tmp

    def get_conn_index(self, conn_name):
        """
        Get the connection index number used by the underlying
        :class:`~PyEW.EWModule` get* and put* methods with human readable/unique
        information contained in the :attr:`~ayahos.core.EWFlowule.connections` attribute

        :param conn_info: either the name of a connection or the ring ID of that connection
        :type conn_info: _type_
        :returns: 
            - **conn_idx** (*int*) -- connection index number
        :rtype: _type_
        """ 
        if conn_name not in self.connections.keys():
            if self.debug:
                self.Logger.critical('conn_info not found in connections')
            return None
        else:
            for conn_idx, (_k, _v) in enumerate(self.connections.items()):
                if conn_name == _k:
                    return conn_idx
        
    def get_wave(self, conn_name):
        """Wraps :meth:`~PyEW.EWModule.get_wave`, allowing use of connection
        names for selecting the target ring to get a wave from

        :param conn_name: connection name
        :type conn_name: str
        :return: wave message, or None - indicating invalid connection
        :rtype: dict or NoneType
        """        
        conn_idx = self.get_conn_index(conn_name)
        if isinstance(conn_idx, int):
            output = super().get_wave(conn_idx)
        else:
            output = None
        return output
    
    def get_msg(self, conn_name, msg_type):
        """Wraps :meth:`~PyEW.EWModule.get_msg`, allowing use of connection
        names for selecting the target ring to get a wave from

        :param conn_name: connection name
        :type conn_name: str
        :return: string-formatted message, or None - indicating invalid connection
        :rtype: dict or NoneType
        """       
        conn_idx = self.get_conn_index(conn_name)
        if isinstance(conn_idx, int):
            output = super().get_msg(conn_idx, msg_type)
        else:
            output = None
        return output
        

    def get_bytes(self, conn_name, msg_type):
        """Wraps :meth:`~PyEW.EWModule.get_bytes`, allowing use of connection
        names for selecting the target ring to get a wave from

        :param conn_name: connection name
        :type conn_name: str
        :return: bytestring message, or None - indicating invalid connection
        :rtype: bytes or NoneType
        """   
        conn_idx = self.get_conn_index(conn_name)
        if isinstance(conn_idx, int):
            output = super().get_bytes(conn_idx, msg_type)
        else:
            output = None
        return output
    
    def put_wave(self, conn_name, wave):
        """Wraps the :meth:`~PyEW.EWModule.put_wave` method to use conn_name
        instead of the connection index number

        :param conn_name: connection name
        :type conn_name: str
        :param wave: wave object to submit
        :type wave: dict
        """       
        conn_idx = self.get_conn_index(conn_name)
        if not is_wave_msg(wave):
            if self.debug:
                self.Logger.critical('input wave is not formatted correctly')
        elif conn_name is None:
            if self.debug:
                self.Logger.critical('input conn_name does not correspond to an active connection')
        else:
            super().put_wave(conn_idx, wave)
        return

    def put_msg(self, conn_name, msg_type, msg):
        """Wraps the :meth:`~PyEW.EWModule.put_msg` method to use conn_name
        instead of the connection index number

        :param conn_name: connection name
        :type conn_name: str
        :param msg: message to submit
        :type msg: str
        :param msg_type: message type (TYPE_* ID number)
        :type msg_type: int
        """        
        conn_idx = self.get_conn_index(conn_name)
        if not isinstance(msg, str):
            if self.debug:
                self.Logger.critical('input msg is not type str')
        elif conn_name is None:
            if self.debug:
                self.Logger.critical('conn_name does not correspond to an active connection')
        elif not isinstance(msg_type, int):
            if self.debug:
                self.Logger.critical('msg_type must be type int')
        elif msg_type <= 0 or msg_type > 255:
            if self.debug:
                self.Logger.critical('msg_type must be \in [1, 255]')
        else:
            super().put_msg(conn_idx, msg_type, msg)
    
    def put_bytes(self, conn_name, msg_type, msg):
        """Wraps the :meth:`~PyEW.EWModule.put_bytes` method to use conn_name
        instead of the connection index number

        :param conn_name: connection name
        :type conn_name: str
        :param msg: message to submit
        :type msg: bytes
        :param msg_type: message type
        :type msg_type: int
        """        
        conn_idx = self.get_conn_index(conn_name)
        if not isinstance(msg, bytes):
            if self.debug:
                self.Logger.critical('input msg is not type bytes')
        elif conn_name is None:
            if self.debug:
                self.Logger.critical('conn_name does not correspond to an active connection')
        elif not isinstance(msg_type, int):
            if self.debug:
                self.Logger.critical('msg_type must be type int')
        elif msg_type <= 0 or msg_type > 255:
            if self.debug:
                self.Logger.critical('msg_type must be \in [1, 255]')
        else:
            super().put_bytes(conn_idx, msg_type, msg)  