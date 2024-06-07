"""
:module:`~ayahos.core.ayahosewmodule`
:auth: Nathan T. Stevens
:org: Pacific Northwest Seismic Network
:email: ntsteven (at) uw.edu
:license: AGPL-3.0
:attribution: This module directly extends from the :class:`~PyEW.EWModule` class
    written by Francisco J Hernandez Ramirez (c) 2019 as part of the PyEarthworm
    project. We remix this source code in compliance with their AGPL-3.0 licensing.
:purpose: 
    This module adds some book-keeping attributes and method wrappers for the Python-
    side operation of PyEarthworm, primarily adding a *EWModule.connections attribute
    that tracks unique named connections between Python and Earthworm (Py<->EW).
"""

from PyEW import EWModule
import logging
Logger = logging.getLogger(__name__)

from ayahos.util.pyew import is_wave_msg

class AyahosEWModule(EWModule):
    """
    An extension of the PyEarthworm PyEW.EWModule class that provides additional
    book-keeping utilities for transport ring connections and queries

    :param default_ring_id: ID of the first ring to connect to, defaults to 1000 (WAVE RING)
    :type default_ring_id: int, optional
    :param module_id: ID number for this module as seen by the Earthworm Environment. Must not be the same as any other
        modules listed in your Earthworm installation's earthworm.d parameter file, defaults to 193 (unused in PNSN Earthworm)
    :type: int, optional
    :param installation_id: installation ID number, defaults to 2 (INST_UW - Pacific Northwest Seismic Network)
    :type installation_id: int, optional
    :param heartbeat_period: how often to send heartbeat messages from this module (in seconds), defaults to 30.
    :type heartbeat_period: float, optional
    :param extended_debug: provide logging messages from within the PyEW.EWModule object? Defaults to False
    :type extended_debug: bool, optional

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
                 extended_debug=False):    
        """
        Initialize an AyahosEWModule object

        :param default_ring_id: ID of the first ring to connect to, defaults to 1000 (WAVE RING)
        :type default_ring_id: int, optional
        :param module_id: ID number for this module as seen by the Earthworm Environment. Must not be the same as any other
            modules listed in your Earthworm installation's earthworm.d parameter file, defaults to 193 (unused in PNSN Earthworm)
        :type: int, optional
        :param installation_id: installation ID number, defaults to 2 (INST_UW - Pacific Northwest Seismic Network)
        :type installation_id: int, optional
        :param heartbeat_period: how often to send heartbeat messages from this module (in seconds), defaults to 30.
        :type heartbeat_period: float, optional
        :param extended_debug: provide logging messages from within the PyEW.EWModule object? Defaults to False
        :type extended_debug: bool, optional

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
                         extended_debug)
        # Capture input values
        self.mod_id = module_id
        self.inst_id = installation_id
        self.hb_period = heartbeat_period
        self.def_ring_id = default_ring_id
        self.debug = extended_debug

        # Create holder for connections
        self.connections = {}

        # Make connections
        for _name, _id in connections.items():
            self.add_ring(_id, _name)

    def __repr__(self):
        rstr = 'Ayahos<->Earthworm Module\n'
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
        to this AyahosModule. It does not prevent creating duplicate
        connections between Py<->EW with different names.

        :param ring_id: _description_
        :type ring_id: _type_
        :param conn_name: _description_
        :type conn_name: _type_
        """        
        if conn_name in self.connections.keys():
            Logger.warning(f'connection already exists under {conn_name} to ring_id {self.connections[conn_name]}')
            Logger.warning(f'Connection renaming can be done using AyahosModule.update_conn_name')
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
            Logger.error('newname already in connections names. Cannot apply update')
        elif oldname not in self.connections.keys():
            Logger.error('oldname is not in connections names. Cannot apply update')
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
        information contained in the :attr:`~ayahos.core.AyahosModule.connections` attribute

        :param conn_info: either the name of a connection or the ring ID of that connection
        :type conn_info: _type_
        :returns: 
            - **conn_idx** (*int*) -- connection index number
        :rtype: _type_
        """ 
        if conn_name not in self.connections.keys():
            if self.debug:
                Logger.critical('conn_info not found in connections')
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
                Logger.critical('input wave is not formatted correctly')
        elif conn_name is None:
            if self.debug:
                Logger.critical('input conn_name does not correspond to an active connection')
        else:
            super().put_wave(conn_idx, wave)
        return

    def put_msg(self, conn_name, msg, msg_type):
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
                Logger.critical('input msg is not type str')
        elif conn_name is None:
            if self.debug:
                Logger.critical('conn_name does not correspond to an active connection')
        elif not isinstance(msg_type, int):
            if self.debug:
                Logger.critical('msg_type must be type int')
        elif msg_type <= 0 or msg_type > 255:
            if self.debug:
                Logger.critical('msg_type must be \in [1, 255]')
        else:
            super().put_msg(conn_idx, msg, msg_type)
    
    def put_bytes(self, conn_name, msg, msg_type):
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
                Logger.critical('input msg is not type bytes')
        elif conn_name is None:
            if self.debug:
                Logger.critical('conn_name does not correspond to an active connection')
        elif not isinstance(msg_type, int):
            if self.debug:
                Logger.critical('msg_type must be type int')
        elif msg_type <= 0 or msg_type > 255:
            if self.debug:
                Logger.critical('msg_type must be \in [1, 255]')
        else:
            super().put_bytes(conn_idx, msg, msg_type)   
