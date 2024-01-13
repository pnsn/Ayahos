"""
:module: wyrm.wyrms.ringwyrm
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This module contains class definitions stemming from the Wyrm BaseClass
    that house data transfer methods between in-memory Earthworm RINGs and
    data saved on disk

Wyrm (ð)
|
=-->RingWyrm - Adds attributes to house an established PyEW.EWModule reference
                and a single ring connection index (conn_index)


:attribution:
    This module builds on the PyEarthworm (C) 2018 F. Hernandez interface
    between an Earthworm Message Transport system and Python distributed
    under an AGPL-3.0 license.

"""
import pandas as pd
from wyrm.wyrms.wyrm import Wyrm
from wyrm.message.base import _BaseMsg
from wyrm.message.trace import TraceMsg
from wyrm.structures.rtinststream import RtInstStream
# import PyEW


class RingWyrm(Wyrm):
    """
    Base class provides attributes to house a single PyEW.EWModule
    connection to an Earthworm RING.

    The EWModule initialization and ring connection should be conducted
    using a single HeartWyrm (wyrm.core.sequential.HeartWyrm) instance
    and then use links to these objects when initializing a RingWyrm

    e.g.,
    heartwyrm = HeartWyrm(<args>)
    heartwyrm.initalize_module()
    heartwyrm.add_connection(RING_ID=1000, RING_Name="WAVE_RING")
    ringwyrm = RingWyrm(heartwyrm.module, 0)
    """

    def __init__(
        self,
        module=None,
        conn_id=0,
        max_buff_sec=150,
        max_pulse_size=12000,
        debug=False
    ):
        """
        Initialize a RingWyrm 
        """
        Wyrm.__init__(self, debug=debug, max_pulse_size=max_pulse_size)
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

        # Compatability checks for RtInstStream.__init__()
        self.max_buff_sec = self._bounded_floatlike_check(max_buff_sec, name='max_buff_sec', minimum=1.)
        
        # Initialize Realtime Instrument Stream Object as buffer
        self.buffer = RtInstStream(max_length=self.max_buff_sec)
        

    def __str__(self, extended=False):
        rstr = super().__str__()
        rstr = f'Module: {self.module} | Conn ID: {self.conn_id}\n'
        rstr += 'FLOW:   EW --> PY\n\n'
        rstr += 'RingWyrm.buffer contents:\n'
        rstr += f'{self.buffer.__repr__(extended=extended)}'
        return rstr
    
    def __repr__(self, extended=False):
        rstr = self.__str__(extended=extended)
        return rstr

    def pulse(self, x=None):
        for _ in range(self.pulse_size):
            _wave = self.module.get_wave(self.conn_id)
            if _wave == {}:
                break
            else:
                _msg = TraceMsg(_wave)
                self.buffer.append(_msg)
        y = self.buffer
        return y



        

