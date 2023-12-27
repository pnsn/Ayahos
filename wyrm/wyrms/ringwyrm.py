"""
:module: wyrm.core.io
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
import os
import pandas as pd
from collections import deque
from wyrm.wyrms.wyrm import Wyrm
from wyrm.core.message import _BaseMsg, TraceMsg, HDEQ_Dict
from obspy import Stream, read
import fnmatch
import PyEW


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

    NOTE: This base class retains the vestigial pulse(x) class method of
    Wyrm, so it's functionality on its own is limited. Child classes provide
    more-specific functionalities.

    :: ATTRIBUTES ::
    :attrib module: [PyEW.EWModule]
    :attrib conn_index: [int]
    :attrib queue_dict: [dict of {sncl: {q:deque, a:age}}] buffer that either stores received
                        messages from Earthworm or outgoing messages to Earthworm
                        in a pulse. Each sncl-keyed entry consists of a dictionary
                        composed of:
                            q: deque - [collections.deque]
                                    double-ended queue that stores type _BaseMsg + children
                                    objects
                            a: age - [int] number of pulses since the last time a new element was
                                    added to the associated deque
    :attrib mtype: [str] Earthworm message TYPE_* name
    :attrib mcode: [int] Earthworm message type integer code
    :attrib _to_ring: [bool] is data flow from python to earthworm?
    :attrib _max_pulse_size: [int] maximum number of messages to pull/put per pulse(x)
                Default: 12000 = 800 stations x 3 channels x 5 pulses x 1 sec-long messages x 1 sec pulse_rate
    :attrib _max_queue_size: [int] maximum number of messages permitted per queue.
                Default: 150 = 60 sec windows x 2 non-overlapping windows x 1.25 FOS
    :attrib _max_queue_age: [int] maximum number of pulses a given queue can go without
                                receiving new data.
                Default: 60 = 60 sec window x 1 sec pulse_rate

    NOTES:
    queueing rule: FIFO - first in first out
                New entries are added to deques with queue.appendleft(x) and entries are
                removed from deques with x = queue.pop()
                    pop() is used by:
                        + subsequent Wyrms to claim SNCL matched messages
                        + this Wyrm when clearing/cleaning buffers
    clean/flush rules:
        For a given 'SCNL' at the end of a pulse:
            CLEAN
            if len(self.queue_dict['SNCL']['q']) > _max_queue_size
                -> pop entries until len(*) = _max_queue_size
            FLUSH
            if self.queue_dict['SNCL']['a'] > _max_queue_age
                -> clear all entries from self.queue_dict['SCNL']['q']
                    i.e., self.queue_dict['SCNL']['q'] = deque([])


    """

    def __init__(
        self,
        module,
        conn_info,
        mtype="TYPE_TRACEBUFF2",
        mcode=19,
        flow_direction="to python",
        max_pulse_size=12000,
        max_queue_length=150,
        max_age=60,
    ):
        # Initialize parent classes
        Wyrm.__init__(self)
        # DEQ_Dict.__init__(self)
        # Attach empty Heirarchical DEQue with AGE extra field
        self.hdeq = HDEQ_Dict(extra_contents={"age": 0})
        # Run compatability checks on module
        if isinstance(module, PyEW.EWModule):
            self.module = module
        else:
            raise TypeError("module must be a PyEW.EWModule object!")

        # Compatability check for conn_info
        # Handle Series Input
        if isinstance(conn_info, pd.Series):
            if all(x in ["RING_ID", "RING_Name"] for x in conn_info.index):
                self.conn_info = pd.DataFrame(conn_info).T
            else:
                raise KeyError("new_conn_info does not contain the required keys")

        # Handle DataFrame input
        elif isinstance(conn_info, pd.DataFrame):
            if len(conn_info) == 1:
                if all(x in ["RING_ID", "RING_Name"] for x in conn_info.columns):
                    self.conn_info = conn_info
                else:
                    raise KeyError("new_conn_info does not contain the required keys")
            else:
                raise ValueError("conn_info must resemble a 1-row DataFrame")
        # Kick TypeError for any other input
        else:
            raise TypeError(
                "conn_info must be a Series or 1-row DataFrame (see RingWyrm doc)"
            )
        # Final bit of formatting for __repr__ purposes
        if self.conn_info.index.name is not "index":
            self.conn_info.index.name = "index"

        # Compatability check for flow_direction
        if flow_direction.lower() in [
            "from ring",
            "to python",
            "to py",
            "from earthworm",
            "from ew",
            "from c",
            "c2p",
            "ring2py",
            "ring2python",
        ]:
            self._from_ring = True
        elif flow_direction.lower() in [
            "from python",
            "to ring",
            "from py",
            "to earthworm",
            "to c",
            "to ew",
            "p2c",
            "py2ring",
            "python2ring",
        ]:
            self._from_ring = False
        else:
            raise ValueError(
                "flow direction type {flow_direction} is not supported. See RingWyrm header doc"
            )

        # Comptatability check for message type using wyrm.core.message._BaseMsg
        try:
            test_msg = _BaseMsg(mtype=mtype, mcode=mcode)
            self.mtype = test_msg.mtype
            self.mcode = test_msg.mcode
        except TypeError:
            raise TypeError(f"from mtype:mcode | {mtype}:{mcode}")
        except SyntaxError:
            raise SyntaxError(f"from mtype:mcode | {mtype}:{mcode}")
        except ValueError:
            raise ValueError(f"from mtype:mcode | {mtype}:{mcode}")

        # Initialize buffer dictionary & limit indices
        self._max_pulse_size = max_pulse_size
        self._max_queue_size = max_queue_length
        self._max_queue_age = max_age

    def __repr__(self, extended=False):
        rstr = f"Module: {self.module}\n"
        rstr += f"Conn Info: {self.conn_info}\n"
        rstr += f"{self.hdeq.__repr__(extended=extended)}"
        return rstr

    def change_conn_info(self, new_conn_info):
        if isinstance(new_conn_info, pd.DataFrame):
            if len(new_conn_info) == 1:
                if all(x in ["RING_Name", "RING_ID"] for x in new_conn_info.columns):
                    self.conn_info = new_conn_info
                else:
                    raise KeyError("new_conn_info does not contain the required keys")
            else:
                raise ValueError(
                    "new_conn_info must be a 1-row pandas.DataFrame or pandas.Series"
                )
        elif isinstance(new_conn_info, pd.Series):
            if all(x in ["RING_Name", "RING_ID"] for x in new_conn_info.index):
                self.conn_info = pd.DataFrame(new_conn_info).T
            else:
                raise KeyError("new_conn_info does not contain the required keys")

    def _flush_clean_queue(self):
        """
        Assess the age and size of each queue in self.queue_dict
        and conduct clean/flush operations based on rules described

        CLEAN - if queue length exceeds max queue length
                pop off oldest (right-most) elements in queue
                until queue length equals max queue length

        FLUSH - if queue age exceeds max queue age, clear out
                all contents of the queue and reset age = 0
        """
        for _k in self.queue_dict.keys():
            _qad = self.queue_dict[_k]
            # Run FLUSH check
            # If too old, run FLUSH
            if _qad["age"] > self._max_queue_age:
                # Reset deque to an empty deque
                _qad["q"] = deque([])
                # Reset age to 0
                _qad["age"] = 0
            # If too young to FLUSH
            else:
                # Check if queue is too long
                if len(_qad["q"]) > self._max_queue_size:
                    # Run while loop that terminates when the queue is max_queue size
                    while len(_qad["q"]) > self._max_queue_size:
                        junk = _qad["q"].pop()
                    # NOTE: Age modification in CLEAN is ambiguous
                # If queue is shorter than max, do nothing
                else:
                    pass
        # END OF _flush_clean_queue

    def _flush_clean_hdeq(self):
        """
        Assess the age and size of each queue in self.hdeq
        and conduct clean/flush operations based on rules described

        CLEAN - if queue length exceeds max queue length
                pop off oldest (right-most) elements in queue
                until queue length equals max queue length

        FLUSH - if queue age exceeds max queue age, clear out
                all contents of the queue and reset age = 0
        """
        # Iterate across each SNCL code
        for _sncl in self.hdeq.codes:
            # Extract target
            _target = self.hdeq._get_sncl_target(_sncl)
            # If HDEQ_Dict entry 'age' is too old, reset
            if _target["age"] > self._max_queue_age:
                _target["q"] = deque([])
                _target["age"] = 0
            # Else
            else:
                # If HDEQ_Dict 'q' is too long, prune from right
                if len(_target["q"]) > self._max_queue_size:
                    while len(_target["q"]) > self._max_queue_size:
                        _ = self.hdeq.pop(_sncl, key="q")
                else:
                    pass
        # END OF _flush_clean_hdeq

    def _to_ring_pulse(self, x=None):
        # If working with wave-like messaging, use class-methods
        # written into TraceMsg
        if self.mtype == "TYPE_TRACEBUF2" and self.mcode == 19:
            # Iterate across all _sncl
            for _sncl in self.hdeq.codes:
                # Isolate SCNL-keyed deque
                _target = self.hdeq[_sncl]
                _qlen = len(_target["q"])
                if self._max_pulse_size >= _qlen > 0:
                    _i = _qlen
                elif _qlen > self._max_pulse_size:
                    _i = self._max_pulse_size
                else:
                    _i = 0
                if _i > 0:
                    # iterate across _q items
                    for _ in range(_i):
                        # Pop off _msg
                        _msg = _target["q"].pop()
                        # If instance of TraceMsg
                        if isinstance(_msg, TraceMsg):
                            # Send to EW
                            _msg.to_ew(self._conn_index)
                        # Otherwise, append value back to the head of the queue
                        else:
                            _target["q"].appendleft(_msg)
                    # If all messages are recycled, increase age
                    if _qlen == len(_target["q"]):
                        _target["age"] += 1
            else:
                NotImplementedError(
                    "Other mtype:mcode combination handling not yet developed"
                )
        # END of _to_ring_pulse(x)

    def _from_ring_pulse(self, x=None):
        # If working with wave-like messaging, use class-methods
        # written into TraceMsg
        if self.mtype == "TYPE_TRACEBUF2" and self.mcode == 19:
            # Itrate for _max_pulse_size (but allow break)
            for _ in range(self._max_pulse_size):
                _wave = self.module.get_wave()
                # If an empty message is returned, end iteration
                if _wave == {}:
                    break
                # Otherwise
                else:
                    # Convert into TraceMsg
                    _msg = TraceMsg(_wave)
                    # Build branch (if it doesn't already exist)
                    self.hdeq._add_index_branch(_msg.scnl)
                    # Append message to 'q'
                    self.hdeq.appendleft(_msg.scnl, _msg, key="q")
        # If mtype:mcode arent for TYPE_TRACEBUF2, kick "IN DEVELOPMENT" error
        else:
            NotImplementedError(
                "Other mtype:mcode combination handling not yet developed"
            )
        # END of _from_ring_pulse()

    def pulse(self, x=None):
        """
        Pulse produces access to self.queue_dict via
        y = self.queue_dict

        :: INPUT ::
        :param x: [None] - placeholder to match fundamental definition
                    of the pulse(x) class method

        :: OUTPUT ::
        :return y: variable accessing this RingWyrm's self.queue_dict attribute
        """
        # If flowing to a ring (PY->EW)
        if self._to_ring:
            # Submit data to ring
            self._to_ring_pulse(x=x)
            # Then do queue cleaning
            self._flush_clean_hdeq()

        # If flowing from a ring (EW->PY)
        elif not self._to_ring:
            # Assess flush/clean for queues first
            self._flush_clean_hdeq()
            # Then bring in new data/increase age of un-updated queues
            self._from_ring_pulse(x=x)
        else:
            raise RuntimeError("Dataflow direction from self._to_ring not valid")
        # Make queue_dict accessible to subsequent (chained) Wyrms
        y = self.queue_dict
        return y
