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
import pandas as pd
from wyrm.wyrms.wyrm import Wyrm
from wyrm.message.base import _BaseMsg
from wyrm.message.trace import TraceMsg
from wyrm.structures.castings.deq_dict import DEQ_Dict
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
        conn_id,
        max_pulse_size=12000,
        max_queue_size=150,
        max_queue_age=60,
        flow_direction="to python",
        mtype="TYPE_TRACEBUFF2",
        mcode=19
    ):
        # Initialize parent classes
        Wyrm.__init__(self)
        # Create an empty DEQ_dict
        self.queues = DEQ_Dict(extra_fields={"age": 0})

        # Run compatability checks on module
        if isinstance(module, PyEW.EWModule):
            self.module = module
        else:
            raise TypeError("module must be a PyEW.EWModule object!")

        # Compatability check for conn_id
        if not isinstance(conn_id, int):
            raise TypeError('conn_id must be int')
        elif conn_id < 0:
            raise ValueError('conn_id must be 0+')
        else:
            self._conn_id = conn_id
        
        names = ['max_pulse_size','max_queue_size','max_queue_age']
        for _i, _x in enumerate([max_pulse_size, max_queue_size, max_queue_age]):
            if not isinstance(_x, int):
                raise TypeError(f'{names[_i]} must be type int')
            elif _x < 1: 
                raise ValueError(f'{names[_i]} must be positive')
            else:
                eval(f'self.{names[_i].split('max')[-1]} = _x')

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

        



    def __repr__(self, extended=False):
        rstr = f"Module: {self.module}\n"
        rstr += f'CONN: {self._conn_id} "{self._ring_name}"\n'
        if self._from_ring:
            rstr += 'FLOW: EW->PY\n'
        else:
            rstr += 'FLOW: PY->EW\n'
        rstr += f"{self.queues.__repr__(extended=extended)}"
        return rstr

    def _flush_clean_queues(self):
        """
        Assess the age and size of each queue in self.queue_dict
        and conduct clean/flush operations based on rules described

        CLEAN - if queue length exceeds max queue length
                pop off oldest (right-most) elements in queue
                until queue length equals max queue length

        FLUSH - if queue age exceeds max queue age, clear out
                all contents of the queue and reset age = 0
        """
        # Iterate across all SNCL keys
        for _sncl in self.queues.keys():
            # Get SNCL keyed queue
            _queue = self.queue_dict[_sncl]
            # Run FLUSH check
            # If too old, run FLUSH
            if _queue["age"] > self.pulse_limits['queue_age']:
                # Reset entry to default template
                self.queues.add_blank_entry(_sncl, overwrite=True)
            # If too young to FLUSH
            else:
                # Check if queue is too long
                if len(_queue["q"]) > self.pulse_limits['queue_size']:
                    # Run while loop that terminates when the queue is max_queue size
                    while len(_queue["q"]) > self.pulse_limits['queue_size']:
                        _ = self.queues.pop(_sncl)
                # If queue is shorter than max, do nothing
                else:
                    pass
        # END OF _flush_clean_queue


    def _to_ring_pulse(self, x=None):
        # If working with wave-like messaging, use class-methods
        # written into TraceMsg
        if self.mtype == "TYPE_TRACEBUF2" and self.mcode == 19:
            _iwhile = 0
            killswitch = False
            sncl_list = list(self.queues.keys())
            # Iterate until max pulse size is reached or killswitch is activated
            while _iwhile  < self._pulse_size or not killswitch:
                # Create a bool list for this iteration
                updated_list = []
                # Iterate across SNCL codes
                for _j, _sncl in sncl_list:
                    # Pop message off queue
                    msg = self.queues.pop(_sncl, queue='q')
                        # If queue is empty, remove it from consideration
                    if isinstance(msg, type(None)):
                        sncl_list.remove(_sncl)

                    # If TraceMsg
                    elif isinstance(msg, TraceMsg):
                        # submit to earthworm
                        msg.to_ew(self._conn_info['CONN_ID'])
                        # Increase iteration counter
                        _iwhile += 1
                        # Append True to bool_set to keep killswitch off this time
                        updated_list.append(True)
                        # Reset queue age to 0
                        self.queues[_sncl]['age'] = 0


                    # If anything else
                    else:
                        # Reappend message to left end of queue
                        self.queues.appendleft(msg, _sncl, queue='q')
                        # Append False to updated_list to signal this transaction does not negate killswitch
                        updated_list.append(False)
                        # Increase iteration counter
                        _iwhile += 1 
                # If any queues submitted, continue
                if any(updated_list):
                    killswitch = False
                # If all queues failed to submit, activate killswitch
                else:
                    killswitch = True
            # As clean-up, increment queue age
            for _sncl in self.queues.keys():
                if len(self.queues[_sncl]['q']) == 0:
                    self.queues[_sncl]['age'] = 0
                else:
                    self.queues[_sncl]['age'] += 1

        else:
            NotImplementedError("Other mtype:mcode combination handling not yet developed")
        # END of _to_ring_pulse(x)

    def _from_ring_pulse(self, x=None):
        # If working with wave-like messaging, use class-methods
        # written into TraceMsg
        if self.mtype == "TYPE_TRACEBUF2" and self.mcode == 19:


            # Itrate for _max_pulse_size (but allow break)
            for _ in range(self._pulse_size):
                _wave = self.module.get_wave(self._conn_id)
                # If an empty message is returned, end iteration
                if _wave == {}:
                    break
                # Otherwise
                else:
                    # Convert into TraceMsg
                    _msg = TraceMsg(_wave)
                    # Append message to queues
                    self.queues.appendleft(_msg, _msg.sncl, queue='q')
                    # Temporarily set age to -1 (will increment up to 0 in cleanup)
                    self.queues[_msg.scnl]['age'] = -1

            # CLEANUP, increase all ages by 1
            for _sncl in self.queues.keys():
                self.queues[_sncl]['age'] += 1

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
            self._flush_clean_queues()

        # If flowing from a ring (EW->PY)
        elif not self._to_ring:
            # Assess flush/clean for queues first
            self._flush_clean_queues()
            # Then bring in new data/increase age of un-updated queues
            self._from_ring_pulse(x=x)
        else:
            raise RuntimeError("Dataflow direction from self._to_ring not valid")
        # Make queue_dict accessible to subsequent (chained) Wyrms
        y = self.queue_dict
        return y
