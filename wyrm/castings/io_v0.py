"""
:module: wyrm.core.io
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module houses class definitions for handling data Input/Output
    from the Earthworm Message Transport System (memory rings) and
    from disk (in development)

    Classes

    RingWyrm - submodule for handling individual transactions between the Python
                and Earthworm environment, built on the PyEarthworm package
                (PyEW.EWModule). This provides access to the full range of get_
                and put_ class methods for a EWModule.

    EarWyrm - primary wave-fetching submodule for getting sets of tracebuff2 messages
            from a single WAVE RING and buffering them in a TieredBuffer(TraceBuffer)
            structure
                                 /Inst0-{Z: TraceBuffer, N: TraceBuffer, E: TraceBuffer}|         
            EW                  /                                                       |
            WAVE -> EarWyrm -> {--Inst1-{Z: TraceBuffer}                                |}
            RING                \                                                       |
                                 \InstN-{Z: TraceBuffer, 1: TraceBuffer, 2: Tracebuffer}|
                                ^^ TieredBuffer ^^
"""
import os
import re
#import PyEW
from obspy import Stream, read
from wyrm.core._base import Wyrm
from wyrm.util.pyew import is_wave_msg, wave2mltrace
from wyrm.util.compatability import bounded_floatlike, bounded_intlike
from wyrm.data.mltrace import MLTrace, MLTraceBuffer
from wyrm.data.dictstream import DictStream


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
    
    def __init__(self, module=None, conn_id=0, pulse_method_str='get_wave', msg_type=19, max_pulse_size=10000, debug=False):
        
        
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
        # # Update _in_types and _out_types private attributes for this Wyrm's pulse method
        # self._update_io_types(itype=(Trace, str, bytes), otype=(dict, bool, type(None)))

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
        # Run compatability checks on x
        self._matches_itype(x)                        
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

    def __str__(self):
        """
        Provide a string representation of this RingWyrm object

        :: OUTPUT ::
        :return rstr: [str] representative string
        """
        # Print from Wyrm
        rstr = super().__str__()
        # Add lines for RingWyrm
        rstr += f'\nModule: {self.module} | Conn ID: {self.conn_id} | '
        rstr += f'Method: {self.pulse_method} | MsgType: {self.msg_type}'
        return rstr
    
    def __repr__(self):
        rstr = f'wyrm.wyrms.io.RingWyrm(module={self.module}, '
        rstr += f'conn_id={self.conn_id}, pulse_method_str={self.pulse_method}, '
        rstr += f'msg_type={self.msg_type}, '
        rstr += f'max_pulse_size={self.max_pulse_size}, debug={self.debug})'
        return rstr


class EarWyrm(RingWyrm):
    """
    Wrapper child-class of RingWyrm specific to listening to an Earthworm
    WAVE Ring and populating MLTraceBuffers housed in a DictStream
    """

    def __init__(
        self,
        module=None,
        conn_id=0,
        max_length=150,
        restrict_past_append=True,
        max_pulse_size=12000,
        debug=False,
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
        :param debug: [bool] should this RingWyrm be run in debug mode?
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
            debug=debug
            )
        # Let TieredBuffer handle the compatability check for max_length
        # self.tree = BufferTree(
        #     buff_class=TraceBuffer,
        #     max_length=max_length,
        #     **options
        #     )
        self.max_length = bounded_floatlike(
            max_length,
            name='max_length',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        self.stream = DictStream()

        # self.options={}
        # for _k, _v in options.items():
        #     if _k in dir(MLTraceBuffer):
        #         self.options.update({_k: _v})
        # self._update_io_types(itype=(str, type(None)), otype=BufferTree)

    def pulse(self, x=None, **options):
        # Iterate up to max_pulse_size
        for _i in range(self.max_pulse_size):
            # Run pulse method from get_wave() formatted RingWyrm to get single wave
            _wave = super().pulse()
            # Early stopping if RingWyrm.pulse() returns False
            if not _wave:
                break
            # Otherwise
            else:
                # Convert wave to MLTrace
                mlt = wave2mltrace(_wave)
                # If MLTrace id is not in the dictstream keys
                if mlt.id not in self.stream.keys():
                    # Initialize a MLTraceBuffer
                    mltb = MLTraceBuffer(
                        max_length=self.max_length,
                        restrict_past_append=self.restrict_past_append)
                    # Append 
                    mltb.__add__(mlt)
                    self.stream.__add__(mlt, key_attr='id', **options)
                else:
                    self.stream.__add__(mlt, key_attr='id', **options)
                


    def pulse(self, x=None):
        """
        Execute a pulse wherein this RingWyrm pulls copies of 
        tracebuff2 messages from the connected Earthworm Wave Ring,
        converts them into obspy.core.trace.Trace objects, and
        appends traces to the RingWyrm.buffer attribute
        (i.e. a TieredBuffer object terminating with BuffTrace objects)

        A single pulse will pull up to self.max_pulse_size waveforms
        from the WaveRing using the PyEW.EWModule.get_wave() method,
        stopping if it hits a "no-new-messages" message.

        :: INPUT ::
        :param x: None or [str] 
                None (default) input accepts all waves read from ring, as does "*"

                All other string inputs:
                N.S.L.C formatted / unix wildcard compatable
                string for filtering read waveforms by their N.S.L.C ID
                (see obspy.core.trace.Trace.id). 
                E.g., 
                for only accelerometers, one might use:
                    x = '*.?N?'
                for only PNSN administrated broadband stations, one might use:
                    x = '[UC][WOC].*.*.?H?'

                uses the re.match(x, trace.id) method

        
        :: OUTPUT ::
        :return y: [wyrm.buffer.structure.TieredBuffer] 
                alias to this RingWyrm's buffer attribute, 
                an initalized TieredBuffer object terminating 
                in TraceBuff objects with the structure:
                y = {'NN.SSSSS.LL.CC': {'C': TraceBuff()}}

                e.g., 
                y = {'UW.GNW..BH: {'Z': TraceBuff(),
                                   'N': TraceBuff(),
                                   'E': TraceBuff()}}

        """
        # Start iterations that pull single wave object
        for _ in range(self.max_pulse_size):
            # Run the pulse method from RingWyrm for single wave pull
            _wave = super().pulse(x=None)
            # If RingWyrm.pulse() returns False - trigger early stopping
            if not _wave:
                break

            # If it has data (i.e., non-empty dict)
            else:
                # Convert PyEW wave to obspy trace
                trace = wave2trace(_wave)
                # If either accept-all case is provided with x, skip filtering
                if x == '*' or x is None:
                    key0 = trace.id[:-1]
                    key1 = trace.id[-1]
                    self.tree.append(trace, key0, key1)
                # otherwise use true-ness of re.search result to filter
                elif re.search(x, trace.id):
                    # Use N.S.L.BandInst for key0
                    key0 = trace.id[:-1]
                    # Use Component Character for key1
                    key1 = trace.id[-1]
                    self.tree.append(trace, key0, key1)
        # Present tree object as output for sequencing
        y = self.tree
        return y
        
    def __str__(self, extended=False):
        """
        Return representative, user-friendly string that details the
        contents of this EarWyrm
        
        :: INPUT ::
        :param extended: [bool] should the TieredDict
        
        """
        # Populate information from RingWyrm.__str__
        rstr = super().__str__()
        # Add __str__ from TieredBuffer
        rstr += f'\n{self.tree.__str(extended=extended)}'
        return rstr

    def __repr__(self):
        """
        Return descriptive string of how this EarWyrm was initialized
        """
        rstr = f'wyrm.wyrms.io.EarWyrm(module={self.module}, '
        rstr += f'conn_id={self.conn_id}, max_length={self.tree._template_buff.max_length}, '
        rstr += f'max_pulse_size={self.max_pulse_size}, debug={self.debug}'
        for _k, _v in self.options.items():
            rstr += f', {_k}={_v}'
        rstr += ')'
        return rstr


class StreamWyrm(Wyrm):
    """
    A class for simulating pulsed waveform loading behavior feeding off a pre-loaded
    ObsPy Stream object wherein all traces have unique ID's (i.e., stream is pre-merged)
    """

    def __init__(self, max_length=300, dt_segment=1, realtime=False, max_pulse_sie=300, debug=False):
        super().__init__(max_pulse_size=max_length, debug=debug)
        self.dt_segment = bounded_floatlike(
            dt_segment,
            name='dt_segment',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        if not isinstance(realtime, bool):
            raise TypeError
        elif realtime:
            raise NotImplementedError
        else:
            self.realtime = realtime
        self.i_stream = DictStream()
        self.o_stream = DictStream()
        self.index = {}


    def ingest_stream(self, stream):
        """
        Load data via a pre-merged obspy.core.stream.Stream object into this StreamWyrm's
        i_stream attribute and populate the index attribute with trace id's and trace
        starttimes
        """
        uid = []; trst = []
        for _tr in stream:
            if _tr.id not in uid:
                uid.append(_tr.id)
                trst.append(_tr.stats.starttime)
            else:
                raise ValueError('repeat trace ID found - merge stream before ingestion')
        self.index = dict(zip(uid, trst))
        self.i_stream = DictStream(traces=stream)

    

    def pulse(self, x=None):
        for _i in range(self.max_pulse_size):
            # Early stopping if index is exhausted
            if len(self.index) == 0:
                break
            
            # Iterate across traces in i_sream
            for id, _tr in enumerate(self.i_stream.traces.items()):
                t0 = self.index[_tr.id]
                t1 = t0 + self.dt_segment
                itr = _tr.copy().trim(startime=t0, endtime=t1)
                if t1 <= _tr.stats.endtime:
                    self.index[_tr.id] += self.dt_segment
                else:
                    self.index.pop(_tr.id)
                    self.i_stream.pop(_tr.id)
                # If new ID to output DictStream, generate MLTraceBuffer
                if id not in self.o_stream.traces.keys():
                    itr = MLTraceBuffer(max_length=self.max_length).__add__(itr)

                # Use __add__ for generalized update/append
                self.o_stream.__add__(itr)
                if self.realtime:
                    print('under construction - add stoppage time to simulate real-time operation')
        y = self.o_stream
        return y



class DiskWyrm(Wyrm):

    def __init__(self, event_files, max_length=300, reinit_period=1, max_pulse_size=1, debug=False):

        super().__init__(max_pulse_size=max_pulse_size, debug=debug)
        
        # Compose file_collections from input
        self.file_collections = {}
        if isinstance(event_files, str):
            self.file_collections.update({'tmp': [event_files]})
        elif isinstance(event_files, (list, tuple)):
            if all(isinstance(_e, str) for _e in event_files):
                self.file_collections = dict(zip(range(len(event_files)), event_files))
            else:
                raise TypeError('all elements of event_files in a list-like must be type str')
        elif isinstance(event_files, dict):
            for _k, _v in event_files.items():
                if isinstance(_v, (list, tuple)):
                    if not all(isinstance(_e, str) for _e in _v):
                        raise TypeError(f'all elements of event_files[{_k}] must be type str')
                    else:
                        self.file_collections.update({_k: list(_v)})
                elif isinstance(_v, str):
                    self.file_collections.update({_k: [_v]})     
        else:
            raise TypeError(f'event_files type {type(event_files)} not supported')
        
        self.max_length = wcc.bounded_intlike(
            max_length,
            name='max_length',
            minimum=0,
            maximum=None,
            inclusive=False
        )

        self.tree = BufferTree(buff_class=TraceBuffer, max_length=self.max_length)
        self.used_file_collections = {}

    def pulse(self, x=None, **options):
        """
        Conduct a pulse that loads max_pulse_size keyed file lists from self.file_collections
        and append those data to a BufferTree with TraceBuffer buds. Input x is a bool
        switch that indicates whether the BufferTree (self.tree) should be re-initialized
        (i.e., delete all data) at the start of this pulse
        
        :: INPUTS ::
        :param x: [bool] - should self.tree be re-initialized?
                    True = DELETE ALL BUFFERED DATA
                    False = Preserve all buffered data and attempt appending the next
                            set of load files to matching station ID's
                            NOTE: This comes with it's own

        """
        if len(self.used_file_collections) % self.reinit_period == 0:
            self.tree = BufferTree(buff_class=TraceBuffer, max_length=self.max_length)

        # Get initial length of file_collections
        fclen = len(self.file_collections)
        # Pose max_pulse_size as a for-loop, in the event multiple loads are specified
        for _i in range(self.max_pulse_size):
            # Early stopping clause
            if fclen == 0:
                break
            # Early stopping clause
            elif _i + 1 > fclen:
                break
            # Core process
            else:
                # form stream
                st = Stream()
                _k, _v = self.file_collections.popitem()
                for _f in _v:
                    st += read(_f)
                # Append stream to tree
                self.tree.append_stream(st, **options)
                # Shift file list to used 
                self.used_file_collections.update({_k: _v})
        
        y = self.tree

        return y
        


        
