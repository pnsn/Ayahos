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
import wyrm.util.compatability as wcc
#import PyEW
from obspy import Stream, read
from wyrm.core._base import Wyrm
from wyrm.util.pyew_translate import is_wave_msg, wave2trace, trace2wave
from wyrm.core.data import BufferTree, TraceBuffer


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
            self.msg_type = wcc.bounded_intlike(
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
    WAVE Ring and populating a TieredBuffer object for subsequent sampling
    by Wyrm sequences
    """

    def __init__(
        self,
        module=None,
        conn_id=0,
        max_length=150,
        max_pulse_size=12000,
        debug=False,
        **options
    ):
        """
        Initialize a EarWyrm object with a TieredBuffer + TraceBuff.
        This is a wrapper for a read-from-wave-ring RingWyrm object

        :: INPUTS ::
        :param module: [PyEW.EWModule] active PyEarthworm module object
        :param conn_id: [int] index number for active PyEarthworm ring connection
        :param max_length: [float] maximum TraceBuff length in seconds
                        (passed to TieredBuffer for subsequent buffer element initialization)
        :param max_pulse_size: [int] maximum number of get_wave() actions to execute per
                        pulse of this RingWyrm
        :param debug: [bool] should this RingWyrm be run in debug mode?
        :param **options: [kwargs] additional kwargs to pass to TieredBuff as
                    part of **buff_init_kwargs
                    see wyrm.buffer.structures.TieredBuffer
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
        self.tree = BufferTree(
            buff_class=TraceBuffer,
            max_length=max_length,
            **options
            )
        self.options = options
        # self._update_io_types(itype=(str, type(None)), otype=BufferTree)

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


class DiskWyrm(Wyrm):

    def __init__(self, event_files, max_length=300, max_pulse_size=1, debug=False):

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

    def pulse(self, x=True, **options):
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
        # Check if BufferTree should be re-initialized
        if x:
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
        


        
