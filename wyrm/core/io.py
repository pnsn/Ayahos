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
|   |          and a single ring connection index (conn_index)
|   |          Symbolic Representation: O
|   |            
|   =-->Wave2PyWyrm - Provides class methods and attributes to pull wave messages
|   |                   from a WAVE_RING into Python, reformat the messages as
|   |                   WaveMsg objects, and organize them into a dictionary with
|   |                   LOGO (SNCL code) keys and deques to hold queues of messages
|   |                   for the same
|   |               Symbolic Representation Ov^->
|   |
|   =-->Py2WaveWyrm - Provides class methods and attributes to submit WaveMsg
|   |                 objects into a target WAVE-like RING in Earthworm
|   |               Symbolic Representation v^->O
|   |
|   =-->Pick2PyWyrm - Provides class methods an attributes to pull pick messages
|   |               from a PICK_RING into Python
|   |               Symbolic Representation: O|->
|   |   
|   =-->Py2PickWyrm - Provides class methods and attributes to send PickMsg objects
|                   in Python to a PICK_RING in EarthWorm
|                   Symbolic Representation |->O
|
|
=-->DiskWyrm - Adds attributes and methods to house I/O with local storage
    |          built on top of the ObsPy `read` method and `Trace` and `Stream`
    |          classes.
    |          Symbplic Representations: D-> / ->D
    |
    =-->TankWyrm - Provides attributes and methods to make a rudamentary,
                pure-python approximation of an EarthWyrm tank player WAVE_RING

:attribution:
    This module builds on the PyEarthworm (C) 2018 F. Hernandez interface
    between an Earthworm Message Transport system and Python distributed
    under an AGPL-3.0 license.

"""
import os
import pandas as pd
from glob import glob
from collections import deque
from wyrm.core.wyrm import Wyrm
from wyrm.core.message import _BaseMsg, TraceMsg, StreamMsg, DEQ_Dict
from obspy import Stream, read
import fnmatch
import PyEW


##########################
### RINGWYRM BASECLASS ###
##########################
class RingWyrm(Wyrm, DEQ_Dict):
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
            mtype='TYPE_TRACEBUFF2',
            mcode=19,
            flow_direction='to python',
            max_pulse_size=12000,
            max_queue_length=150,
            max_age=60):
        
        # Initialize parent classes
        Wyrm.__init__(self)
        DEQ_Dict.__init__(self)

        # Run compatability checks on module
        if isinstance(module, PyEW.EWModule):
            self.module = module
        else:
            raise TypeError("module must be a PyEW.EWModule object!")

        # Compatability check for conn_info
        # Handle Series Input
        if isinstance(conn_info, pd.Series):
            if all(x in ['RING_ID','RING_Name'] for x in conn_info.index):
                self.conn_info = pd.DataFrame(conn_info).T
            else:
                raise KeyError('new_conn_info does not contain the required keys')
        # Handle DataFrame input
        elif isinstance(conn_info, pd.DataFrame):
            if len(conn_info) == 1:
                if all(x in ['RING_ID','RING_Name'] for x in conn_info.columns):
                    self.conn_info = conn_info
                else:
                    raise KeyError('new_conn_info does not contain the required keys')
            else:
                raise ValueError('conn_info must resemble a 1-row DataFrame')
        # Kick TypeError for any other input
        else:
            raise TypeError('conn_info must be a Series or 1-row DataFrame (see RingWyrm doc)')
        # Final bit of formatting for __repr__ purposes
        if self.conn_info.index.name is not 'index':
            self.conn_info.index.name = 'index'
        
        # Compatability check for flow_direction
        if flow_direction.lower() in ['from ring','to python','to py','from earthworm','from ew','from c','c2p','ring2py', 'ring2python']:
            self._from_ring = True
        elif flow_direction.lower() in ['from python','to ring','from py','to earthworm','to c','to ew', 'p2c','py2ring','python2ring']:
            self._from_ring = False
        else:
            raise ValueError('flow direction type {flow_direction} is not supported. See RingWyrm header doc')
        
        # Comptatability check for message type using wyrm.core.message._BaseMsg
        try: 
            test_msg = _BaseMsg(mtype=mtype, mcode=mcode)
            self.mtype = test_msg.mtype
            self.mcode = test_msg.mcode
        except TypeError:
            raise TypeError(f'from mtype:mcode | {mtype}:{mcode}')
        except SyntaxError:
            raise SyntaxError(f'from mtype:mcode | {mtype}:{mcode}')
        except ValueError:
            raise ValueError(f'from mtype:mcode | {mtype}:{mcode}')
       
        # Initialize buffer dictionary & limit indices
        self._max_pulse_size = max_pulse_size
        self._max_queue_size = max_queue_length
        self._max_queue_age = max_age

    def __repr__(self):
        rstr = f"Module: {self.module}\n"
        rstr += f"Conn Info: {self.conn_info}\n"
        rstr += f"{DEQ_Dict.__repr__(self)}"
        return rstr

    def change_conn_info(self, new_conn_info):
        if isinstance(new_conn_info, pd.DataFrame):
            if len(new_conn_info) == 1:
                if all(x in ['RING_Name','RING_ID'] for x in new_conn_info.columns):
                    self.conn_info = new_conn_info
                else:
                    raise KeyError('new_conn_info does not contain the required keys')
            else:
                raise ValueError('new_conn_info must be a 1-row pandas.DataFrame or pandas.Series')
        elif isinstance(new_conn_info, pd.Series):
            if all(x in ['RING_Name','RING_ID'] for x in new_conn_info.index):
                self.conn_info = pd.DataFrame(new_conn_info).T
            else:
                raise KeyError('new_conn_info does not contain the required keys')

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
            if _qad['age'] > self._max_queue_age:
                # Reset deque to an empty deque
                _qad['q'] = deque([])
                # Reset age to 0
                _qad['age'] = 0
            # If too young to FLUSH
            else:
                # Check if queue is too long
                if len(_qad['q']) > self._max_queue_size:
                    # Run while loop that terminates when the queue is max_queue size
                    while len(_qad['q']) > self._max_queue_size:
                        junk = _qad['q'].pop()
                    # NOTE: Age modification in CLEAN is ambiguous
                # If queue is shorter than max, do nothing
                else:
                    pass
        # END OF _flush_clean_queue
                
    def _to_ring_pulse(self, x=None):
        # If working with wave-like messaging, use class-methods
        # written into TraceMsg
        if self.mtype == 'TYPE_TRACEBUF2' and self.mcode == 19:
            # Iterate across all _sncl
            deltas = {}
            for _k in self.queue_dict.keys():
                # Isolate SCNL-keyed deque
                _qad = self.queue_dict[_k]
                _qlen = len(_qad['q'])
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
                        _msg = _qad['q'].pop()
                        # If instance of TraceMsg
                        if isinstance(_msg, TraceMsg):
                            # Send to EW
                            _msg.to_ew(self._conn_index)
                        # Otherwise, append value back to the head of the queue
                        else:
                            _qad['q'].appendleft(_msg)
                    # If all messages are recycled, increase age
                    if _qlen == len(_qad['q']):
                        _qad['age']+=1
            else:
                NotImplementedError('Other mtype:mcode combination handling not yet developed')
        # END of _to_ring_pulse(x)

    def _from_ring_pulse(self, x=None):
        # If working with wave-like messaging, use class-methods
        # written into TraceMsg
        if self.mtype == 'TYPE_TRACEBUF2' and self.mcode == 19:
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
                    # If SNCL exists in queue keys
                    if _msg.sncl in self.queue_dict.keys():
                        # Appendleft
                        self.queue_dict[_msg.sncl]['q'].appendleft(_msg)
                        # and reset age
                        self.queue_dict[_msg.sncl]['age'] = 0
                    # If new SNCL, generate a new queue
                    else:
                        new_queue = {'q': deque([_msg]), 'age': 0}
                        self.queue_dict.update(new_queue)
        # If mtype:mcode arent for TYPE_TRACEBUF2, kick "IN DEVELOPMENT" error
        else:
            NotImplementedError("Other mtype:mcode combination handling not yet developed")
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
            self._flush_clean_queue()

        # If flowing from a ring (EW->PY)
        elif not self._to_ring:
            # Assess flush/clean for queues first
            self._flush_clean_queue()
            # Then bring in new data/increase age of un-updated queues
            self._from_ring_pulse(x=x)
        else:
            raise RuntimeError('Dataflow direction from self._to_ring not valid')
        # Make queue_dict accessible to subsequent (chained) Wyrms
        y = self.queue_dict
        return y
            


class Disk2PyWyrm(Wyrm, DEQ_Dict):
    """
    This Wyrm provides a pulsed behavior that reads waveform files from disk and 
    presents a queue_dict attribute with identical formatting as RingWyrm, providing
    a 
    
    """
    def __init__(self, fdir, file_glob_str='*.mseed', file_format='MSEED', max_pulse_size=50, max_queue_size=50, max_age=50):
        Wyrm.__init__(self)
        # Initialize queue_dict
        DEQ_Dict.__init__(self)

        # Compatability check on dir
        if not isinstance(fdir, str):
            raise TypeError('fdir must be type str')
        elif not os.path.exists(fdir):
            raise FileExistsError(f'Directory {fdir} does not exist')
        else:
            self.fdir = fdir

        if not isinstance(file_glob_str, str):
            raise TypeError('file_glob_str must be type str')
        else:
            flist = glob(os.path.join(self.fdir,file_glob_str))
            if len(flist) == 0:
                raise SyntaxError(f"No files glob'd with {os.path.join(self.fdir, file_glob_str)}")
            else:
                flist.sort()
                self.file_queue = deque(flist)

        # Compatability Check on file_format
        if isinstance(file_format, str):
            self.file_format = file_format
        else:
            raise TypeError
        # Compatability check on max_ integer values
        try:
            max_pulse_size/1
            self.max_pulse_size = int(max_pulse_size)
        except TypeError:
            raise TypeError
        try:
            max_queue_size/1
            self.max_queue_size = int(max_queue_size)
        except TypeError:
            raise TypeError
        try:
            max_age/1
            self.max_age = int(max_age)
        except TypeError:
            raise TypeError
        
    def __repr__(self):
        rstr = f"Source Dir: {self.fdir}\n"
        rstr += f"No. Queued Files: {len(self.file_queue)}\n"
        rstr += DEQ_Dict.__repr__(self)
        return rstr

    
    def pulse(self, x=None):
        """
        Load up to max_pulse_size files from disk
        """
        x = x
        for _ in range(self.max_pulse_size):
            # If file queue is empty
            if len(self.file_queue) == 0:
                break
            else:
                # Pop file name from file_queue
                _file = self.file_queue.pop()
                # Try to load file
                try:
                    _st = read(_file)
                # Propagate TypeError from obspy.read if it raises TypeError
                except TypeError:
                    raise TypeError(f'something went wrong loading {_file}')
                # NOTE: For development only - must clean up hanging except
                except:
                    print('SOMETHING ELSE WENT WRONG WITH obspy.read')
                    breakpoint()
                
                # If _st read successfully, iterate across traces in _st
                for _tr in _st:
                    # Convert into TraceMsg, attempting to inherit dtype from the trace
                    try:
                        _trMsg = TraceMsg(_tr, dtype=_tr.data.dtype)
                    # Failing that, interpret as float32
                    except TypeError:
                        _trMsg = TraceMsg(_tr, dtype='f4')
                    # Use DEQ_Dict append to add 
                    self.append_msg_left(_trMsg)

        # Present access to self.queues as a dictionary
        y = self.queues
        return y

class Py2DiskWyrm(Wyrm):
    def __init__(self,
                 root_dir,
                 sncl_filter='*.*.*.*',
                 save_format='{NET}/{STA}/{SNCL}_{TS}_{TE}.mseed',
                 max_pulse_size=1000):

        # Compatability checks for root_dir
        if not isinstance(root_dir,str):
            raise TypeError('root_dir must be type str')
        elif not os.path.exists(root_dir):
            os.makedirs(root_dir)
            print(f'Generated {root_dir}')
            self.root_dir = root_dir
        else:
            self.root_dir = root_dir

        # Compatability checks for sncl_filter
        if not isinstance(sncl_filter, str):
            raise TypeError('sncl filter must be type str')
        elif len(sncl_filter.split('.')) != 4:
            raise SyntaxError('sncl filter should be four "." delimited strings, minimally *.*.*.*')
        else:
            self.sncl_filter = sncl_filter

        # Compatability checks for save_format
        if not isinstance(save_format, str):
            raise TypeError('save_format must be type str')
        else:
            self.save_format=save_format
        
        # Compatability checks for max_pulse_size
        if not isinstance(max_pulse_size, (int, float)):
            raise TypeError('max_pulse_size must be int-like')
        else:
            self.max_pulse_size = int(max_pulse_size)

    def __repr__(self):
        rstr = f'OUTPUT FMT: {os.path.join(self.root_dir, self.save_format)}\n'
        rstr += f'SNCL Filt: {self.sncl_filter} | '
        rstr += f'MAX Pulse: {self.max_pulse_size}\n'
        return rstr


    def pulse(self, x):
        """
        Pulse this Py2DiskWyrm with input of a DEQ_Dict.queues dictionary
        and prune off TraceMsg entries for S.N.C.L keyed queues that match
        self.sncl_filter

        :: INPUT ::
        :param x: [dict] from a DEQ_Dict (it's queues attribute)
        
        :: OUTPUT ::
        :return y: [None]
        """

        # Fitler for matching keys
        sncl_list = fnmatch.filter(x.keys(), self.sncl_filter):
        # Get number of sncl entries
        _j = len(sncl_list)
        # Get count of total elements
        _e = sum([len(x[_sncl]['q']) for _sncl in sncl_list])
        # Iterate for max_pulse_size
        for _i in range(self.max_pulse_size):
            # If total elements hits 0, break iteration
            if _e <= 0:
                break
            # Otherwise, proceed
            else:
                # Use modulus index to wrap sncl list if it's shorter than max_iter
                _sncl = sncl_list[_i%_j]
                # If this queue is empty, continue iterating
                if len(x[_sncl]['q']) == 0:
                    continue
                # If queue is not empty, pop message
                else:
                    _msg = x[_sncl]['q'].pop()
                    # If message is a wyrm *Msg
                    if isinstance(_msg, _BaseMsg):
                        # If the message is not a TraceMsg
                        if not isinstance(_msg, TraceMsg):
                            # Append the item back on the list
                            x[_sncl]['q'].appendleft(_msg)
                            # Do not decrease element counter
                            _e -= 0
                        # If the message is a TraceMsg, proceed with saving
                        else:
                            # Format file name
                            fname = self.save_format.format(NSLC=_msg.id,
                                                            SNCL=_msg.sncl,
                                                            STA=_msg.stats.station,
                                                            NET=_msg.stats.network,
                                                            CHA=_msg.stats.channel,
                                                            LOC=_msg.stats.location,
                                                            TS=_msg.stats.starttime,
                                                            TSY=_msg.stats.starttime.year,
                                                            TSJ=_msg.stats.starttime.julday,
                                                            TSH=_msg.stats.starttime.hour,
                                                            TSm=_msg.stats.starttime.minute,
                                                            TSs=_msg.stats.starttime.second,
                                                            TE=_msg.stats.endtime,
                                                            TEY=_msg.stats.endtime.year,
                                                            TEJ=_msg.stats.endtime.julday,
                                                            TEH=_msg.stats.endtime.hour,
                                                            TEm=_msg.stats.endtime.minute,
                                                            TEs=_msg.stats.endtime.second)
                            # Compose full file path/name
                            fnp = os.path.join(self.root_dir, fname)
                            # Extract path
                            path = os.path.split(fnp)
                            # Create path if it doesn't exist
                            if not os.path.exists(path):
                                os.makedirs(path)
                            # Write to disk
                            _msg.write(fnp)
                            # Decrease element count by 1
                            _e -= 1
        y = None
        return y


        














#########################
### RINGWYRM CHILDREN ###
#########################

class Wave2PyWyrm(RingWyrm):
    """
    Child class of RingWyrm, adding a pulse(x) method
    to complete the Wyrm base definition. Also adds a maximum
    iteration attribute to limit the number of wave messages
    that can be pulled in a single Wave2PyWyrm.pulse(x) call.
    """
    def __init__(
            self,
            module,
            conn_index,
            stations='*',
            networks='*',
            channels='*',
            locations='*',
            max_iter=int(1e5),
            max_staleness=300):
        """
        Initialize a Wave2PyWyrm object
        :: INPUTS ::
        :param module: [PyEW.EWModule] pre-initialized module
        :param conn_index: [int] index of pre-established module
                        connection to an EW WAVE RING
        :param stations: [str] or [list] station code string(s)
        :param networks: [str] or [list] network code string(s)
        :param channels: [str] or [list] channel code string(s)
        :param locations: [str] or [list] location code string(s)
        :param max_iter: [int] (default 10000) maximum number of
                        messages to receive for a single pulse(x)
                        command
        :param max_staleness: [int] (default 300) maximum number of
                        pulses an given SNCL keyed entry in queue_dict
                        can go without receiving new data.

        :: ATTRIBUTES ::
        -- Inherited from RingWyrm --
        :attrib module: [PyEW.EWModule] connected EWModule
        :attrib conn_index: [int] connection index

        -- New Attributes --
        :attrib queue_dict: [dict] SNCL keyed dict with tuple values:
                        
                        {'S.N.C.L': (deque, staleness_index)}
                        e.g. 
                        {'GNW.UW.BHN.--': (deque([WaveMsg]),0)}

                        with the dequeue adding newest entries on 
                        the left end of the deque. Subsequent
                        interactions with this attribute should
                        pop() entries to pull the oldest members

                        staleness_index starts at 0 and increases
                        for each pulse where a given SNCL code
                        does not appear. This serves as a mechanism
                        for clearing out outdated data that may arise
                        from a station outage and help prevent ingesting
                        large data gaps in subsequent

        """
        # Use RingWyrm baseclass initialization
        super().__init__(module, conn_index)
        self.stations = None
        self.networks = None
        self.channels = None
        self.locations = None

        # Conduct compatability checks on SNCL filtering inputs
        iter_tuples = [('stations', self.stations, stations),
                       ('networks', self.networks, networks),
                       ('channels', self.channels, channels),
                       ('locations', self.locations, locations)]
        for _name, _selfx, _inx in iter_tuples:
            if isinstance(_inx, (str, int)):
                _selfx = str(_inx)
            elif isinstance(_inx, (list, deque)):
                if all(isinstance(_sta, (str, int)) for _sta in _inx):
                    _selfx = [str(_sta) for _sta in _inx]
                else:
                    _selfx = [str(_sta) for _sta in _inx if isinstance(_sta, (str, int))]
                    if len(_selfx) == 1:
                        _selfx = _selfx[0]
                    elif len(_selfx) == 0:
                        _selfx = None
                    UserWarning(f'{len(_selfx)} of {len(_inx)} passed validity checks')
            else:
                raise TypeError(f'Invalid input type for {_name}. Accepted types: str, int, list, deque')
       
        # Compatability check for _max_iter
        try:
            self._max_iter = int(max_iter)
        except TypeError:
            print('max_iter must be int-like!')
            raise TypeError
        
        # Compatability check for _max_staleness
        try:
            self._max_staleness = int(max_staleness)
        except TypeError:
            print('max_staleness must be positive and int-like')
            raise TypeError
        

    def __repr__(self):
        """
        Command line representation of class object
        """
        rstr = super().__repr__()
        rstr += f'Max Iter: {self._max_iter}\n'
        rstr += f'Max Staleness: {self._max_staleness}\n'
        rstr += f'STA Filt: {self.stations}\n'
        rstr += f'NET Filt: {self.networks}\n'
        rstr += f'CHA Filt: {self.channels}\n'
        rstr += f'LOC Filt: {self.locations}\n'
        return rstr
    
    def _get_wave_from_ring(self):
        """
        Fetch the next available message from the WAVE ring
        connection and check if it is on the approved list
        defined by SNCL information

        :: OUTPUT ::
        :return msg: [wyrm.core.pyew_msg.WaveMsg] 
                            - if a wave message is received and passes filtering
                     [False] 
                            - if an empty wave message is received
                     [True]
                            - if a wave message is received and fails filtering
        """
        wave = self.module.get_wave(self.conn_index)
        # If wave message is empty, return False
        if wave == {}:
            msg = False
        # If the wave has information, check if it's on the list
        else:
            sta_bool, net_bool, cha_bool, loc_bool = False, False, False, False
            # Do sniff-test that 
            if all(_k in wave.keys() for _k in ['station','network','channel','locaiton'])
                # Station Check
                if isinstance(self.stations, list):
                    if wave['station'] in self.stations:
                        sta_bool = True
                elif isinstance(self.stations, str):
                    fnsta = fnmatch.filter([wave['station']], self.stations)
                    if len(fnsta) == 1:
                        sta_bool = True

                # Network Check
                if isinstance(self.networks, list):
                    if wave['network'] in self.networks:
                        net_bool = True
                elif isinstance(self.networks, str):
                    fnnet = fnmatch.filter([wave['network']], self.networks)
                    if len(fnnet) == 1:
                        net_bool = True

                # Channel Check
                if isinstance(self.channels, list):
                    if wave['channel'] in self.channels:
                        cha_bool = True
                elif isinstance(self.channels, str):
                    fncha = fnmatch.filter([wave['channel']], self.channels)
                    if len(fncha) == 1:
                        cha_bool = True

                # Location Check
                if isinstance(self.locations, list):
                    if wave['location'] in self.locations:
                        loc_bool = True
                elif isinstance(self.locations, str):
                    fnloc = fnmatch.filter([wave['location']], self.locations)
                    if len(fnloc) == 1:
                        loc_bool = True
            else:
                print('Received something that does not look like a wave message')
                raise TypeError
            
            # If all fields match filtering criteria
            if all([sta_bool, net_bool, cha_bool, loc_bool]):
                # Convert message from type dict to type WaveMsg
                msg = WaveMsg(wave)
            # If filtering criteria fails, but there was a valid message
            else:
                msg = True            
            return msg

    def _add_msg_to_buffer_dict(self, wavemsg):
        """
        Add a WaveMsg to the buffer_dict, creating a new entry
        {'S.N.C.L': (deque([wavemsg]), staleness_index)}
        if the WaveMsg SNCL code is not in the buffer_dict keys,
        and left-appending the WaveMsg to an active deque keyed to
        code if it already exists and resets staleness_index = 0

        :: INPUT ::
        :param wavemsg: [wyrm.core.message.WaveMsg] WaveMsg formatted
                        tracebuff2 message contents
    
        :: UPDATE ::
        :attrib buffer_dict: -- See above.
                        
        :: OUTPUT ::
        :None:
        """
        # If the SNCL code of the WaveMsg is new, create a new dict entry
        if wavemsg.code not in self.buffer_dict.keys():
            new_member = {wavemsg.code: (deque([wavemsg]), 0)}
            self.buffer_dict.update(new_member)
            # And return True state
            return wavemsg.code
        # If the SNCL code of the WaveMsg is already known
        elif wavemsg.code in self.buffer_dict.keys():
            # Left append the message to the existing queue
            new_message = deque([wavemsg])
            self.buffer_dict[wavemsg.code][0].appendleft(new_message)
            # reset "staleness index"
            self.buffer_dict[wavemsg.code][1] = 0
            return wavemsg.code
        # If something unexpected happens, raise KeyError
        else:
            raise KeyError('Something went wrong with matching keys')

    def _update_staleness(self,updated_codes=[]):
        """
        Increase the "staleness index" of each SNCL keyed
        entry in self.buffer_dict by 1 that is not present
        in the provided updated_codes list
        """
        # For each _k(ey) in buffer_dict
        for _k in self.buffer_dict.keys():
            # If the key is not in the updated_codes list
            if _k not in updated_codes:
                # If max staleness is met
                if self.buffer_dict[_k][1] >= self._max_staleness:
                    # Determine if there are any data in the queue
                    nele = len(self.buffer_dict[_k][0])
                    # If there are data, pop the last entry in the queue
                    if nele > 0:
                        self.buffer_dict[_k][0].pop();
                    # If there are no data in the queue, pop the entire SNCL keyed entry
                    else:
                        self.buffer_dict.pop(_k);
                # Otherwise increase that entry's staleness index by 1
                else:
                    self.buffer_dict[_k][1] += 1
        

    def pulse(self, x=None):
        """
        Iterate for some large number (max_iter) pulling
        wave messages from the specified WAVE RING connection,
        appending to a list if the messages are unique, and stopping
        the iteration at the first instance of an empty dictionary
            i.e., wave = {}. 
        Not to be confused with a dataless wave message 
            i.e. wave['data'] =  np.array([]))
        
        :: INPUT ::
        :param x: [NoneType] or [int] Connection index
                [NoneType] is default, uses self.conn_index for the connection index
                [int] superceds self.conn_index.
        
        :: OUTPUT ::
        :return y: [list] list of 
        """
        # Create holder for codes of updated SNCL buffers
        updated_codes = []
        # Start large maximum iteration count for-loop
        for _ in range(self._max_iter):
            # Get a single wave message from the ring
            wavemsg = self._get_wave_from_ring()
            # If the wave message is indeed a WaveMsg
            if isinstance(wavemsg, WaveMsg):
                # Attempt to add message to buffer_dict if it matches SNCL filters
                updated_code = self._add_msg_to_buffer_dict(wavemsg)
                # If code was not already in the updated SNCL codes
                if updated_code  not in updated_codes:
                    # Add new code to updated_codes
                    updated_codes.append(updated_code)
            
            # If the wave message is carrying a continue/break Bool code
            elif isinstance(wavemsg, bool):
                # If wavemsg is True, this signals a valid message that did not
                # meet SNCL filtering specifications
                if wavemsg:
                    pass
                # If wave is False, this signals an empty message. 
                elif not wavemsg:
                    break
            
            else:
                raise RuntimeError('Something unexpected happened when parsing a wave message from Earthworm...')
        
        # Increase staleness index and clear out overly stale data
        self._update_staleness(updated_codes)

        # Pass self.buffer_dict as output
        y = super().pulse(x)
        return y


class Py2WaveWyrm(RingWyrm):
    """
    Child class of RingWyrm, adding a pulse(x) method
    to complete the Wyrm base definition. Also adds a maximum
    message buffer (_max_msg_buff) attribute to limit the number 
    of previously sent messages buffered in this object for the purpose
    of preventing repeat message transmission.
    """

    def __init__(self, module, conn_index, max_msg_buff=1000):
        """
        Initialize a Py2WaveWyrm object
        :: INPUTS ::
        :param module: [PyEW.EWModule] Connected module
        :param conn_index: [int] Connection index for target WAVE RING
        :param max_msg_buff: [int-like] Maximum number of messages
                        that are buffered in object memory before purging
        """
        super().__init__(module, conn_index)
        self._max_msg_buff = max_msg_buff

    def __repr__(self):
        rstr = super().__repr__()
        rstr += f'Max Buffer: {self._max_msg_buff}\n'
        return rstr


    def pulse(self, x):
        """
        Submit data from PyEW_WaveMsg objects contained in `x`
        to the specified WAVE RING as tracebuff2 messages
        :: INPUTS ::
        :param x: [list] of PyEW_WaveMsg objects

        :: OUTPUT ::
        :param y: [int] count of successfully sent messages

        """
        send_count = 0
        # If working with a single message, ensure it's a list
        if isinstance(x, PyEW_WaveMsg):
            x = [x]
        # Iterate across list of input message(s)
        for _msg in x:
            # Ensure the message is not a duplicate of a recent one
            if isinstance(_msg, PyEW_WaveMsg) and _msg not in self.past_msg_queue:
                # SEND MESSAGE TO RING
                try:
                    self.module.put_wave(_msg.to_tracebuff2(), self.conn_index)
                    send_count += 1
                # TODO: Clean this dev choice up...
                except:
                    continue
                # Append message to to dequeue on left
                self.past_msg_queue.appendleft(_msg)
                # If we've surpassed the buffer allotment, pop oldest msg at right
                if len(self.past_msg_queue)+ 1 < self._max_msg_buff:
                    self.past_msg_queue.pop()
        # output number of sent messages
        y = int(send_count)
        return y

# class Pick2PyWyrm(RingWyrm):


# class Py2PickWyrm(RingWyrm):
    


##########################
### DISKWYRM BASECLASS ###
##########################


class DiskWyrm:
    """
    Provide a Wyrm wrapper around ObsPy read/write utilities
    for waveform data 
    """
    def __init__(self, files=deque(), pulse_size=-1, read_kwargs = {}):
        # Compatablitiy checks for files
        if isinstance(files, str):
            self.files = deque([files])
        elif isinstance(files, list):
            self.files = deque(files)
        elif isinstance(files, deque):
            self.files = files
        else:
            raise TypeError
        # Compatability check for pulse_size
        try:
            pulse_size/1
        except TypeError:
            raise TypeError
        if isinstance(pulse_size, float):
            pulse_size = int(pulse_size)
        if pulse_size < 0:
            pulse_size = len(self.files)
        if isinstance(read_kwargs, dict):
            self.read_kwargs = read_kwargs
        else:
            raise TypeError
        self.stream = Stream()
        self.fails = deque([])

    def __repr__(self):
        rstr = f'Files in queue: {len(self.files)}\n'
        rstr += f'Pulse size: {self.pulse_size}\n'
        rstr += f'Buffered Stream:\n{self.stream}\n'
        return rstr
    
    def pulse(self, x):
        for _ in range(self.pulse_size):
            if len(self.files) > 0:
                _f = self.files.pop()
                try:
                    _st = read(_f, **self.read_kwargs)
                    self.stream += _st
                except TypeError:
                    self.fails.appendleft(_f)
        y = self.stream
        return y