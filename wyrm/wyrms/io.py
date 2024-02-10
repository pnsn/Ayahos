import os
from glob import glob
from wyrm.wyrms.base import Wyrm
from wyrm.buffer.structures import TieredBuffer
from wyrm.buffer.trace import TraceBuff
from wyrm.util.PyEW_translate import wave2trace, trace2wave
# import PyEW
from obspy import read
import fnmatch



class RingWyrm(Wyrm):
    
    def __init__(self, module=None, conn_id=0, pulse_method_str='get_wave', max_pulse_size=None, debug=False, **options):
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

        if not isinstance(pulse_method_str, str):
            raise TypeError('pulse_method_str must be type_str')
        elif pulse_method_str not in ['get_wave','put_wave','get_msg','put_msg','get_bytes','put_bytes']:
            raise ValueError(f'pulse_method_str {pulse_method_str} unsupported. See documentation')
        else:
            self.pulse_method = pulse_method_str
        
        # Compatability checks for **options
        if self.pulse_method in ['get_msg','put_msg','get_bytes','put_bytes']:
            if 'msg_type' not in options.keys():
                raise TypeError('Optional inputs must include kwarg "msg_type"')
            else:
                self.msg_type=options['msg_type']
        # In the case of get/put_wave, default to msg_type=19 (tracebuff2)
        else:
            msg_type=19

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
        if 'get' in self.pulse_method:
            if 'wave' in self.pulse_method:
                msg = self.module.get_wave(self.conn_id)
            else:
                evalstr = f'self.module.{self.pulse_method}(self.conn_id, self.msg_type)'
                msg = eval(evalstr)
            # Flag empty message results as 'False'
            if msg == {}:
                msg = False
            elif msg == '':
                msg = False
            else:
                pass

            return msg

        elif 'put' in self.pulse_method:
            if 'wave' in self.pulse_method:
                self.module.put_wave(self.conn_id, x)
            else:
                evalstr = f'self.module.{self.pulse_method}(self.conn_id, self.msg_type, x)'
                eval(evalstr)
        


class WaveInWyrm(RingWyrm):
    """

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
        Initialize a RingWyrm object with a TieredBuffer + TraceBuff 

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


        # Let TieredBuffer handle the compatability check for max_length
        self.buffer = TieredBuffer(buff_class=TraceBuff, max_length=max_length, **options)

    def pulse(self, x='*'):
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
        :param x: [str] N.S.L.C formatted / unix wildcard compatable
                string for filtering read waveforms by their N.S.L.C ID
                (see obspy.core.trace.Trace.id). 
                E.g., 
                for only accelerometers, one might use:
                    x = '*.?N?'
                for only PNSN administrated broadband stations, one might use:
                    x = '[UC][WOC].*.*.?H?'
        
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
        for _ in range(self.pulse_size):
            # Pull tracebuff2 message from ring
            _wave = self.module.get_wave(self.conn_id)
            # Check if it's a null result (i.e., no new waves)
            if _wave == {}:
                # If it is a null, early stop to iteration
                break
            # If it has data (i.e., non-empty dict)
            else:
                # Convert PyEW wave to obspy trace
                trace = wave2trace(_wave)
                if len(fnmatch.filter([trace.id], x)) > 0:
                    # Append trace to buffer
                    # Use N.S.L.BandInst for key0
                    key0 = trace.id[:-1]
                    # Use Component Character for key1
                    key1 = trace.id[-1]
                    self.buffer.append(trace, key0, key1)
        # Present buffer object as output for sequencing
        y = self.buffer
        return y
        
    def __str__(self, extended=False):
        """
        Provide a string representation of this RingWyrm object

        :: INPUT ::
        :param extended: [bool] passed to TieredBuffer's __str__() method
                        to display buffer status.
        :: OUTPUT ::
        :return rstr: [str] representative string
        """
        rstr = super().__str__()
        rstr = f'Module: {self.module} | Conn ID: {self.conn_id}\n'
        rstr += 'FLOW:   EW --> PY\n\n'
        rstr += 'RingWyrm.buffer contents:\n'
        rstr += f'{self.buffer.__repr__(extended=extended)}'
        return rstr
    
    def __repr__(self, extended=False):
        rstr = self.__str__(extended=extended)
        return rstr



class RingOutWyrm()

### UPDATES NEEDED - NTS 2/9/2024
class Disk2PyWyrm(Wyrm):
    """
    This Wyrm provides a pulsed read-from-disk functionality
    
    """
    def __init__(self, fdir, file_glob_str='*.mseed', file_format='MSEED', max_pulse_size=50, max_queue_size=50, max_age=50):
        Wyrm.__init__(self)


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
        
    def __repr__(self, extended=False):
        rstr = f"Source Dir: {self.fdir}\n"
        rstr += f"No. Queued Files: {len(self.file_queue)}\n"
        rstr += f"{self.hdeq.__repr__(exdended=extended)}"
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
                    # Make branch (if needed)
                    self.hdeq._add_index_branch(_trMsg.sncl)
                    # Append message to HDEQ_Dict
                    self.hdeq.appendleft(_trMsg.sncl, _trMsg, key='q')

        # Present access to self.queues as a dictionary
        y = self.hdeq
        return y


class Py2DiskWyrm(Wyrm):
    """
    This Wyrm provides a pulsed write-to-disk utility
    """

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