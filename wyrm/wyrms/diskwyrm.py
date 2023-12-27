import os
from glob import glob
from collections import deque
from wyrm.wyrms.wyrm import Wyrm
from wyrm.core.message import _BaseMsg, TraceMsg, HDEQ_Dict
from obspy import read
import fnmatch


class Disk2PyWyrm(Wyrm):
    """
    This Wyrm provides a pulsed read-from-disk functionality
    
    """
    def __init__(self, fdir, file_glob_str='*.mseed', file_format='MSEED', max_pulse_size=50, max_queue_size=50, max_age=50):
        Wyrm.__init__(self)
        # Initialize Heirarchical DEQueue
        self.hdeq = HDEQ_Dict()

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