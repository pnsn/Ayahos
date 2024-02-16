from wyrm import Wyrm, TraceMsg, DEQ_Dict
from obspy.realtime import RtTrace
from obspy import UTCDateTime
import fnmatch

class BuffWyrm(Wyrm):

    def __init__(
            self,
            sub_str='*.*.*.*',
            buffer_sec=150,
            max_pulse_size=12000
            ):
        """
        Initialize a BuffWyrm object that buffers TraceMsg data for distinct
        SNCL codes and provides access to these buffers for a subsequent WindWyrm

        """
        Wyrm().__init__(self)

        if not isinstance(sub_str, str):
            raise TypeError('sub_str must be type str')
        elif len(sub_str.split('.')) != 4:
            raise SyntaxError('sub_str must be a 4-element, "."-delimited string')
        else:
            self.sub_str = sub_str

        if isinstance(buffer_sec, float):
            self.blen = buffer_sec
        elif isinstance(buffer_sec, int):
            self.blen = float(buffer_sec)
        else:
            raise TypeError('buffer_sec must be type float or int')
        
        if self.blen < 0:
            raise ValueError('buffer_sec must be positive')

        if isinstance(max_pulse_size, int):
            self._pulse_size = float(max_pulse_size)
        else:
            raise TypeError('max_pulse_size must be type int')
        
        if self.blen < 0:
            raise ValueError('buffer_sec must be positive')

        # Initialize DEQ_Dict
        self.queues = DEQ_Dict(
            extra_fields={'buffer': RtTrace(max_length=self.blen)})
        

    def _validate_sncl(self, sncl, verbose=False):
        if not isinstance(sncl, str):
            if verbose:
                print('sncl must be type str')
            return False
        elif fnmatch.filter([sncl], self.sub_str) == [sncl]:
            return True
            

    def _buffer_tracemsg(self, tracemsg, debug=False):
        if not isinstance(tracemsg, TraceMsg):
            raise TypeError('tracemsg must be type TraceMsg')
        else:
            pass
        # Get SNCL code
        _sncl = tracemsg.scnl
        # If label is invalid, cease operation unless debugging
        if not self._validate_sncl(self, _sncl):
            if debug:
                raise SyntaxError('tracemsg.sncl is not compatable with self.sub_str')
            else:
                pass
        # If label is valid
        else:
            # If label is currently not in queues, create new entry
            if _sncl not in self.queues.keys():
                self._generate_new_entry(self, tracemsg)
            # Otherwise append Trace rendition of tracemsg to RtTrace object
            else:
                self.queues[_sncl]['buffer'].append(tracemsg.to_trace())
            # Assess if the RtTrace has enough data to ge

    def pulse(self, x):
        """
        :param x: [DEQ_Dict] from a preceding RingWyrm with EW->PY flow
        """
        _iwhile = 0
        killswitch = False
        while _iwhile < self._pulse_size or not killswitch:
            for _sncl in x.keys():
                _msg = x.pop(_sncl,queue='q')
                # Handle empty queue

                # If Trace
                if self._validate_sncl(_sncl)

                # If anything else

                
        # def _window_data(self, sncl):
        #     if sncl in self.queues.keys():
        #         que = self.queues[sncl]
        #         rttr = que['buffer']
        #         wt0 = que['next_starttime']
        #         wt1 = que['next_endtime']
        #         rtt0 = rttr.stats.starttime
        #         rtt1 = rttr.stats.endtime
        #         if wt0 is None:
        #             que['next_starttime'] = rtt0
        #             que['next_endtime'] = 








        def _generate_new_entry(self, tracemsg):
            if not isinstance(tracemsg, TraceMsg):
                raise TypeError('tracemsg must be type TraceMsg')
            else: 
                pass
            _scnl = tracemsg.scnl
            if _sncl not in self.queues.keys():
                # Check if candidate is 
                if fnmatch.filter([_sncl],self.sub_str) == [_sncl]