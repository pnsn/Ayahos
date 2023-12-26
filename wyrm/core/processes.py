"""
:module: wyrm.core.unit_wyrms
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This module contains class definitions stemming from the Wyrm BaseClass
    that serve as segments of a Python-side processing line for "pulsed"
    data flow 

Wyrm (ð)
|
=-->BuffWyrm - x: access to the flat dictionary of 
|                 SNCL keyed deques of WaveMsg objects
|                 housed in a Wave2PyWyrm object
|              y: access to self.buffer.rttrace objects
|               Provides WaveMsg data subsampling and buffering, 
|               and data gap handling
|              Symbolic Representation: BUFF
|
=-->WindowWyrm - x: access to rttrace objects housed in a BuffWyrm object
|                y: access to self.out_buffer deque of MLWMsg objects
|               Provides methods and attributes to produce preprocessed,
|               striding windows for input to ML models.
|               Provides highly flexible class-method supported pipeline
|               construction through the use of chained `eval()` statements
|               on obspy.Trace and numpy.NdArray objects.



:attribution:
    This module builds on the PyEarthworm (C) 2018 F. Hernandez interface
    between an Earthworm Message Transport system and Python distributed
    under an AGPL-3.0 license.

"""
from wyrm.core.wyrm import Wyrm
import numpy as np
import pandas as pd
import seisbench.models as sbm
from obspy import UTCDateTime, Stream, Trace, Inventory
from obspy.realtime import RtTrace
from wyrm.core.message import TraceMsg
import fnmatch


class BuffWyrm(Wyrm):
    """
    This Wyrm handles WaveMsg data buffering in the form
    of a dictionary of rttrace objects

    HDEQ_Dict.queues --> self.buffer = Stream(SNCL: rttrace, ...}
    """

    def __init__(self, age_threshold=5, sncl_list=['GNW.UW.BHZ.--','GNW.UW.BHN.--','GNW.UW.BHE.--'], max_length=300):
        """
        Initialize a BuffWyrm object that subscribes to a particular set of SNCL codes (LOGO)

        :: ATTRIBUTES ::
        :attrib sncl_list: [list] ordered list of SNCL codes for a single
                            seismometer
        :attrib max_length: [float-like] number of seconds to set as RtTrace
        """
        # Static SNCL fields (one instance per site)
        if not isinstance(sncl_list, (list)):
            raise TypeError('sncl_list must be type list')
        elif not all(len(sncl.split('.')) != 4 for sncl in sncl_list):
            raise SyntaxError('sncl_list entries must be formatted as a "." delimited string (see default example)')
        else:
            pass

        if max_length <= 0:
            raise ValueError('max_length must be positive')
        elif not np.isfinite(max_length):
            raise ValueError('max_length must be finite')
        else:
            self.max_length = max_length
        
        if not isinstance(age_threshold, int):
            raise TypeError('age_threshold must be type int')
        elif gap_thresold < 1:
            raise ValueError('age_threshold must be positive valued')
        elif not np.isfinite(age_threshold):
            raise ValueError('age_threshold must be finite')
        else:
            self.age_threshold = age_threshold

        # Populate buffer
        self.buffer = {}
        for _k in self.sncl_list:
            comp = _k.split('.')[2][-1]
            if comp in ['Z','3']:
                comp = 'Z'
            elif comp in ['N','1']:
                comp = 'N'
            elif comp in ['E','2']:
                comp = 'E'
            else:
                raise SyntaxError(f'Provided sncl {_k} has invalid component code {comp}')
            self.buffer.update({_k:{'trace': RtTrace(max_length=self.max_length),
                                    'age': 0,
                                    'comp': comp}})
        
        self.next_starttime = None
        self.next_endtime = None

        

    def __resp__(self):
        rstr = f'Search String: {self.sncl_list}\n'
        rstr += f'Max Buff: {self.max_length:.3f} sec\n' 
        rstr += '--- Buffered RtTraces ---\n'
        for _k in self.buffer.keys()
            rstr += f'({self.buffer[_k]['comp']}) '
            rstr += f'{self.buffer[_k]['trace']} | '
            rstr += f'{self.buffer[_k]['age']} pulses'
        return rstr
         
    def _append_tracemsg_to_buffer(self, tracemsg):
        if not isinstance(tracemsg, TraceMsg):
            raise TypeError('tracemsg must be type TraceMsg')
        else:
            msg_sncl = tracemsg.sncl

        if msg_sncl not in self.buffer.keys():
            raise KeyError('tracemsg SNCL is not in this buffer')
        else:
            self.buffer[msg_sncl]['trace'].append(tracemsg.to_trace())
        #END
    
    def _claim_traces(self, hdeq_dict):
        for _k in self.buffer.keys():
            if _k in hdeq_dict.codes:
                _target = hdeq_dict._get_sncl_target(_k)
                _qlen = len(_target['q'])
                for _ in range(_qlen):
                    _msg = _target['q'].pop()
                    if isinstance(_msg, TraceMsg):
                        self._append_tracemsg_to_buffer(_msg)
                    else:
                        _target['q'].appendleft(_msg)
                # Reset age if the hdeq buffer got shorter
                if len(_target['q']) < _qlen:
                    self.buffer[_k]['age'] = 0
                # Add to age if the length of the buffer didn't change from this action
                else:
                    self.buffer[_k]['age'] += 1
            # Add to age if the component hasn't shown up in input hdeq_dict
            else:
                self.buffer[_k]['age'] += 1
    

    def _validate_trace_window(self, sncl):
        _tr = self.buffer[sncl]['trace']
        _age = self.buffer[sncl]['age']
        # If there are valid start/end times from supervisor
        if self.next_starttime not None and self.next_endtime not None:
            # If there are enough data to window the trace
            if _tr.stats.starttime >= self.next_starttime and _tr.stats.endtime <= self.next_endtime:
                return True
            # If either case fails, return false
            else:
                return False
        # If invalid start/end times from supervisor, return false
        else:
            return False
        
    def _validate_buffer_window(self):
        # Check if all traces have data in the specified window
        bool_list = [self.validate_trace_window(_k) for _k in self.buffer.keys()]
        if all(bool_list):
            return True
        
        elif any(bool_list):
            
        
    
        # If there are valid start/end times
        if self.next_starttime not None and self.next_endtime not None:
            bool_set = []
            for _k in self.buffer.keys():
                _rtts = self.buffer[_k]['trace'].stats.starttime
                _rtte = self.buffer[_k]['trace'].stats.endtime
                _age = self.buffer[_k]['age']
                if _rtts >= self.next_starttime and _rtte <= self.next_endtime:
                    bool_set.append(True)
                else:
                    bool_set.append(False)
            if all(bool_set):
                return True
            if any(bool_set):



        # If there isn't a valid starttime
        if self.next_starttime is None:
            # And all traces have a reasonable starttime
            if all(self.buffer[_k]['trace'].stats.starttime > UTCDateTime(0) for _k in self.buffer.keys()):
                # Get the maximum starttime as initial start time
                self.next_starttime = max([self.buffer[_k]['trace'].stats.starttime])
                self.next_endtime = self.next_starttime + self.window_sec
            # Or if any traces have a reasonable starttime
            elif any(self.buffer[_k]['trace'].stats.starttime > UTCDateTime(0) for _k in self.buffer.keys()):
                bool_set = []
                for _k in self.buffer.keys():
                    if self.buffer[_k].stats.starttime > UTCDateTime(0):
                        bool_set.append(True)
                
                    elif self.buffer[_k]['age'] >= self.age_threshold:
                        
            # If there is no usable data, return False
            else:
                return False
                
            # Check if there is enough data
        if all(self.buffer)
            
        for _k in self.buffer.keys():
            # Check that there are some data
            

            
    def _window_buffer(self):
        # If all data have reasonably fresh information
        if all(self.buffer[_k]['age'] <= self.age_threshold):
            



    def pulse(self, x):
        self._claim_traces(x)
        if self.validate_window():
            self._window_buffer()
        y = self.window_queue
        return y


    def _append_to_buffer(self, tracemsg):
        # Run compatability check
        if not isinstance(tracemsg, TraceMsg):
            raise TypeError('tracemsg must be type TraceMsg')
        else:
            pass
        # Get SNCL code
        _sncl = tracemsg.sncl
        # If sncl is not in buffer keys, create new entry
        if _sncl not in self.buffer.keys():
            self.buffer.update({_sncl: 
                {'trace': RtTrace(max_length=self.max_length),
                 'mtype': tracemsg.mtype,
                 'mcode': tracemsg.mcode,
                 'dtype': tracemsg.dtype}})
        # Otherwise, continue
        else:
            pass
        # Append trace to buffer
        self.buffer[_sncl]['trace'].append(tracemsg.to_trace())

    def _pull_from_hdeq(self, hdeq):


    def pulse(self, x):
        """
        Claim SNCL matched messages from an input HDEQ_Dict
        structured dictionary (wyrm.core.message.HDEQ_Dict)

        :: INPUT ::
        :param x: [HDEQ_Dict] structured dictionary
                    (see wyrm.core.message.HDEQ_Dict)
        
        :: OUTPUT ::
        :return y: [dict] access to self.buffer with structure:
                    {'s.n.c.l': {'trace': obspy.realtime.RtTrace,
                                 'mtype': 'TYPE_TRACEBUF2',
                                 'mcode': 19,
                                 'dtype': 'f4'}}
        """
        
        # Compatability check for x
        if not isinstance(x, dict):
            raise TypeError('Input must be type dict')
        else:
            pass
        # Get matching SNCL codes
        _sncl_list = fnmatch(x.keys(), self.sncl_search_code)
        # Iterate across matches
        for _sncl in _sncl_list:
            # Get message deque
            _queue = x[_sncl]['q']
            # Iterate through all messages in queue
            for _i in range(len(_queue)):
                _msg = _queue.pop()
                # If message is TraceMsg
                if isinstance(_msg, TraceMsg):
                    # Append to buffer
                    self._append_to_buffer(_msg)
                # Otherwise, re-append the message to the source deque
                else:
                    self.x[_sncl]['q'].append_left(_msg)
        # Provide access to the buffer as output
        y = self.buffer
        return y


class WindowWyrm(Wyrm):
    """
    This wyrm creates windowed copies of traces from a BuffWyrm and produces
    a DEQ_Dict of windowed traces keyed by Station, Network, Location, and
    the Band/Insturment characters of channel codes 
    """
    def __init__(self, search_code='GNW.UW.*.BH?', order='ZNE', window_sec=60, stride_sec=18):
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.search_code = search_code
        self.next_starttime = None
        self.next_endtime = None
        self.window_hdeq = HDEQ_Dict(extra_contents={})

    def pulse(self,x):
        """
        
        """

    def _fetch_matches(self, hdeq_dict):
        matches = fnmatch(self.search_code, buffer.keys):

        

    def _validate_next_window(self, rttrace):
        # Check that a long enough window can be extracted from target rttrace
        if rttrace.max_length < self.window_sec:
            raise ValueError('Specified window_sec for this Wyrm is longer than max_length of received RtTrace objects')
        else:
            pass

        # If this is the first check, initialize window indices
        if self.next_starttime is None:
            self.next_starttime = rttrace.stats.starttime
            self.next_endtime = self.next_starttime + self.window_sec
        else:
            pass
        
        # If next_start/endtime fall within the bounds of target rttrace
        if self.next_endtime <= rttrace.stats.endtime and self.next_starttime >= rttrace.stats.starttime:
            return True
        # If insufficient data are available
        else:
            return False
    

    def _extract_window(self, rtstream):
        # If all 
        if all(self._check_for_next_window(_x) for _x in rtstream):

            rttr = rttrace.copy().trim(starttime=self.next_starttime,
                                           endtime=self.next_endtime)
            
            
            

###############################################









class WindowWyrm(Wyrm):
    """
    This Wyrm copies striding windows off the ends of realtime traces and
    conducts pre-processing steps via obspy Trace class methods and numpy
    NDArray class methods iteratively called using x = eval('x.{method_str}') 
    statments.

    """
    def __init__(self, station, network, channels='*', locations='*', obspy_evals=[], numpy_evals=[]):
        if isinstance(obspy_evals, str):
            obspy_evals = [obspy_evals]
        if isinstance(numpy_evals, str):
            numpy_evals = [numpy_evals]
        # Check at minimum that the method looks like a method
        oelist = []
        for _e in obspy_evals:
            if isinstance(_e, str):
                if _e[0].isalpha() and '(' in _e and _e[-1] == ')':
                    oelist.append(_e)
                else:
                    raise SyntaxError
        nelist = []
        for _e in numpy_evals:
            if isinstance(_e, str):
                if _e[0].isalpha() and '(' in _e and _e[-1] == ')':
                    nelist.append(_e)
                else:
                    raise SyntaxError
        
        self.obspy_evals = oelist
        self.numpy_evals = nelist
        self.station = station
        self.network = network
        self.channels = channels
        self.locations = locations
        self.next_starttime = None
        self.next_endtime = None
        self.out_msg_buffer = {}

    def __repr__(self):
        rstr = '--- obspy evals ---\n'
        for _e in self.obspy_evals:
            rstr += f'_tr.{_e}\n'
        rstr += '--- numpy evals ---\n'
        for _e in self.numpy_evals:
            rstr += f'_X.{_e}\n'
        return rstr


    def _apply_class_method_eval(x, e):
        if isinstance(e):
            try:
                x = eval(f'x:{e}')
            except:
                raise AttributeError
        return x
    



    def pulse(self, x):
        """
        Execute the processing chain for this PrepWyrm object

        :: INPUT ::
        :param x: [dict] variable representation of a BuffWyrm.buffer
                    contents

        :: OUTPUT ::
        :return y: [dict] Dictionary of 
        """
        for _k in x.keys():
            _tr = x[_k].copy()
            if _tr.stats.endtime >= self.next_endtime:
                    
                for _e in self.obspy_evals:
                    try:
                        _tr = eval('_tr.{_e}')
                    except SyntaxError:
                        raise SyntaxError
                _X = _tr.data

                for _e in self.numpy_evals:

















class WindowWyrm(Wyrm):
    """
    
    rttraces --> WindowMsg queue

    Iterative groups pulling data off
    
    """
    def __init__(self, order='Z3N1E2',  #
                 #pp_cfg):
        # self.cfg_list = pp_cfg
        # For eventual encapsulation into cfg_dict
        
        # Indexing/Buffering Attributes
        self.queue_in = deque([])
        self.queue_out = deque([])
        if first_startt is None:
            self.next_startt = -999
            self.next_endt = -999
        elif first_startt > 0:
            self.next_startt = first_startt
            self.next_endt = self.next_startt + ml_samps/target_sr

        


    def _copy_one_window_from_rtstream(self,rtstream):
        # Check that target rtstream is composed of rttrace objects
        if all(isinstance(_rttr, RtTrace) for _rttr in rtstream):


    def _apply_cfg

    


class SNCLEarWyrm(Wyrm):
    """
    This Wyrm listens for a specific Station/Network/Channel/Location
    (SNCL) combination in offered of PyEarthworm `wave` objects. 
    Matching messages are aggregated and converted into an ObsPy Trace
    by the EarWyrm.pulse(x) method 
    """
    def __init__(self, SNCL_tuple, ):
        self.station = SNCL_tuple[0]
        self.network = SNCL_tuple[1]
        self.channel = SNCL_tuple[2]
        self.location = SNCL_tuple[3]
        self._sncl_dict = dict(
            zip(['station', 'network', 'channel', 'location'],
                [station, network, channel, location]))

    def __repr__(self):
        fstr = '~~EarWyrm~~\nListening For: '
        fstr += f'{".".join(list(self._sncl_dict.values()))}'
        return fstr

    def pulse(self, x):
        """
        This pulse(x) takes an array-like set of PyEarthworm
        `wave` messages and returns a list that has matching
        SNCL values for this particular EarWyrm

        :: INPUT :: 
        :param x: [list] List of PyEarthworm `wave` objects

        :: OUTPUT ::
        :param y: [list] List of PyEarthworm `wave` objects
                  with matching SNCL labels
        """
        # Create a holder stream for trace elements
        waves = []
        # Iterate across presented messages
        for _x in x:
            # Use the `all` iteration operator to match 
            match = all(_x[_k] == _v for _k, _v in self._sncl_dict)
            # If SNCL is a perfect match, proceed
            if match:
                waves.append(_x)
        y = waves
        return y


 
        """


class BuffWyrm(Wyrm):
    """
    This Wyrm hosts ordered ObsPy RealTime Trace (RtTrace) objects that are
    populated from incoming PyEarthworm `wave` objects, serves as a waveform 
    data buffer between pulses and provides access to RtTrace processing steps 
    via the `eval` on buffered data if they meet certain time bounds
    """
    
    def __init__(self, rtstream=None, rttrace_processing=['.'])
        if rtstream is None
            self.rtstream = Stream()

        elif isinstance(rtstrea, Stream):
            match = all(isinstance(_tr, RtTrace) for _tr in rtstream)
            if match:
                self.rtstream = rtstream
        else:
            raise TypeError

    def pulse(self, x):
        # Handle x = empty list
        if x == []:
            match = True
        # Handle instance of a slngle PyEW_Wave object as input for x
        if isinstance(x, PyEW_Wave):
            x = [x]
            match = True
        # Check entries in list of probable PyEW_Wave objects
        elif isinstance(x, (list, np.array)):
            match = all(isinstance(_x, PyEW_Wave) for _x in x)
        # Raise error in all other cases
        else:
            match = False
            raise TypeError

        if match:
            for _x in x:
                # Convert PyEW_Wave into Obspy Trace
                _tr = pew2otr(_x)
                # Append new trace to real-time trace buffer

        # Clean outdated data out of the buffer
        
        time_match = all(_tr.stats.endtime >= self.next_window_end for _tr in self.)
            


class WindowWyrm(Wyrm):




#class ELEPWyrm(MLWyrm):
# Placeholder for later development using the ensemble picking approach from ELEP (Yuan et al., 2023)

#class OctoWyrm(Wyrm):
# Placeholder for later development of a Wyrm that wraps PyOcto (Münchmeyer et al., in review)

# class ObsWyrm(BuffWyrm):
#     """
#     This Wyrm applies obspy.Trace class methods to copies of
#     rttrace objects contained in a preceding BuffWyrm through
#     iterative application of _tr = eval(f'_tr.{_eval_entry}')
#     calls.

#     E.g. 

#     eval_list = ["filter('bandpass',freqmin=1, freqmax=45)",
#                  "detrend('linear')",
#                  "resample(100)"]

#     resulting in a copy of a rttrace (_tr = rttrace.copy())
#     with the following processing:
    
#         _tr = rttrace.copy().\
#               filter('bandpass', freqmin=1, freqmax=45).\
#               detrend('linear').\
#               resample(100)

#     which is then written out to a WaveMsg
#     """
#     def __init__(self, obspy_eval_list):
#         self.buffer = {}
#         self.eval_list = obspy_eval_list

#     def __repr__(self):
#         rstr = '--- eval mockup ---\n'
#         for _e in self.eval_list:
#             rstr += 'tr = eval(_tr.{_e})\n'
    
#     def pulse(self, x):
#         """
#         Execute the processing chain for this 
#         """
#         y = {}
#         for _k in x.keys()
#             # Create copy of realtime trace from BuffWyrm
#             _tr = x[_k].copy()
#             for _e in self.eval_list:
#                 _tr = eval(f'_tr.{_e}')
#             _wavemsg = WaveMsg(_tr)
#             y.update({_k,_tr})

