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
import PyEW
from collections import deque
from wyrm.core.base_wyrms import Wyrm, RingWyrm
import pandas as pd
import seisbench.models as sbm
from obspy import UTCDateTime, Stream, Trace
from obspy.realtime import RtTrace
from wyrm.core.message import WaveMsg
import fnmatch


class BuffWyrm(Wyrm):
    """
    This Wyrm handles WaveMsg data buffering in the form
    of a dictionary of rttrace objects

    {SNCL: (deque, sindex)} --> {SNCL: rttrace}
    """

    def __init__(self, station, network, channels='*', locations='*', max_buff_sec=300, rt_proc_steps=None):
        # Static SNCL fields (one instance per site)
        self.station = station
        self.network = network
        self.channels = channels
        self.locations = locations
        self.max_buff_sec = max_buff_sec
        self.sncl_search_code = f'{self.station}.{self.network}.{self.channels}.{self.locations}'
        self.buffer = {}

    def __resp__(self):
        rstr = f'Search String: {self.sncl_search_code}\n'
        rstr += f'Max Buff: {self.max_buff_sec:.3f} sec\n' 
        rstr += '--- Buffered RtTraces ---\n'
        for _k in self.buffer.keys():
            rstr += f'{self.buffer[_k]}\n'
        return rstr
         
    def _wavemsg2buffer(self, wavemsg):
        """
        Convert an input WaveMsg into an obspy trace and add
        it to the buffer attribute of this BuffWyrm, creating
        and populating a new RtTrace object if the SNCL code
        of the WaveMsg is not a key in BuffWyrm.buffer.keys()
        or appending the WaveMsg data to a matching SNCL entry.

        :: INPUT ::
        :param wavemsg: [wyrm.core.pyew_msg.WaveMsg] wave message
                        object that contains at least 1 data point

        :: UPDATE ::
        :attrib buffer[wavemsg.code]:

        :: OUTPUT ::
        :return None:

        """

        # Check that the wavemsg is WaveMsg and has some data
        if isinstance(wavemsg, WaveMsg):
            if wavemsg.npts > 0:
                # Use WaveMsg class method to convert to obspy.Trace
                _tr = wavemsg.to_trace()
                # If SNCL exists in buffer, append data to associated rttrace object
                if wavemsg.code in self.buffer.keys():
                    self.buffer[wavemsg.code].append(_tr)
                # If SNCL is new to this buffer, create a new buffer key:value entry
                if wavemsg.code not in self.buffer.keys():
                    new_rttrace = RtTrace(max_length=self.max_buff_sec)
                    new_rttrace.append(_tr)    
                    self.buffer.update({wavemsg.code:new_rttrace})
                else:
                    raise KeyError
            elif wavemsg.npts == 0:
                pass
            else:
                print('wavemsg.npts is negative...')
                raise ValueError
        else:
            print(f'wavemsg is type {type(wavemsg)} --> must be type WaveMsg')
            raise TypeError
        
    def _pull_from_Wave2PyWyrm_queue_dict(self, queue_dict):
        """
        Iteratively pop WaveMsg objects out of the queue_dict
        attribute of a Wave2PyWyrm object that have matching
        SNCL codes that this particular BuffWyrm is seeking

        :: INPUT ::
        :param queue_dict: [dict] accessible variable version of
                            the data cointained in a Wave2PyWyrm
                            object's `queue_dict` attribute.
        
        :: UPDATES ::
        :Wave2PyWyrm.queue_dict[SNCL]: WaveMsg entries are shifted
                            to this object, converted into obspy.Trace
                            objects and...
        :BuffWyrm.buffer[SNCL]: ...appended to the corresponding RtTrace
                            object in this object. Idea is to minimize
                            copying/doubling of data in memory
        
        :: OUTPUT ::
        :return None:
        """
        # Find matching SNCL entries in queue_dict
        matches = fnmatch.filter(queue_dict.keys(), self.sncl_search_code)
        # Iterate across SNCL codes
        for _sncl in matches:
            # alias queue_dict from Wave2PyWyrm object
            in_queue = queue_dict[_sncl][0]
            # Until the queue_dict deque is empty...
            _safety_catch = 10000
            while len(in_queue) > 0:
                # ...pop out oldest message...
                _msg = in_queue.pop()
                # ...and add it to the buffer
                self._wavemsg2buffer(_msg)

                # DEV ELEMENT TO SAFEGUARD AGAINS UNBOUNDED `while`
                _safety_catch -= 1
                if _safety_catch == 0:
                    print('BUFFWYRM - HIT SAFETY CATCH ON DATA TRANSFER!!!')
                    break

    def pulse(self, x):
        """
        Execute the processing chain for this BuffWyrm object

        :: INPUT ::
        :param x: [dict] variable representation of a preceding
                Wave2PyWyrm.queue_dict 

        :: OUTPUT ::
        :return y: [dict] variable representation of this 
                object's BuffWyrm.buffer 
        """
        self._pull_from_Wave2PyWyrm_queue_dict(x)
        y = self.buffer
        return y




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

