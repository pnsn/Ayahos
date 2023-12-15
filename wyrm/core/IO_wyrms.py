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
from wyrm.core.pyew_msg import *

##################
### RING WYRMS ###
##################

class Wave2PyWyrm(RingWyrm):
    """
    Child class of RingWyrm, adding a pulse(x) method
    to complete the Wyrm base definition. Also adds a maximum
    iteration attribute to limit the number of wave messages
    that can be pulled in a single Wave2PyWyrm.pulse(x) call.
    """
    def __init__(self, module, conn_index, max_iter=int(1e5)):
        """
        Initialize a Wave2PyWyrm object
        :: INPUTS ::
        :param module: [PyEW.EWModule] pre-initialized module
        :param conn_index: [int] index of pre-established module
                        connection to an EW WAVE RING
        :param max_iter: [int] (default 10000) maximum number of
                        messages to receive for a single pulse(x)
                        command
        """
        super().__init__(module, conn_index)

        # Compatability check for max_iter
        try:
            self._max_iter = int(max_iter)
        except TypeError:
            print('max_iter must be int-like!')
            raise TypeError

    def __repr__(self):
        """
        Representation
        """
        rstr = super().__repr__()
        rstr += f'Max Iter: {self._max_iter}\n'
        return rstr

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
        # Provide option for passing an alternative index
        if x is not None and isinstance(x, int):
            idx = x
        else:
            idx = self.conn_index

        # Initialize a new holder
        waves = deque([])
        # Run for loop
        for _ in range(self._max_iter):
            # Get wave message
            wave = self.module.get_wave(idx)
            # Check that wave is not a no-new-messages message
            if wave != {}:
                # Convert into a PyEW_WaveMsg object
                wave = PyEW_WaveMsg(wave)
                # Check if unique to this call
                if wave not in waves:
                    # Ensure the message didn't come up last time
                    if wave not in self.msg_queue:
                        waves.append(wave)
            # Break at the first instance of a no-new-messages message
            else:
                break
        # Alias waves to y to meet standard syntax of y = *wyrm.pulse(x)
        y = waves
        # Update msg_queue buffer
        self.past_msg_queue = waves
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

class Pick2PyWyrm(RingWyrm):

class Py2PickWyrm(RingWyrm):


##################
### DISK WYRMS ###
##################

class 

##################
###   
    
class TreeWyrm(Wyrm):


class StationWyrm(Wyrm):


class ObsBuffWyrm(Wyrm):


class WindowWyrm(Wyrm):





class MessageEarWyrm(Wyrm):

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


class BookWyrm(Wyrm):
    """
    This class acts as in indexer and sorter for data arrays
    keyed by SNCL entries
    """

    def __init__(self, msg_class=PyEW_WaveMsg):
        self.msg_type = msg_class
        self.SNCL_dataframe = pd.DataFrame(columns=['Station','Network','Location','Channel'])

    def reset_SNCL_dataframe(self):
        self.SNCL_dataframe = pd.DataFrame(columns=['Station','Network','Location','Channel'])

    def append_to_SNCL_dataframe(self, x):
        for _i, _x in enumerate(x):
            if isinstance(_x, self.msg_type):
                self.SNCL_dataframe = pd.concat([self.SNCL_dataframe, pd.DataFrame(_x.SNCL_dict), index=[_i]],axis=1,ignore_index=False)

    def pulse(self, x):
        """
        :param x: [list] unsorted list of PyEW_Msg objects

        :return 
        """


class TubeWyrm(Wyrm):
    """
    Contain a linear sequence of Wyrm objects that must have 
    compatable hand-offs in data/data-type for inputs and 
    outputs of their .pulse() class methods. A TubeWyrm provides 
    a pulse(x) method that accepts an arbitrary input (x) that 
    must comply with self.wyrm_can[0].pulse(x) and will return
    the output of the sequence of wyrm.pulse(x)'s

    :: ATTRIBUTES ::
    :attrib index: [int-like] Index number for this WyrmCan
    :attrib wyrm_list: [list] List of Wyrm* objects that have
                compatable sequential input/output data from
                their versions of the .pulse() class method
    :attrib cfg: [dict] Dictionary to hold configuration information
                NOTE: Presently not used

    :: CLASS METHODS ::
    :method __init__:
    :method pulse:
    """
    def __init__(self, index, wyrm_list, cfg=None):
        """
        Initialize a TubeWyrm object with the following
        input parameters
        :: INPUTS ::
        :param index: [int-like] Index number for this WyrmCan TODO: OBSOLITE
        :param wyrm_list: [list] List of Wyrm* objects that have
                    compatable sequential input/output data from
                    their versions of the .pulse() class method
        :param cfg: [dict] Dictionary to hold configuration information
                    NOTE: Presently not used

        :: OUTPUT ::
        None
        """
        # Type handling for index
        if isinstance(index, (int, np.int32, np.int64)):
            self.index = index
        elif isinstance(index, (float, np.float32, np.float64)):
            self.index = int(index)
        else:
            raise TypeError
        
        # Type checking for all members of Wyrm
        if isinstance(wyrm_list, list):
            # Check that all items in wyrm_list are a child of Wyrm
            match = all(isinstance(_wyrm, Wyrm) for _wyrm in wyrm_list)
            if match:
                self.wyrm_list = wyrm_list
            # Raise TypeError otherwise
            else:
                raise TypeError
        # Handle the case where the WyrmCan is given a single Wyrm object
        elif isinstance(wyrm_list, Wyrm):
            self.wyrm_list = [wyrm_list]
        # Otherwise, raise TypeError
        else:
            raise TypeError

        self.cfg = cfg
        return None

    def __repr__(self):
        fmsg = 'TubeWyrm with\n'
        fmsg += f'Index: {self.index}\n'
        fmsg += 'WyrmList:\n'
        for _wyrm in self.wyrm_list:
            fmsg += f'  {type(_wyrm)}\n'
        fmsg += f'cfg: {self.cfg}'
        return fmsg

    def pulse(self, x):
        """
        Run .pulse() for each member of self.wyrm_list in sequence,
        passing the output of the i_th *wyrm.pulse(x) = y to the input
        to the input for the i+1_th *wyrm.pulse(x = y).

        :: INPUT ::
        :param x: Input for *wyrm.pulse(x) for the first member of
                  self.wyrm_list
        :: OUTPUT ::
        :return y: [dict] output of the last *wyrm.pulse(x) in 
                  self.wyrm_list with an associated key self.index
                  from this WyrmCan object
        """
        for _wyrm in self.wyrm_list:
            x = _wyrm.pulse(x)
        y = {self.index: x}
        return y



###############################
### DATA PROCESSING CLASSES ###
###############################

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

##########################################
### ML PROCESSING (GRAND)CHILD CLASSES ###
##########################################

class StreamMLWyrm(MLWyrm):
    """
    ML Prediction module for models where the input (data) and output (pred) arrays
    consist of windowed time-series with associated metadata
    """
    def __init__(self, model, device, ml_input_shape, window_axis, sample_axis, channel_axis):
        super().__init__(model, device)
        self.ml_input_shape = ml_input_shape
        self.window_axis = window_axis
        self.sample_axis = sample_axis
        self.channel_axis = channel_axis
    
    def __repr__(self):
        rstr = f'Input Dimensions: {self.ml_input_shape}\n'
        rstr += f'Window Axis: {self.window_axis}\nSample Axis: {self.sample_axis}\nChannel Axis: {self.channel_axis}\n'
        rstr += super().__repr__()

        return rstr
    
    def _preprocess_data(self, x, )


#class ELEPWyrm(MLWyrm):
# Placeholder for later development using the ensemble picking approach from ELEP (Yuan et al., 2023)

#class OctoWyrm(Wyrm):
# Placeholder for later development of a Wyrm that wraps PyOcto (Münchmeyer et al., in review)

