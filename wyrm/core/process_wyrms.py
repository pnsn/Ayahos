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

