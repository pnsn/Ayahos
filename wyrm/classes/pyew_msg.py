"""
:module: wyrm.classes.pyew_msg
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This module contains class definitions for Python-side earthworm messages
    based on the `wave` and `pick` objects from PyEarthworm (C) F. Hernandez, 2018
    that are expanded to facilitate expedited sorting and processing within an 
    operating Wyrm module

:attribution:
    This module is based on the python dict formatting of `wave` (tracebuff2) messages
    and str formatting of `pick` (pick2k) messages from PyEarthworm developed by
    F. Hernandez and are used in compliance with their AGPL-3.0 license for PyEarthworm

"""
import numpy as np
from copy import deepcopy
from obspy import Trace, UTCDateTime

class PyEW_Msg:
    """
    PyEarthworm Message Base Class

    This BaseClass Does Nothing on its own

    Provides additional attributes and supports
    data validation, sorting, and message I/O on 
    top of the dictionaries provided in PyEarthworm
    as the standard Type for messages
    """

    def __init__(self):
        return None
    
    def __repr__(self):
        rstr = 'PyEW_Msg\nBaseClass for PyEarthworm Messages\n'
        return rstr
    

class PyEW_WaveMsg(PyEW_Msg):
    """
    Formalized class definition of the `wave` object in PyEarthworm used to 
    describe tracebuff2 messages in the Python-side of the module.
    
    """
    def __init__(self, station=None, network=None, channel=None, location=None, startt=None, samprate=1, data=np.array([]), datatype='i4'):
        """
        Create a PyEW_WaveMsg object

        :: INPUTS ATTRIBUTES ::
        :param station: [str] station code
        :param network: [str] network code
        :param channel: [str] channel code
        :param location: [str] location code
        :param startt: [float] epoch start time [sec since 1970-01-01]
        :param samprate: [int-like] sampling rate in samples per second
        :param data: [numpy.ndarray] data vector
        :param datatype: [str] Earthworm data type code
                    'i2' = 16 bit integer
                    'i4' = 32 bit integer
                    'i8' = 64 bit integer
                    's4' = 32 bit signed integer
        :attrib nsamp: [int] number of data
        :attrib endt: [float] epoch end time for data [sec since 1970-01-01]
        :attrib SNCL_dict: [dict] dictionary summarizing SNCL codes, used for matching/indexing
           
        """
        self.station = station
        self.network = network
        self.channel = channel
        self.location = location
        self.startt = startt
        self.samprate = samprate
        self.datatype = datatype
        self.dataclass = {"i2": np.int16, "i4": np.int32, "i8": np.int64, "s4": np.int32}[datatype]
        self.data = data.astype(self.dataclass)
        self.nsamp = len(self.data)
        self.datatype = datatype
        self.endt = self.startt + self.nsamp/self.samprate
        self.SNCL_dict = dict(
            zip(['station', 'network', 'channel', 'location'],
                [station, network, channel, location]))

    def __repr__(self):
        rstr = 'PyEW_WaveMsg\n'
        rstr += f"{'.'.join(list(self.SNCL_dict.values()))}"
        rstr += f" | {self.nsamp} samples | {self.samprate:.1f} Hz | {self.startt:.3f}\n"
        return rstr
    
    def to_trace(self):
        """
        Return an obspy.core.trace.Trace copy of the WaveMessage
        without altering the initial message data
        """
        # Initialize empty trace
        trace = Trace()
        # Add SNCL information
        trace.stats.station = self.station
        trace.stats.network = self.network
        trace.stats.channel = self.channel
        trace.stats.location = self.location
        # Add timing
        trace.stats.starttime = UTCDateTime(self.startt)
        trace.stats.sampling_rate = self.samprate
        # Add data
        _dtype = {"i2": np.int16, "i4": np.int32, "i8": np.int64, "s4": np.int32}[self.datatype]
        _data = deepcopy(self.data).astype(_dtype)
        trace.data = _data
        # Return completed trace
        return trace

    def to_tracebuff(self):
        """
        Return a copy of data contained in this message in the PyEarthworm
        TraceBuff dictionary format.
        """
        keys = ['station',
                'network',
                'channel',
                'location',
                'nsamp',
                'samprate',
                'startt',
                'datatype',
                'data'
               ]
        # Enforce appropriate datatype on data
        _dtype = {"i2": np.int16, "i4": np.int32, "i8": np.int64, "s4": np.int32}[self.datatype]
        values = [self.station,
                  self.network,
                  self.channel,
                  self.location,
                  self.nsamp,
                  self.samprate,
                  self.startt,
                  self.datatype, 
                  self.data.astype(_dtype)
                 ]
        tracebuff_msg = dict(zip(keys, values))
        return tracebuff_msg


# class PyEW_PickMsg(PyEW_Msg):
