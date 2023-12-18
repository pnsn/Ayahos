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
from obspy import Trace, UTCDateTime, Stream


class PEWMsg:
    """
    SNCL keyed PyEarthworm Message Base Class

    Provides additional attributes and supports
    data validation, sorting, and message I/O on
    top of the dictionaries provided in PyEarthworm
    as the standard Type for messages
    """

    def __init__(self, station=None, network=None, channel=None, location=None):
        # Compatability check & formatting for station
        if station is None:
            self.station = ""
        elif type(station) is bool:
            raise TypeError
        elif isinstance(station, (str, int)):
            self.station = str(station)
        else:
            raise TypeError
        # Compatability check & formatting for network
        if type(network) is bool:
            raise TypeError
        elif isinstance(network, (str, int)):
            self.network = str(network)
        elif network is None:
            self.network = ""
        else:
            raise TypeError
        # Compatability check & formatting for channel
        if type(channel) is bool:
            raise TypeError
        elif isinstance(channel, (str, int)):
            self.channel = str(channel)
        elif channel is None:
            self.channel = ""
        else:
            raise TypeError
        # Compatability check & formatting for location
        if type(location) is bool:
            raise TypeError
        elif isinstance(location, (str, type(None))):
            # Handle special case where location is an empty or white-space only
            if location is None or all(x == " " for x in location):
                self.location = "--"
            else:
                self.location = str(location)
        elif isinstance(location, int):
            self.location = str(location)
        else:
            raise TypeError

        # Passing all checks, compose a SNCL code string attribute
        self.code = f"{self.station}.{self.network}.{self.channel}.{self.location}"

    def __repr__(self):
        rstr = self.code
        return rstr

    # datatype conversion methods for inheritence purposes

    def ew2np_dtype(self, dtype):
        """
        Convert from a specified Earthworm/C dtype string into
        a numpy.dtype object

        :: INPUT ::
        :param dtype: [str] input Earthworm/C datatype string

        :: OUTPUT ::
        :return ew_dtype: [numpy.dtype] - numpy dtype object
        """
        dtypes_dict = {"i2": np.int16, "i4": np.int32, "i8": np.int64, "s4": np.float32}

        if isinstance(dtype, str):
            if dtype not in list(dtypes_dict.keys()):
                raise KeyError
            elif dtype in list(dtypes_dict.keys()):
                return dtypes_dict[dtype]
            else:
                raise TypeError
        else:
            raise TypeError

    def np2ew_dtype(self, dtype):
        """
        Convert from a specified numpy.dtype object into a mapped Earthworm/C
        dtype string

        :: INTPUT ::
        :param ew_dtype: [numpy.dtype] - numpy dtype object

        :: OUTPUT ::
        :return dtype: [str] input Earthworm/C datatype string

        """
        dtypes_dict = {np.int16: "i2", np.int32: "i4", np.int64: "i8", np.float32: "s4"}

        if isinstance(dtype, type):
            if dtype in list(dtypes_dict.values()):
                key = list(dtypes_dict.keys())[list(dtypes_dict.values()) == dtype]
            else:
                raise ValueError


# class WaveMsg(PEWMsg):
#     """
#     Formalized class definition of the `wave` object in PyEarthworm used to
#     describe tracebuff2 messages in the Python-side of the module.

#     This class is used to carry non-masked numpy data arrays representing
#     individual, continuous, evenly sampled time series within the Python-side
#     environment.

#     NOTE:
#     Any transmission of gappy (i.e., masked numpy arrays) or multi-channel
#     (multi-dimensional numpy.ndarrays) time series data should be handled
#     with the obspy.Trace and obspy.Stream objects.

#     The following methods provide easy interface with the obspy API:
#         WaveMsg.to_trace()
#         WaveMsg.from_trace()

#     The following methods provide easy conversion to/from the PyEarthworm
#     `wave` object
#     """
#     def __init__(self, wave, default_dtype='s4'):
#         """
#         Create a WaveMsg object
#         """
#         if not isinstance(wave, (dict,Trace, Stream)):
#             raise TypeError
#         elif isinstance(wave, Stream):
#             if len(wave) > 1:
#                 print('Too many entries in stream')
#                 raise IndexError
#             elif len()


#     def __init__(self, station=None, network=None, channel=None, location=None, startt=None, samprate=1, data=np.array([]), datatype='i4'):
#         """
#         Create a PyEW_WaveMsg object

#         :: INPUTS ATTRIBUTES ::
#         :param station: [str] station code
#         :param network: [str] network code
#         :param channel: [str] channel code
#         :param location: [str] location code
#         :param startt: [float] epoch start time [sec since 1970-01-01]
#         :param samprate: [int-like] sampling rate in samples per second
#         :param data: [numpy.ndarray] data vector
#         :param datatype: [str] Earthworm data type code
#                     'i2' = 16 bit integer
#                     'i4' = 32 bit integer
#                     'i8' = 64 bit integer
#                     's4' = 32 bit signed integer

#         :: INIT GENERATED ATTRIBUTES ::
#         :attrib nsamp: [int] number of data
#         :attrib endt: [float] epoch end time for data [sec since 1970-01-01]
#         :attrib code: [str] SNCL code
#         """
#         super().__init__(station, network, channel, location)

#         if startt is None:
#             self.startt = -999
#         elif str(startt).isnumeric():
#             if np.isfinite(startt):
#                 self.startt = float(startt)
#         if samprate is None:
#             self.samprate = 1
#         elif not str(samprate).isnumeric():
#             self.samprate = 1.
#         else:
#             self.samprate = float(samprate)

#         self.datatype = datatype
#         self.dataclass = self.ew2np_dtype(datatype)
#         self.data = data.astype(self.dataclass)
#         self.nsamp = len(self.data)
#         self.datatype = datatype

#     def __repr__(self):
#         rstr = super().__repr__()
#         rstr += f" | {self.nsamp} samples | {self.samprate:.1f} Hz | {self.startt:.3f}\n"
#         return rstr

#     def to_trace(self):
#         """
#         Return an obspy.core.trace.Trace copy of the WaveMessage
#         without altering the initial message data
#         """
#         # Initialize empty trace
#         trace = Trace()
#         # Add SNCL information
#         trace.stats.station = self.station
#         trace.stats.network = self.network
#         trace.stats.channel = self.channel
#         trace.stats.location = self.location
#         # Add timing
#         trace.stats.starttime = UTCDateTime(self.startt)
#         trace.stats.sampling_rate = self.samprate
#         # Add data
#         _dtype = self.ew2np_dtype(self.datatype)
#         _data = deepcopy(self.data).astype(_dtype)
#         trace.data = _data
#         # Return completed trace
#         return trace

#     def to_tracebuff(self):
#         """
#         Return a copy of data contained in this message in the PyEarthworm
#         TraceBuff dictionary format.
#         """
#         keys = ['station',
#                 'network',
#                 'channel',
#                 'location',
#                 'nsamp',
#                 'samprate',
#                 'startt',
#                 'datatype',
#                 'data'
#                ]
#         # Enforce appropriate datatype on data
#         _dtype = self.ew2np_dtype(self.datatype)
#         values = [self.station,
#                   self.network,
#                   self.channel,
#                   self.location,
#                   self.nsamp,
#                   self.samprate,
#                   self.startt,
#                   self.datatype,
#                   self.data.astype(_dtype)
#                  ]
#         tracebuff_msg = dict(zip(keys, values))
#         return tracebuff_msg


# def WindMessage(data)

# # class PyEW_PickMsg(PyEW_Msg):
