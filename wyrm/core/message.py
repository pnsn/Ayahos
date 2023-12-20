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

:earthworm_global.d Information on Messages:

--- Starting on Line 245 of earthworm_global.d ---
#--------------------------------------------------------------------------
#                          Message Types
#
#  Define all message name/message-type pairs that will be global
#  to all Earthworm systems.
#
#  VALID numbers are:
#
#   0- 99 Message types 0-99 are defined in the file earthworm_global.d.
#         These numbers are reserved by Earthworm Central to label types
#         of messages which may be exchanged between installations. These
#         string/value mappings must be global to all Earthworm systems
#         in order for exchanged messages to be properly interpreted.
#
#
#  OFF-LIMITS numbers:
#
# 100-255 Message types 100-255 are defined in each installation's
#         earthworm.d file, under the control of each Earthworm
#         installation. These values should be used to label messages
#         which remain internal to an Earthworm system or installation.
#
#
# The maximum length of the type string is 32 characters.
#
#--------------------------------------------------------------------------

# Global Earthworm message-type mappings (0-99):
 Message  TYPE_WILDCARD          0  # wildcard value - DO NOT CHANGE!!!
 Message  TYPE_ADBUF             1  # multiplexed waveforms from DOS adsend
 Message  TYPE_ERROR             2  # error message
 Message  TYPE_HEARTBEAT         3  # heartbeat message
 Message  TYPE_TRACE2_COMP_UA    4  # compressed waveforms from compress_UA, with SCNL
#Message  TYPE_NANOBUF           5  # single-channel waveforms from nanometrics
 Message  TYPE_ACK               6  # acknowledgment sent by import to export
 Message  TYPE_PICK_SCNL         8  # P-wave arrival time (with location code)
 Message  TYPE_CODA_SCNL         9  # coda info (plus station/loc code) from pick_ew
 Message  TYPE_PICK2K           10  # P-wave arrival time (with 4 digit year)
                                    #   from pick_ew
 Message  TYPE_CODA2K           11  # coda info (plus station code) from pick_ew
#Message  TYPE_PICK2            12  # P-wave arrival time from picker & pick_ew
#Message  TYPE_CODA2            13  # coda info from picker & pick_ew
 Message  TYPE_HYP2000ARC       14  # hyp2000 (Y2K hypoinverse) event archive
                                    #   msg from eqproc/eqprelim
 Message  TYPE_H71SUM2K         15  # hypo71-format hypocenter summary msg
                                    #   (with 4-digit year) from eqproc/eqprelim
#Message  TYPE_HINVARC          17  # hypoinverse event archive msg from
                                    #   eqproc/eqprelim
#Message  TYPE_H71SUM           18  # hypo71-format summary msg from
                                    #   eqproc/eqprelim
 Message  TYPE_TRACEBUF2        19  # single-channel waveforms with channels
                                    #   identified with sta,comp,net,loc (SCNL)
 Message  TYPE_TRACEBUF         20  # single-channel waveforms from NT adsend,
                                    #   getdst2, nano2trace, rcv_ew, import_ida...
 Message  TYPE_LPTRIG           21  # single-channel long-period trigger from
                                    #   lptrig & evanstrig
 Message  TYPE_CUBIC            22  # cubic-format summary msg from cubic_msg
 Message  TYPE_CARLSTATRIG      23  # single-channel trigger from carlstatrig
#Message  TYPE_TRIGLIST         24  # trigger-list msg (used by tracesave modules)
                                    #   from arc2trig, trg_assoc, carlsubtrig
 Message  TYPE_TRIGLIST2K       25  # trigger-list msg (with 4-digit year) used
                                    #   by tracesave modules from arc2trig,
                                    #   trg_assoc, carlsubtrig
 Message  TYPE_TRACE_COMP_UA    26  # compressed waveforms from compress_UA
#Message  TYPE_STRONGMOTION     27  # single-instrument peak accel, peak velocity,
                                    #   peak displacement, spectral acceleration
 Message  TYPE_MAGNITUDE        28  # event magnitude: summary plus station info
 Message  TYPE_STRONGMOTIONII   29  # event strong motion parameters
 Message  TYPE_LOC_GLOBAL       30  # Global location message used by NEIC & localmag
 Message  TYPE_LPTRIG_SCNL      31  # single-channel long-period trigger from
                                    #   lptrig & evanstrig (with location code)
 Message  TYPE_CARLSTATRIG_SCNL 32  # single-channel trigger from carlstatrig (with loc)
 Message  TYPE_TRIGLIST_SCNL    33  # trigger-list msg (with 4-digit year) used
                                    #   by tracesave modules from arc2trig,
                                    #   trg_assoc, carlsubtrig (with location code)
 Message  TYPE_TD_AMP           34  # time-domain reduced-rate amplitude summary
                                    #   produced by CISN RAD software & ada2ring
 Message  TYPE_MSEED            35  # Miniseed data record
 Message  TYPE_NOMAGNITUDE      36  # no event magnitude generated by localmag

 Message  TYPE_NAMED_EVENT      94  # TWC message for Windows compat
 Message  TYPE_HYPOTWC          95  # ATWC message
 Message  TYPE_PICK_GLOBAL      96  # ATWC message
 Message  TYPE_PICKTWC          97  # ATWC message
 Message  TYPE_ALARM            98  # ATWC message

#      !!!!   DO NOT make any changes to this file.  !!!!
---END OF FILE---

"""
import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.realtime import RtTrace
import torch

# CREATE GLOBAL VARIABLES FOR MESSAGE TYPES FOR NOW..
# TODO: HAVE SOME FUNCTIONALITY TO CROSS-REFERENCE WITH earthworm_global.d AND
#       INSTALLATION SPECIFIC MESSAGE CODE (earthworm_local.d?) BEFORE STARTUP
EW_GLOBAL_MESSAGE_CODES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11,\
                           12, 13, 14, 15, 17, 18, 19, 20, 21, 22,\
                           23, 24, 25, 26, 27, 28, 29, 30, 31, 32,\
                           33, 34, 35, 36, 94, 95, 96, 97, 98]

EW_GLOBAL_MESSAGE_TYPES = ['TYPE_WILDCARD','TYPE_ADBUF','TYPE_ERROR','TYPE_HEARTBEAT',\
                           'TYPE_TRACE2_COMP_UA','TYPE_NANOBUF','TYPE_ACK','TYPE_PICK_SCNL',\
                           'TYPE_CODA_SCNL','TYPE_PICK2K','TYPE_CODA2K','TYPE_PICK2','TYPE_CODA2','TYPE_HYP2000ARC',\
                           'TYPE_H71SUM2K','TYPE_HINBARC','TYPE_H71SUM','TYPE_TRACEBUF2',\
                           'TYPE_TRACEBUF','TYPE_LPTRIG','TYPE_CUBIC','TYPE_CARLSTATRIG',\
                           'TYPE_TRIGLIST','TYPE_TRIGLIST2K','TYPE_TRACE_COMP_UA','TYPE_STRONGMOTION',\
                           'TYPE_MAGNITUDE','TYPE_STRONGMOTIONII','TYPE_LOC_GLOBAL','TYPE_LPTRIG_SCNL',\
                           'TYPE_CARLSTATRIG_SCNL','TYPE_TRIGLIST_SCNL',
                           'TYPE_TD_AMP','TYPE_MSEED','TYPE_NOMAGNITUDE','TYPE_NAMED_EVENT',\
                           'TYPE_HYPOTWC','TYPE_PICK_GLOBAL','TYPE_PICKTWC','TYPE_ALARM']

EW_GLOBAL_DICT = dict(zip(EW_GLOBAL_MESSAGE_TYPES, EW_GLOBAL_MESSAGE_CODES))

# NOTE: I've seen both 's4' and 'f4' show up in PyEarthworm documentation. Include both for redundance
NPDTYPES = [np.int16, np.int32, np.float32, np.float32]
EWDTYPES = ['i2','i4','s4','f4']
NP2EWDTYPES = dict(zip(NPDTYPES, EWDTYPES))
EW2NPDTYPES = dict(zip(EWDTYPES, NPDTYPES))


class _BaseMsg:
    """
    Message fundamental BaseClass to provide richer descriptions and handling of PyEW message
    objects in the Python-side
    """
    def __init__(self, mtype='TYPE_TRACEBUF2', mcode=19):
        # Compatability check message-type definition
        if isinstance(mtype, str):
            if mtype.upper() == mtype:
                self._mtype = mtype
            else:
                raise SyntaxError('"mtype" must be all uppercase to match Earthworm syntax')
        else:
            raise TypeError('"mtype" must be a string')
        if isinstance(mcode, (int, float)):
            if 0 <= int(mcode) <= 255:
                self._mcode = int(mcode)
            else:
                raise SyntaxError('"mcode" value outside [0-255] is illegal')
        else:
            raise TypeError('"code" must be an int-like number')
        # Validate type:code combination
        # Cross reference with earthworm_global.d
        if 0 <= self._mcode < 100:
            # First check if the code is a valid value
            if self._mcode not in EW_GLOBAL_MESSAGE_CODES:
                raise SyntaxError('"mcode" value is not defined in earthworm_global.d')
            if self._mtype not in EW_GLOBAL_MESSAGE_TYPES:
                raise SyntaxError('"mtype" value is not defined in earthworm_global.d')
            # Then see if it matches with the TYPE_* name
            if self._mcode in EW_GLOBAL_MESSAGE_CODES and self._mtype in EW_GLOBAL_MESSAGE_TYPES:
                if EW_GLOBAL_DICT[self._mtype] == self._mcode:
                    pass
                else:
                    raise SyntaxError('earthworm_global.d Type:Code mismatch!')

        ## TODO: Eventually include a way to read installation-specific message formats for validation
        else:
            Warning(f'"mcode" value {self._mcode} falls into the Installation-specific range of message codes. Proceed at your own risk')
        

        
        
    def __repr__(self):
        rstr = f'MTYPE: {self._mtype}\n'
        rstr += f'MCODE: {self._mcode}\n'
        return rstr


class _SNCLMsg(_BaseMsg):
    """
    SNCL keyed message
    :: ATTRIBUTES ::
    :attrib _mtype: [str] EW message TYPE (_BaseMsg super())
    :attrib _mcode: [int] EW message code (_BaseMsg super())
    :attrib station: [4-string] station code
    :attrib network: [2-string] network code
    :attrib channel: [3-string] channel code
    :attrib location: [2-string] location code. No code = '--'
    :attrib sncl: [string] station.network.channel.location code
    """
    def __init__(self, station=None, network=None, channel=None, location=None, mtype='TYPE_TRACEBUF2', mcode=19):
        super().__init__(mtype=mtype, mcode=mcode)
        if station is None:
            self.station = ''
        elif isinstance(station, (str, int)):
            self.station = str(station)
        else:
            raise TypeError
        if network is None:
            self.network = ''
        elif isinstance(network, (int, str)):
            self.network = str(network)
        else:
            raise TypeError
        if channel is None:
            self.channel = ''
        elif isinstance(channel, (int, str)):
            self.channel = str(channel)
        else:
            raise TypeError
        if location is None:
            self.location = '--'
        elif isinstance(location, (int, float)):
            if len(str(int(location))) > 2:
                print(f'location code is too long {location}, truncating to the trailing 2 integers')
                self.location = str(int(location))[-2:]
            else:
                self.location = f'{int(location):02d}'
        elif isinstance(location, str):
            if len(location) == 2:
                self.location = location
            if len(location) == 1 and location != ' ':
                self.location = '0'+location
        
        if len(self.station) > 4:
            self.station = self.station[:4]
        if len(self.network) > 2:
            self.network = self.network[:2]
        if len(self.channel) > 3:
            self.channel = self.channel[:3]
        if len(self.location) > 2:
            self.location = self.location[:2]
        if self.location in ['',' ','  ']:
            self.location='--'

        self.sncl = f'{self.station}.{self.network}.{self.channel}.{self.location}'
    
    def __repr__(self):
        rstr = super().__repr__()
        rstr += self.sncl
        return rstr


class WaveMsg(_SNCLMsg):
    """
    Message Class Built on top of the TYPE_TRACEBUF2 Earthworm message type
    and the PyEarthworm EWModule.get_wave() / .put_wave() syntax to streamline
    handling of 1-C waveform data and metadata between Python and Earthworm
    memory rings.

    This class provides attributes for ingesting traces with gaps (i.e.,
    trace.data as MaskedArray's) and options for altering the fill_value
    prior to generating a `wave` dictionary for use with .get_/.put_wave() 

    :: ATTRIBUTES ::
    :attrib _mtype: [str] TYPE_TRACEBUF2 (super from _Base_Msg)
    :attrib _mcode: [int] 19 (super from _Base_Msg) 
    :attrib station: [4-string] station code
    :attrib network: [2-string] network code
    :attrib channel: [3-string] band/instrument/component SEED code (channel code)
    :attrib location: [2-string] location code (no-code = '--')
    :attrib nsamp: [int] number of samples
    :attrib samprate: [numpy.float32] sampling rate in Hz (samples per second)
    :attrib startt: [np.float32] epoch start time (seconds since 1970-01-01:00:00:00)
    :attrib endt: [np.float32] epoch end time (seconds since 1970-01-01:00:00:00)
    :attrib datatype: [str] Earthworm / C data-type name
    :attrib data: [(n, ) numpy.ndarray] data
    :attrib mask_array: [None] or [(n, ) numpy.ndarray of bool] Bool mask for self.data
    :attrib fill_value: [self.datatype] value to fill entries in self.data[self.mask_array]
                    when exporting Trace or `wave` message representations of this message
    :attrib torchtensorflag: [bool] Are the data in this msg actually a flattened representation
                    of a 2-dimensional tensor?

    """

    def __init__(self, input=None):
        # Initialize baseclass defs with hard-set for tracebuf2   
        super().__init__(mtype='TYPE_TRACEBUF2', mcode=19)      
        self.nsamp = np.int32(0)
        self.samprate = np.float32(1.)
        self.startt = np.float32(0.)
        self.endt = np.float32(0.)
        self.datatype = 's4'
        self.data = np.array([], dtype=EW2NPDTYPES[self.datatype])
        self.mask_array = np.array([], dtype=bool)
        self.fill_value = 0
        # TODO: Move this to separate class
        # self.torchtensorflag = False

        # If input is None-type, return an empty WaveMsg object with the above defaults
        if input is None:
            pass
        # Otherwise, run compatability checks for trace-like inputs
        elif isinstance(input, (Trace, RtTrace)):
            self._trace2msg(input)
        elif isinstance(input, Stream):
            if len(input) == 1:
                if isinstance(input[0], (Trace, RtTrace)):
                    self._trace2msg(input[0])
                else:
                    raise TypeError(f'First entry of Stream "input" is not type Trace or RtTrace!')
            else:
                raise TypeError(f'"input" of type obspy.Stream must only contain 1 trace. This contains {len(input)} elements')
        
        # Handle PyEW `wave` dictionary objects
        elif isinstance(input, dict):
            if self._validate_wave(input):
                self._wave2msg(input)
        else:
            raise TypeError('"input" type is invalid. Accepted classes: None-type, obspy Trace (Trace, RtTrace), PyEW wave (dict)')

    def __repr__(self):
        rstr = super().__repr__()
        rstr += f' | {self.startt:.3f} - {self.endt:.3f} | '
        rstr += f'{self.samprate} Hz | {self.nsamp} samples | {self.datatype}\n'
        return rstr

    def _trace2msg(self, trace):
        """
        PRIVATE METHOD

        Supporting method for ingesting obspy Trace-like objects
        that may contain gaps

        :: INPUT ::
        :param trace: [obspy.Trace] or [obspy.realtime.RtTrace]

        :: UPDATE ::
        :attrib msg_data_mask: Update if input contains a MaskedArray with 
                    the mask boolean vector from trace.data.mask
        :attrib fill_value: Update if input contains a Masked array with the
                    default fill value from trace.data.fill_value
        :attrib msg: 
        """
        if not isinstance(trace, (Trace, RtTrace)):
            raise TypeError('"trace" input is an invalid class. Accepted classes: obspy.Trace, obspy.realtime.RtTrace')
        else:
            # Apply max string length truncations and loc special formatting
            sta = trace.stats.station
            if len(sta) > 4:
                sta = sta[:4]
            net = trace.stats.network
            if len(net) > 2:
                net = net[:2]
            cha = trace.stats.channel
            if len(cha) > 3:
                cha = cha[:3]
            loc = trace.stats.location
            if len(loc) > 2:
                loc = loc[:2]
            if loc in ['',' ','  ']:
                loc = '--'
            # Bring in SNCL info into WaveMsg
            self.station = sta
            self.network = net
            self.channel = cha
            self.location = loc
            self.sncl = f'{sta}.{net}.{cha}.{loc}'
            # Get sampling/timing information
            self.nsamp = int(trace.stats.npts)
            self.samprate = trace.stats.sampling_rate
            self.startt = trace.stats.starttime.timestamp
            self.endt = trace.stats.endtime.timestamp

            # Handle potential input data formatting scenarios
            _data = trace.data
            # Check that _data is a numpy.ndarray
            if isinstance(_data, np.ndarray):
                # Sanity check that trace.data is a vector
                if len(_data.shape) != 1:
                    raise TypeError('trace contained a multi-dimensional array - not allowed!')
                else:
                    # Check if the array is masked
                    if np.ma.is_masked(_data):
                        self.data = trace.data.data
                        self.msg_data_mask = trace.data.mask
                        self.fill_value = trace.data.fill_value
                    # If unmasked, write to 
                    else:
                        self.data = _data
            else:
                raise TypeError('trace.data was not a numpy.ndarray - not allowed!')
                
    def _wave2msg(self, wave, stricttype=False):
        if self._validate_wave(wave, stricttype=stricttype):
            self.station = wave['station']
            self.network = wave['network']
            self.channel = wave['channel']
            self.location = wave['location']
            self.sncl = f'{self.station}.{self.network}.{self.channel}.{self.location}'
            self.nsamp = wave['nsamp']
            self.samprate = wave['samprate']
            self.startt = wave['startt']
            self.endt = wave['endt']
            self.data = wave['data']
        else:
            raise SyntaxError('Invalid formatting for input "wave"')

    def _validate_wave(self, wave, stricttype=False):
        """
        Validate input `wave` reasonably looks like the PyEW representation of a tracebuf2 message in Python
        """
        keys = ['station','network','channel','location','nsamp','samprate','startt','endt','datatype','data']
        # NOTE: The 'int' requirement on 'samprate' is going to cause issues with analog sensors...
        # types = [str, str, str, str, int, float, int, int, ]
        types = [str, str, str, str, int, int, int, int, str, np.ndarray]
        key_types = dict(zip(keys,types))

        # Confirm `wave` is a dictionary
        if isinstance(wave, dict):
            # Check that all keys are present
            if all(x.lower() in keys for x in wave.keys()):
                if stricttype:
                    # Check that all keyed values have appropriate type
                    if all(isinstance(wave[_k], key_types[_k]) for _k in keys):
                        return True
                    else:
                        return False
                else:
                    return True
            else:
                # raise KeyError('Required keys missing from input "wave"')
                return False
        else:
            # raise TypeError('"wave" must be type dict')
            return False
        

    def output_wave(self):
        """
        Generate a PyEW `wave` message representation of a tracebuf2 message
        """
        out_wave = {'station': self.station,
                    'network': self.network,
                    'channel': self.channel,
                    'location': self.location,
                    'nsamp': self.nsamp,
                    'samprate': self.samprate,
                    'startt': self.startt,
                    'endt': self.endt,
                    'datatype': self.datatype,
                    'data': self.data}
        # Handle case if there's a masked array instance
        if len(self.mask_array) != 0:
            if self.mask_array.shape == self.data.shape:
                out_wave['data'][self.mask_array] = self.fill_value
        
        return out_wave

    def output_trace(self):
        """
        Generate an obspy.Trace object
        """
        tr = Trace()
        tr.stats.station = self.station
        tr.stats.network = self.network
        tr.stats.channel = self.channel
        tr.stats.location = self.location
        tr.stats.starttime = UTCDateTime(self.startt)
        tr.stats.sampling_rate = self.samprate
        if len(self.mask_array) != 0:
            if self.mask_array.shape == self.data.shape:
                tr.data = np.ma.MaskedArray(data=self.data,
                                            mask=self.mask_array,
                                            fill_value=self.fill_value,
                                            dtype=EW2NPDTYPES[self.datatype])
        else:
            tr.data = self.data
        
        return tr



class TensorMsg(_BaseMsg):
    """
    
    """

    def __init__(self, tensor, sncl, startt, samprate, order='ZNE'):
        super().__init__(mtype='TYPE_TRACEBUF2', mcode=19)
        self.tensor = torch.Tensor()
        self.sncl = '...--'
        self.station = ''
        self.network = ''
        self.channels = ''
        self.location = '--'
        self.samprate = 100.
        self._order = order
        if isinstance(tensor, np.ndarray):
            self.tensor = torch.Tensor(tensor)
        elif isinstance(tensor, torch.Tensor):
            self.tensor = tensor
        else:
            raise TypeError('Input "tensor" must be type numpy.ndarray or torch.Tensor')
        
        if len(self._order) in self.tensor.shape():
            for _i in range(len(self.tensor.shape)):
                if self.tensor.shape == _i:
                    self.channel_axis = _i
                else:
                    self.data_axis = _i
        else:
            raise ValueError('input "tensor" does not have compatable dimensions with proposed "order"')

        # Compat. checks on 'sncl'
        if isinstance(sncl, str):
            if len(sncl.split('.')) == 4:
                self.sncl = sncl
                parts = sncl.split('.')
                if len(parts[0]) <= 4:
                    self.station = parts[0]
                else:
                    self.station = parts[0][:4]
                if len(parts[1]) <= 2:
                    self.network = parts[1]
                else:
                    self.network = parts[1][:2]
                if len(parts[2]) == len(order):
                    self.channel = parts[2]
                    if self.channel != self.order:
                        if all(x in self.order for x in self.channel):
                            # Advise sort
                            Warning('channel and order elements match, but require a sort')
                        else:
                            raise ValueError('channel and order have mismatched element(s)')
                else:
                    raise ValueError('input "sncl" channel element does not have the right number of characters')
                if len(parts[3]) <= 2:
                    self.location = parts[3]
                else:
                    self.location = parts[3][:2]
                if self.location in ['',' ','  ']:
                    self.location = '--'
            else:
                raise IndexError('Insufficient .delimited entries in input "sncl" - requires 4')
        else:
            raise TypeError('Input "sncl" must be type str')
        
        if isinstance(startt, (float, np.float32)):
            self.startt = startt
        else:
            raise TypeError('Input "startt" must be type float or numpy.float32')



    def __repr__(self):
        rstr = super().__repr__()
        rstr += 