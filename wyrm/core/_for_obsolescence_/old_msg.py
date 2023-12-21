class SNCLMsg:
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


class WaveMsg(PEWMsg):
    """
    Formalized class definition of the `wave` object in PyEarthworm used to
    describe tracebuff2 messages in the Python-side of the module.

    This class is used to carry non-masked numpy data arrays representing
    individual, continuous, evenly sampled time series within the Python-side
    environment.

    NOTE:
    Any transmission of gappy (i.e., masked numpy arrays) or multi-channel
    (multi-dimensional numpy.ndarrays) time series data should be handled
    with the obspy.Trace and obspy.Stream objects.

    The following methods provide easy interface with the obspy API:
        WaveMsg.to_trace()
        WaveMsg.from_trace()

    The following methods provide easy conversion to/from the PyEarthworm
    `wave` object
    """
    def __init__(self, wave, default_dtype='s4'):
        """
        Create a WaveMsg object
        """
        if not isinstance(wave, (dict,Trace, Stream)):
            raise TypeError
        elif isinstance(wave, Stream):
            if len(wave) > 1:
                print('Too many entries in stream')
                raise IndexError
            elif len()


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

        :: INIT GENERATED ATTRIBUTES ::
        :attrib nsamp: [int] number of data
        :attrib endt: [float] epoch end time for data [sec since 1970-01-01]
        :attrib code: [str] SNCL code
        """
        super().__init__(station, network, channel, location)

        if startt is None:
            self.startt = -999
        elif str(startt).isnumeric():
            if np.isfinite(startt):
                self.startt = float(startt)
        if samprate is None:
            self.samprate = 1
        elif not str(samprate).isnumeric():
            self.samprate = 1.
        else:
            self.samprate = float(samprate)

        self.datatype = datatype
        self.dataclass = self.ew2np_dtype(datatype)
        self.data = data.astype(self.dataclass)
        self.nsamp = len(self.data)
        self.datatype = datatype

    def __repr__(self):
        rstr = super().__repr__()
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
        _dtype = self.ew2np_dtype(self.datatype)
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
        _dtype = self.ew2np_dtype(self.datatype)
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



# class MLWMsg(PEWMsg):
#     def __init__(self, station=None, network=None, channel=None, location=None, startt=None, samprate=1, data=np.array([]), datatype='i4'):


# class MLWMsg(PEWMsg):

#     def __init__(self, *args, ML_model='PhaseNet'):
#         self.ML_model = ML_model
#         if ML_model in ['PhaseNet','PN']:
#             self.ML_model = 'PhaseNet'
#             self.in_order = 'ZNE'
#             self.out_order = 'PSN'
#             self.axes = ('window','channel','data')
#             self.data_len = 3000

#         elif self.ML_model in ['EQT', 'EQTransformer']:
#             self.ML_model = 'EQTransformer'
#             self.in_order = 'ZNE'
#             self.out_order = 'DPS'
#             self.axes = ('channel','window','data')
#             self.data_len = 6000
        
#         else:
#             raise ValueError(f'ML_model name "{ML_model}" is not supported. Supported models: "PhaseNet", "EQTransformer"')
        
#         if isinstance(input, numpy.ndarray):
#             if input.shape[]


# def WindMessage(data)

# # class PyEW_PickMsg(PyEW_Msg):







class _SNCLMsg(_BaseMsg):
    """
    SNCL keyed message
    :: ATTRIBUTES ::
    :attrib mtype: [str] EW message TYPE (_BaseMsg super())
    :attrib mcode: [int] EW message code (_BaseMsg super())
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
        
        # Handle cases where location is not type STR
        if isinstance(location, (int, float)):
            location = str(int(location))
        elif isinstance(location, str):
            pass
        elif location is None:
            pass
        else:
            raise TypeError('Location must be type None, int, float, or str')

        # None and String parsing  
        if location is None:
            self.location = '--' 
        elif isinstance(location, str):
            # Strip all whitespace
            location = ''.join(location.split())
            if len(location) > 2:
                print(f'location code is too long {location}, truncating to the trailing 2 integers')
                self.location = location[:2]
            elif len(location) == 2:
                self.location = location
            elif len(location) == 1:
                if location.isnumeric():
                    self.location = '0'+location
                else:
                    self.location = location
            else:
                self.location = '--'
        else:
            raise TypeError(f"Something went wrong - got location of type {type(location)}")
        
        if len(self.station) > 4:
            self.station = self.station[:4]
        if len(self.network) > 2:
            self.network = self.network[:2]
        if len(self.channel) > 3:
            self.channel = self.channel[:3]

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
    :attrib mtype: [str] TYPE_TRACEBUF2 (super from _Base_Msg)
    :attrib mcode: [int] 19 (super from _Base_Msg) 
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