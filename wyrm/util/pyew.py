"""
:module: wyrm.util.PyEW_translate.py
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains methods for translating betwee python objects in the PyEarthworm python-side environment
    and other commonly used formats in the python seismology environment. 
"""
import numpy as np
from obspy import UTCDateTime, Trace
from wyrm.core.mltrace import MLTrace


def npy2strdtype(dtype):
    """
    Provide formatting checks for dtype for submitting data to Earthworm

    :: INPUT ::
    :param dtype: [datatype-like object] e.g., np.int32, "int16", "f4"

    :: OUTPUT ::
    :return sdtype: [str] - string datatype representation
                Supported values:
                    'i2' - 16-bit integer
                    'i4' - 32-bit integer
                    'i8' - 64-bit integer
    """
    if dtype == 'i2':
        sdtype = 'i2'
    elif dtype == 'i4':
        sdtype = 'i4'
    elif dtype == 'i8':
        sdtype = 'i8'
    elif dtype == 'f4':
        sdtype = 'f4'
    else:
        raise TypeError(f'dtype {dtype} is not compatable with PyEW. Supported: "i2", "i4", "i8", and "f4", and python equivalents')
    return sdtype


def is_wave_msg(wave):

    # Explicitly define keys and types
    keys = ["station", "network", "channel", "location", 
            "nsamp", "samprate", "startt", "endt", "datatype", "data"]
    types = [str, str, str, str,
             int, float, float, float, str, np.ndarray]
    template = dict(zip(keys, types))
    # Check if not dict
    if not isinstance(wave, dict):
        return False
    # Check if missing/extraneous keys
    elif wave.keys() != template.keys():
        return False
    elif not all(isinstance(wave[_k], template[_k]) for _k in template.keys()):
        return False
    else:
        return True

def validate_wave_msg(wave):
    """
    Provide a more detailed diagnosis of wave message mismatches
    """
    is_wm = is_wave_msg(wave)
    if is_wm:
        status = 'all checks passed'
    else:
        # Explicitly define keys and types
        keys = ["station", "network", "channel", "location", 
                "nsamp", "samprate", "startt", "endt", "datatype", "data"]
        types = [str, str, str, str,
                int, float, float, float, str, np.ndarray]
        template = dict(zip(keys, types))
        # Run initial type check
        if not isinstance(wave, dict):
            status = 'wave is not type dict'
        # Run checks on keys
        if wave.keys() == template.keys():
            # If keys are good, check data types
            if all(isinstance(wave[_k], template[_k]) for _k in template.keys()):
                # If all good - set status to true
                status=True
            # If keys are good, but types are bad, document and return infostring as status
            else:
                status = 'Mismatched types: '
                for _k in template.keys():
                    if not isinstance(wave[_k], template[_k]):
                        status += f'{_k}: {type(wave[_k])} x {template[_k]}, '
                status = status[:-2] + '\n'
        # If keys are mismatched
        else:
            # Conduct bilateral checks on missing and extraneous keys
            stat1 = ''
            stat2 = ''
            # Compose stat1
            for _k in wave.keys():
                if _k not in template.keys():
                    stat1 += f'{_k}, '
            # compose stat2
            for _k in template.keys():
                if _k not in wave.keys():
                    stat2 += f'{_k}, '
            # And compose status infostring
            status = 'Key Mismatches'
            if stat1 != '':
                status += f'\nMissing: {stat1[:-2]}'
            if stat2 != '':
                status += f'\nExtraneous: {stat2[:-2]}'
    # Return status, whatever it may be
    return status
        


def wave2trace(wave):
    """
    Convert the output of an EWModule.get_wave() call into an Obspy Trace.

    :: INPUTS ::
    :param wave: [dictionary]
            Output dictionary from EWModule.get_wave(). This method uses the
            following elements:
                'data' - [numpy.ndarray] or [numpy.ma.Masked_Array] -> trace.data
                'station' - [str] - trace.stats.station
                'network' - [str] - trace.stats.network
                'channel' - [str] - trace.stats.channel
                'location' - [str] - trace.stats.location
                'startt' - [float] -> [UTCDateTime] = trace.stats.starttime
                'samprate' - [int] - trace.stats.sampling_rate
                'datatype' - [str] -> trace.data.dtype

    :: OUTPUT ::
    :return trace: [obspy.core.trace.Trace]
            Trace object
    """
    status = is_wave_msg(wave)
    if isinstance(status, str):
        raise SyntaxError(status)
    trace = Trace()
    # Format data vector
    _data = wave["data"].astype(wave['datatype'])
    # Compose header
    _header = {_k:wave[_k] for _k in ['station','network','channel','location']}
    _header.update({'starttime': UTCDateTime(wave["startt"]),
                    'sampling_rate': wave['samprate']})
    # Initialize trace
    trace = Trace(data=_data, header=_header)
    return trace

def wave2mltrace(wave):
    """
    Convert a PyEW wave dictionary message into
    a wyrm.core.trace.MLTrace object
    """
    status = is_wave_msg(wave)
    if isinstance(status, str):
        raise SyntaxError(status)
    header = {}
    for _k, _v in wave.items():
        if _k in ['station','network','channel','location']:
            header.update({_k:_v})
        elif _k == 'samprate':
            header.update({'sampling_rate': _v})
        elif _k == 'data':
            data = _v
        elif _k == 'startt':
            header.update({'starttime': UTCDateTime(_v)})
        elif _k == 'datatype':
            dtype = _v
    mlt = MLTrace(data=data.astype(dtype), header=header)
    return mlt


def trace2wave(trace, dtype=None):
    """
    Convert an obspy.core.trace.Trace object into a PyEarthworm compatable
    wave-dict object that can be sent from Python to Earthworm via
    and active PyEW.EWModule object (see wyrm.core.io.RingWyrm)

    If the 
    
    """
    if dtype is None:
        try:
            dtype = npy2strdtype(trace.data.dtype)
        except TypeError:
            raise TypeError
    else:
        try:
            dtype = npy2strdtype(dtype)
        except TypeError:
            raise TypeError
    # If data are masked, run split on individual data
    if np.ma.is_masked(trace.data):
        st = trace.split()
        waves = []
        for _tr in st:
            waves.append(trace2wave(_tr, dtype=dtype))
        return waves
    else:

        wave = {"station": trace.stats.station,
                "network": trace.stats.network,
                "channel": trace.stats.channel,
                "location": trace.stats.location,
                "nsamp": trace.stats.npts,
                "samprate": trace.stats.sampling_rate,
                "startt": np.round(trace.stats.starttime.timestamp, decimals=2),
                "dtype": dtype,
                "data": trace.data.astype(dtype)}
        return wave

def stream2waves(stream, dtype=None):
    """
    Convenience wrapper for trace2wave to handle converting an
    obspy.core.stream.Stream's contents into a list of wave
    messages. 

    This method applys the stream.split() method to a copy
    of the input `stream` prior to parsing to remove any
    masked/gappy data.

    :: INPUTS ::
    :param stream: [obspy.core.stream.Stream]
    :param dtype: None, [str], or [numpy.dtype] - Datatype to
                assign to data being converted. None input
                preserves native dtype of input traces contained
                in stream and checks those against compatable
                dtypes for Earthworm (see trace2wave

    :: OUTPUT ::
    :return tracebuff2_messages: [list]
                List of wave messages
    """
    # Iterate across traces and compose messages
    waves = []
    for _tr in stream:
        wave = trace2wave(_tr, dtype=dtype)
        if isinstance(wave, dict):
            waves.append(wave)
        elif isinstance(wave, list):
            waves += wave
    return waves

def format_pick2k_msg(
    modID,
    index,
    sncl,
    utct,
    amps=[0, 0, 0],
    phase_hint="",
    polarity="",
    quality=2,
    messageID=10,
    orgID=1,
):
    """
    Create a formatted string to submit as a TYPE_PICK2K message
    for Earthworm
    http://www.earthwormcentral.org/documentation4/PROGRAMMER/y2k-formats.html

    :: INPUTS ::
    :param modID: [int] Module ID (1-255)
    :param index: [int] Message index assigned by picker (0-9999)
    :param sncl: [4-tuple] Station Network Component Location codes
    :param utct: [obspy.core.utcdatetime.UTCDateTime] pick time
    :param amps: [3-list] amplitudes of 1st, 2nd, and 3rd peak after arrival time
    :param phase_hint: [str] phase type code
    :param polarity: [str] first motion code
    :param quality: [int] pick quality (0-4)
    :param messageID: [int] message-type ID
    :param orgID: [int] organization ID

    :: OUTPUT ::
    :return msg: [str] formatted message ending with a newline (\\n) character.
    """
    # Format UTCDateTime into appropriate string format
    tpcs = [int(x) for x in utct.format_arclink().split(",")]
    fsec = tpcs[-2] + (tpcs[-1] / 1e6)
    # Message-type, module, and organization ID
    msg = f"{messageID:-3d}{modID:-3d}{orgID:-3d}"
    # Message index and Site Network Component
    msg += f" {index:-4d} {sncl[0]:5s}{sncl[1]:2s}{sncl[2]:3s}"
    # Pick polarity, quality, and phase hint
    msg += f" {polarity:1s}{quality:1d}{phase_hint:2s}"
    # YYYYMMDD
    msg += f"{tpcs[0]:04d}{tpcs[1]:02d}{tpcs[2]:02d}"
    # HHmmss.ff
    msg += f"{tpcs[3]:02d}{tpcs[4]:02d}{fsec:05.2f}"
    # amplitudes and newline
    msg += f"{amps[0]:08d}{amps[1]:08d}{amps[2]:08d}\n"

    return msg


def validate_EW_msg_naming(mtype=None, mcode=None):
    """
    Provide a validation check on individual Earthworm Message Type
    names (mtype) or Message codes (mcode) or combinations thereof

    :: INPUTS ::
    :param mtype: [str] or None
                all-caps message type name starting with "TYPE_"
    :param code: [int] or None
                int in the range [0, 255]

    :: OUTPUT ::
    If mtype is None - return mtype associated with mcode, if mcode in [0,99]
    If mcode is None - return mcode associated with mtype if defined in earthworm_global.d
    If neither is None - return True if they match, false if they do not
    """


    EW_GLOBAL_MESSAGE_CODES = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            94,
            95,
            96,
            97,
            98,
        ]

    EW_GLOBAL_MESSAGE_TYPES = [
        "TYPE_WILDCARD",
        "TYPE_ADBUF",
        "TYPE_ERROR",
        "TYPE_HEARTBEAT",
        "TYPE_TRACE2_COMP_UA",
        "TYPE_NANOBUF",
        "TYPE_ACK",
        "TYPE_PICK_SCNL",
        "TYPE_CODA_SCNL",
        "TYPE_PICK2K",
        "TYPE_CODA2K",
        "TYPE_PICK2",
        "TYPE_CODA2",
        "TYPE_HYP2000ARC",
        "TYPE_H71SUM2K",
        "TYPE_HINBARC",
        "TYPE_H71SUM",
        "TYPE_TRACEBUF2",
        "TYPE_TRACEBUF",
        "TYPE_LPTRIG",
        "TYPE_CUBIC",
        "TYPE_CARLSTATRIG",
        "TYPE_TRIGLIST",
        "TYPE_TRIGLIST2K",
        "TYPE_TRACE_COMP_UA",
        "TYPE_STRONGMOTION",
        "TYPE_MAGNITUDE",
        "TYPE_STRONGMOTIONII",
        "TYPE_LOC_GLOBAL",
        "TYPE_LPTRIG_SCNL",
        "TYPE_CARLSTATRIG_SCNL",
        "TYPE_TRIGLIST_SCNL",
        "TYPE_TD_AMP",
        "TYPE_MSEED",
        "TYPE_NOMAGNITUDE",
        "TYPE_NAMED_EVENT",
        "TYPE_HYPOTWC",
        "TYPE_PICK_GLOBAL",
        "TYPE_PICKTWC",
        "TYPE_ALARM",
    ]
    # Form two-way look-up dictionaries
    EW_GLOBAL_TC = dict(zip(EW_GLOBAL_MESSAGE_TYPES, EW_GLOBAL_MESSAGE_CODES))
    EW_GLOBAL_CT = dict(zip(EW_GLOBAL_MESSAGE_CODES, EW_GLOBAL_MESSAGE_TYPES))
    # Handle case where mtype is not provided
    if mtype is None:
        if isinstance(mcode, int):
            if mcode in EW_GLOBAL_MESSAGE_CODES:
                return EW_GLOBAL_CT[mcode]
            elif 0 <= mcode <= 99:
                print(f'mcode {mcode} is in earthworm_global.d reserved range, but unused.')
            elif 100 <= mcode <= 255:
                print(f'mcode {mcode} is in the installation-specific message code range - contact your sysadmin')
            else:
                print(f'Value Warning: mcode {mcode} is out of range [0, 255] for Earthworm messages')
        elif mcode is None:
            print('both inputs are None - returning None')
            return None
        else:
            raise TypeError(f'mcode {mcode} must be type int or None')
    # Handle case where mcode is not provided
    elif mcode is None:
        if isinstance(mtype, str):
            if mtype in EW_GLOBAL_MESSAGE_TYPES:
                return EW_GLOBAL_TC[mtype]
            elif mtype.upper() in EW_GLOBAL_MESSAGE_TYPES:
                print('User Notice: message types should be all-caps')
                return EW_GLOBAL_TC[mtype.upper()]
            else:
                if mtype[:5] != 'TYPE_':
                    print(f'Syntax Warning: EW Message Types all start with "TYPE_" ({mtype})')
                else:
                    print('User Notice: message type {mtype} is not in the default Earthworm Message Types - consider codes in the range [100, 255] for installation specific uses')
        else:
            raise TypeError('mtype must be type str or None')
    
    else:
        if isinstance(mcode, int) and isinstance(mtype, str):
            if EW_GLOBAL_CT[mcode] == mtype:
                return True
            elif EW_GLOBAL_CT[mcode] == mtype.upper():
                print('User Notice: message types should be all-caps')
                return True
            else:
                return False
        elif not isinstance(mcode, int):
            raise TypeError('mcode must be type int or None')
        elif not isinstance(mtype, str):
            raise TypeError('mtype must be type str or None')
            
    
