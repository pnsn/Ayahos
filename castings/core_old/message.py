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
from obspy import Trace, Stream
from obspy.realtime import RtTrace
from collections import deque
from copy import deepcopy

# import PyEW

# CREATE GLOBAL VARIABLES FOR MESSAGE TYPES FOR NOW..
# TODO: HAVE SOME FUNCTIONALITY TO CROSS-REFERENCE WITH earthworm_global.d AND
#       INSTALLATION SPECIFIC MESSAGE CODE (earthworm_local.d?) BEFORE STARTUP
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


# NOTE: I've seen both 's4' and 'f4' show up in PyEarthworm documentation.
#       Include both for redundance
# NPDTYPES = [np.int16, np.int32, np.float32, np.float32]
NPDTYPES = [
    np.dtype("int16"),
    np.dtype("int32"),
    np.dtype("float32"),
    np.dtype("float32"),
]
NPDTYPESSTR = ["int16", "int32", "float32", "float32"]
EWDTYPES = ["i2", "i4", "s4", "f4"]
# Form two-way look-up dictionaries
NP2EWDTYPES = dict(zip(NPDTYPES, EWDTYPES))
EW2NPDTYPES = dict(zip(EWDTYPES, NPDTYPES))

NPSTR2EWDTYPES = dict(zip(NPDTYPESSTR, EWDTYPES))


class _BaseMsg(object):
    """
    Message fundamental BaseClass that provides message type labels for
    Earthworm message formats on the python side of Wyrm / PyEarthworm
    :: ATTRIBUTES ::
    :attrib mtype: [str] TYPE_* Earthworm Message Designation
                    must be all UPPER
    :attrib mcode: [int] in [0, 255] Message code.
                  if mcode in [0,99], must conform to earthworm_global.d
                  conventions - excerpt included in this module's help
                  documentation.
    """

    def __init__(self, mtype="TYPE_TRACEBUF2", mcode=19):
        self.mtype = mtype
        self.mcode = mcode
        self._validate_basemsg()
        super().__init__()

    def _validate_basemsg(self):
        # Validate mcode type and value
        if not isinstance(self.mcode, (int, type(None))):
            raise TypeError("mcode must be int or None type")
        # If int - run cross-checks against earthworm_global.d
        elif isinstance(self.mcode, int):
            # But not in range
            if not 0 <= self.mcode <= 255:
                raise ValueError("mcode must be an int in [0,255]")
            # Otherwise, if in earthworm_global.d range
            elif 0 <= self.mcode <= 99:
                # Error if self.mcode in [0,99] and invalid
                if self.mcode not in EW_GLOBAL_MESSAGE_CODES:
                    raise ValueError(
                        "mcode falls into earthworm_global.d\
                                      reserved range [0,99] but is not a\
                                      valid value"
                    )
                # Pass if self.mcode in [0,99] and valid
                else:
                    pass
            # Pass if self.mcode in [100, 255]
            else:
                pass
        # If None
        else:
            pass

        # Validation on mtype type and syntax sniff-tests
        if not isinstance(self.mtype, (str, type(None))):
            raise TypeError("mtype must be str or None type")
        # If str
        elif isinstance(self.mtype, str):
            # Sniff-test on all-caps syntax
            if self.mtype.upper() != self.mtype:
                raise SyntaxError("mtype must be all-caps")
            # If sniff test passed, continue
            else:
                pass
            # Sniff-test on leading "TYPE_"
            if self.mtype[:5] != "TYPE_":
                raise SyntaxError('mtype must start with "TYPE_"')
            # If sniff test passed, continue
            else:
                pass
        # If None
        else:
            pass

        # Cross validation checks
        # If both are None
        if self.mtype is None and self.mcode is None:
            raise TypeError("Must assign mtype and/or mcode as non-None-type")

        # If mtype is None, but mcode passed individual checks above
        elif self.mtype is None and self.mcode is not None:
            # If self.mcode falls into earthworm_global.d message code range
            if 0 <= self.mcode <= 99:
                self.mtype = EW_GLOBAL_CT[self.mcode]
                return True
            # IF self.mcode falls into the installation range, kick error
            # because both mtype and mcode need to be defined for these
            else:
                raise SyntaxError(
                    "mcode in [100, 255] - installation\
                          message codes - must specify mtype"
                )

        # If mtype passed individual checks above but mcode is None
        elif self.mtype is not None and self.mcode is None:
            # if mtype is in global message types, use this to assign mcode!
            if self.mtype in EW_GLOBAL_MESSAGE_TYPES:
                self.mcode = EW_GLOBAL_TC[self.mtype]
                return True
            # otherwise raise a more-generalized (longer winded) SyntaxError
            else:
                raise SyntaxError(
                    "mtype looks like an EW message type,\
                    but isn't in earthworm_global.d -\
                    assuming this is an institutional message\
                    which needs mtype in [100, 255] - None not allowed"
                )

        # If mtype and mcode passed checks and neither is None
        else:
            # If mcode in earthworm_global.d range
            if 0 <= self.mcode <= 99:
                # If mismatch
                if EW_GLOBAL_CT[self.mcode] != self.mtype:
                    raise ValueError(
                        "mcode is in earthworm_global.d\
                        reserved range [0,99] but mtype does not match"
                    )
                # Otherwise pass
                else:
                    return True
            # If mcode in installation range
            else:
                print(
                    f"Assuming {self.mtype} : {self.mcode} matches\
                          installation message defs"
                )
                return True

    def __repr__(self):
        rstr = f"MTYPE: {self.mtype}\n"
        rstr += f"MCODE: {self.mcode}\n"
        return rstr


class TraceMsg(Trace, _BaseMsg):
    """
    Multiple inheritance class merging obspy.Trace and _SNCLMsg classes
    to facilitate obspy.Trace class method use in this module and provide
    attributes to carry information on Earthworm message formatting
    (i.e., self.mtype and self.mcode) and provide extended class-methods
    for translating between PyEW `wave` and native obspy.Trace objects

    :: ATTRIBUTES ::
     v^ From Trace ^v
    :attrib stats: [obspy.core.trace.Stats] Metadata holder object
    :attrib data: [numpy.ndarray or numpy.ma.MaskedArray]
                    data holder object
     ... and others ... -- see obspy.core.trace.Trace

     ~~ From _BaseMsg ~~
    :attrib mtype: [str] Earthworm message type name
    :attrib mcode: [int] Earthworm message type code

      ++ New Attributes ++
    :attrib dtype: [str] Earthworm data format string
    :attrib sncl: [str] Station.Network.Channel.Location code string
    :attrib _waveflds: [list of str] keys for `wave` dict definition


    """

    def __init__(self, input=None, dtype="f4", mtype="TYPE_TRACEBUF2", mcode=19):
        """
        Create a TraceMsg object from a given input and EW-specific message
        metadata and data formatting

        :: INPUTS ::
        :param input: [obspy.Trace], [obspy.realtime.RtTrace],
                      [dict - `wave`] or [None]
                        inputs of Trace, RtTrace, and `wave` all populate
                        the Trace-inherited-attributes, whereas None
                        results in the default for an empty Trace() values
                        for these attributes
        :param dtype: [str] valid Earthworm datatype name OR
                      [type] valid numpy number format that conforms with Earthworm datatypes
        :param mtype: [str] valid Earthworm Message TYPE_* or [None]
                        see doc for wyrm.core.message._BaseMsg
        :param mcode: [int] valid Earthworm Message code
                        corresponding to mtype or [None]
                        see doc for wyrm.core.message._BaseMsg
        """
        self._waveflds = [
            "station",
            "network",
            "channel",
            "location",
            "nsamp",
            "samprate",
            "startt",
            "endt",
            "datatype",
            "data",
        ]
        # Compatability check on dtype
        if dtype in EWDTYPES:
            self.dtype = dtype
            if str(dtype) in EWDTYPES:
                self.ewdtype = dtype
            else:
                self.ewdtype = NPSTR2EWDTYPES[str(dtype)]
        else:
            raise TypeError(f"dtype must be in {EWDTYPES}")

        # Compatability check on input
        if input is None:
            Trace.__init__(self, data=np.array([]).astype(self.dtype), header={})
        elif isinstance(input, (Trace, RtTrace)):
            try:
                data = input.data.astype(self.dtype)
            except KeyError:
                breakpoint()
            header = input.stats
            Trace.__init__(self, data=data, header=header)
        elif isinstance(input, dict):
            if all(x in self._waveflds for x in input.keys()):
                data = input["data"].astype(self.dtype)
                # Grab SNCL updates
                header = {_k: input[_k] for _k in self._waveflds[:4]}
                header.update({"sampling_rate": input["samprate"]})
                header.update({"starttime": input["startt"]})

                Trace.__init__(self, data=data, header=header)
            else:
                raise SyntaxError(
                    "input dict does not match formatting\
                                   of a `wave` message"
                )
        else:
            raise TypeError(
                '"input" must be type None, obspy.Trace or dict\
                             (in PyEarthworm `wave` format)'
            )
        # Populate sncl based on self.stats
        sncl = f"{self.stats.station}."
        sncl += f"{self.stats.network}."
        sncl += f"{self.stats.channel}."
        sncl += f"{self.stats.location}"
        self.sncl = sncl
        # Initialize mtype and mcode attributes with validation
        _BaseMsg.__init__(self, mtype=mtype, mcode=mcode)

    def __repr__(self):
        """
        Expanded representation of obspy.Trace's __str__
        to include message and dtype information
        """
        rstr = super().__str__()
        rstr += f" | MTYPE: {self.mtype}"
        rstr += f" | MCODE: {self.mcode}"
        rstr += f" | DTYPE: {self.dtype} ({self.ewdtype})"
        return rstr

    def update_basemsg(self, mtype=None, mcode=19):
        """
        Update mtype and/or mcode if validation checks are passed
        :: INPUTS ::
        :param mtype: [str] or [None] message type name
        :param mcode: [int] or [None] message code
        """
        # If either argument presented mismatches with current metadata
        if mtype != self.mtype or mcode != self.mcode:
            # Attempt to make a test _BaseMsg from proposed arguments
            try:
                test_msg = _BaseMsg(mtype=mtype, mcode=mcode)
                # If the above dosen't kick errors during validation
                # update mtype and mcode with test_msg values
                self.mtype = test_msg.mtype
                self.mcode = test_msg.mcode

            except TypeError:
                raise TypeError
            except SyntaxError:
                raise SyntaxError
            except ValueError:
                raise ValueError
            except:
                print("Something else went wrong...")
        # If both match, do nothing and conclude
        else:
            pass

    def from_trace(self, trace, dtype=None):
        """
        Populate/overwrite contents of this TraceMsg
        object using an existing obspy.core.trace.Trace object

        :: INPUTS ::
        :param trace: [obspy.core.trace.Trace] trace object

        :: OUTPUT ::
        :return self: [wyrm.core.message.TraceMsg] TraceMsg object
        """
        if isinstance(trace, Trace):
            if dtype is not None and dtype in EWDTYPES:
                self.data = trace.data.astype(dtype)
                self.dtype = dtype
            elif dtype is None and trace.data.dtype in EWDTYPES:
                self.data = trace.data
                self.dtype = trace.data.dtype
            else:
                raise TypeError("Trace datatype must be in {EWDTYPES}")
            self.stats = trace.stats
            sncl = f"{self.stats.station}."
            sncl += f"{self.stats.network}."
            sncl += f"{self.stats.channel}."
            sncl += f"{self.stats.location}"
            self.sncl = sncl
        else:
            raise TypeError(
                'input "trace" must be\
                             type obspy.core.trace.Trace'
            )

    def to_trace(self):
        """
        Return a pure obspy.core.trace.Trace object
        (i.e., one without the extra TraceMsg bits)
        :: OUTPUT ::
        :return trace: [obspy.core.trace.Trace]
        """
        trace = Trace(data=self.data, header=self.stats)
        return trace

    def from_wave(self, wave):
        """
        Populate/overwrite contents of this TraceMsg
        object using an `wave` dictionary as defined in PyEW

        :: INPUTS ::
        :param wave: [dict] PyEW `wave` message object

        :: OUTPUT ::
        :return self: [wyrm.core.message.TraceMsg] TraceMsg object
        """
        if isinstance(input, dict):
            if all(x in self._waveflds for x in input.keys()):
                # Update dtype
                self.dtype = wave["datatype"]
                # Update data, fixing dtype
                data = input["data"].astype(self.dtype)
                self.data = data
                # Grab run header updates
                header = {_k: input[_k] for _k in self._waveflds[:4]}
                header.update({"sampling_rate": input["samprate"]})
                header.update({"starttime": input["startt"]})
                for _k in header.keys():
                    self.stats[_k] = header[_k]
                # Update SNCL representation
                sncl = f"{self.stats.station}."
                sncl += f"{self.stats.network}."
                sncl += f"{self.stats.channel}."
                sncl += f"{self.stats.location}"
                self.sncl = sncl
            else:
                raise SyntaxError(
                    "input dict does not match formatting of a\
                                   PyEarthworm `wave` message"
                )
        else:
            raise TypeError(
                '"wave" must be type dict\
                             (in PyEarthworm `wave` format)'
            )

    def to_wave(self, fill_value=None):
        """
        Generate a PyEW `wave` message representation of a tracebuf2 message
        from the contents of this TraceMsg object
        :: INPUT ::
        :param fill_value: [None], [int], [float],
                    or self.datatype's numpy equivalent
                    Optional: value to overwrite MaskedArray fill_value in the
                            event that self.data is a numpy.ma.MaskedArray
        """
        out_wave = {
            "station": self.stats.station,
            "network": self.stats.network,
            "channel": self.stats.channel,
            "location": self.stats.location,
            "nsamp": self.stats.npts,
            "samprate": self.stats.sampling_rate,
            "startt": self.stats.starttime.timestamp,
            "endt": self.stats.endtime.timestamp,
            "datatype": self.ewdatatype,
            "data": self.data.astype(self.datatype),
        }
        # If data are masked, apply the fill_value
        if np.ma.is_masked(out_wave["data"]):
            # If no overwrite on fill_value, apply fill_value as-is
            if fill_value is None:
                out_wave["data"] = out_wave["data"].filled()
            # If valid overwirte fill_value provide, use that
            elif isinstance(fill_value, (int, float, EW2NPDTYPES[self.datatype])):
                out_wave["data"] = out_wave["data"].fill(fill_value)
            else:
                raise TypeError(
                    f"fill_value must be type int, float, or\
                                 {EW2NPDTYPES[self.datatype]}"
                )
        else:
            pass
        return out_wave

    def to_ew(self, module, conn_index, fill_value=None):
        """
        Convenience method for generating a `wave` message
        from this TraceMsg and submitting it to a pre-established
        EWModule connection as a TRACEBUF2

        :: INPUTS ::
        :param module: [PyEW.EWModule] established EWModule object
        :param conn_index: [int] index of a pre-established connection
                        between Earthworm and Python hosted by `module`
        :param fill_value: [None], [int], [float],
                    or self.datatype's numpy equivalent
                    Optional: value to overwrite MaskedArray fill_value in the
                            event that self.data is a numpy.ma.MaskedArray
        """
        # Run compatability checks
        if not isinstance(module, PyEW.EWModule):
            raise TypeError("module must be type PyEW.EWModule")
        else:
            pass
        if not isinstance(conn_index, int):
            raise TypeError("conn_index must be type int")
        else:
            pass
        if self.mtype != "TYPE_TRACEBUF2" or self.mtype != 19:
            raise ValueError(
                'mtype must be "TYPE_TRACEBUF2"\
                              and mcode must be 19'
            )
        else:
            pass
        # Generate `wave`
        _wave = self.to_wave(fill_value=fill_value)
        # Submit `wave` to Earthworm
        module.put_wave(conn_index, _wave)

    def from_ew(self, module, conn_index):
        """
        Convenience method for pulling a single `wave` message
        from an Earthworm ring using an established PyEW.EWModule
        connection and populating/overwirint this TraceMsg with the
        pulled message's contents

        :: INPUTS ::
        :param module: [PyEW.EWModule] established EWModule object
        :param conn_index: [int] index of a pre-established connection
                        between Earthworm and Python hosted by `module`

        :: OUTPUT ::
        :return empty_wave: [bool] was the `wave` recovered an empty wave?
        """
        if not isinstance(module, PyEW.EWModule):
            raise TypeError("module must be type PyEW.EWModule")
        else:
            pass
        if not isinstance(conn_index, int):
            raise TypeError("conn_index must be type int")
        else:
            pass
        # Get wave from Earthworm
        _wave = module.get_wave(conn_index)
        # If not an empty_wave
        if _wave != {}:
            # Try to populate/overwrite this TraceMsg with new data
            try:
                self.from_wave(_wave)
                empty_wave = False
            # If SyntaxError is raised from self.from_wave(_wave), diagnose
            except SyntaxError:
                msg = "Missing key(s) from claimed wave:"
                for _k in _wave.keys():
                    if _k not in self._waveflds:
                        msg += f"\n{_k}"
                raise SyntaxError(msg)
            # If TypeError is raised from self.from_wave(_wave), echo TypeError
            except TypeError:
                raise TypeError
        # If empty_wave, change nothing
        else:
            empty_wave = True
        # return empty_wave assessment if no Errors are raised
        return empty_wave


class StreamMsg(Stream, _BaseMsg):
    def __init__(self, input=None, dtype="f4", mtype="TYPE_TRACEBUF2", mcode=19):
        # Initialize _BaseMsg elements (includes validation)
        _BaseMsg.__init__(self, mtype=mtype, mcode=mcode)
        self._waveflds = [
            "station",
            "network",
            "channel",
            "location",
            "nsamp",
            "samprate",
            "startt",
            "endt",
            "datatype",
            "data",
        ]

        # Compatability check on dtype
        if dtype not in EWDTYPES:
            raise ValueError(f"dtype must be in {EWDTYPES}")
        else:
            self.dtype = dtype

        # Compatability check on input
        # None input -> empty Stream
        if input is None:
            Stream.__init__(self, traces=None)
        # If input is list-like
        elif isinstance(input, (list, deque, Stream)):
            # If everything is already a TraceMsg, submit to Stream initialization
            if all(isinstance(x, TraceMsg) for x in input):
                Stream.__init__(self, traces=input)
            # If not everyting is a TraceMsg, but everything has a parent class Trace (or is Trace)
            # Convert everything in input into a TraceMsg and overwrite mtype / mcode / dtype
            elif all(isinstance(x, Trace) for x in input):
                traces = [
                    TraceMsg(tr, dtype=self.dtype, mtype=self.mtype, mcode=self.mcode)
                    for tr in input
                ]
                Stream.__init__(self, traces=traces)
            # Otherwise raise error
            else:
                raise TypeError(
                    "input list-like object must strictly contain\
                                 Trace objects"
                )
        # If input is a TraceMsg, directly pass to Stream
        elif isinstance(input, TraceMsg):
            Stream.__init__(self, traces=input)
        # If input is a Trace, but not a TraceMsg,
        # convert and then pass to Stream
        elif isinstance(input, Trace):
            Stream.__init__(
                self,
                traces=TraceMsg(
                    input, dtype=self.dtype, mtype=self.mtype, mcode=self.mcode
                ),
            )
        # Otherwise, kick TypeError
        else:
            raise TypeError(
                "input must be a Trace object or list-like object\
                             containing Trace objects"
            )

    def __str__(self):
        # Add BaseMsg and data type info to header of Stream's __str__()
        rstr = f"| MTYPE: {self.mtype} "
        rstr += f"| MCODE: {self.mcode} "
        rstr += f"| DTYPE: {self.dtype} |\n"
        rstr += super().__str__()
        return rstr

    def _update_to_tracemsg(self):
        """
        Iterate across traces in StreamMsg and convert
        non-TraceMsg traces into TraceMsg, using StreamMsg
        mtype, mcode, and dtype

        ::
        """
        for _tr in self.traces:
            if not isinstance(_tr, TraceMsg):
                if isinstance(_tr, Trace):
                    _tr = TraceMsg(
                        _tr, mtype=self.mtype, mcode=self.mcode, dtype=self.dtype
                    )

    def from_read(self, **kwargs):
        """
        Load a waveform file from disk using obspy.read
        and convert the read-in obspy.core.stream.Stream
        and it's component obspy.core.trace.Trace objects
        into a StreamMsg of TraceMsg objects.

        Read-in traces are appended to this StreamMsg object

        :: INPUTS ::
        :params **kwargs: see documentation of obspy.read
        """
        st = read(**kwargs)
        for _tr in st:
            self.traces.append(
                TraceMsg(_tr, mtype=self.mtype, mcode=self.mcode, dtype=self.dtype)
            )


class DEQ_Dict(object):
    """
    Double Ended Queue Dictionary
    Message buffer data structure for wyrm.core.io.*Wyrm classes

    A SNCL-keyed dictionary containing dictionaries with:
    'q': deque([]) for messages
    'age': int for number of pulses the queue has experienced where
            the number of elements in DEQ_Dict['sncl']['q'] is unchanged
    """

    def __init__(self, queues=None):
        self.queues = {}
        # If queues is None, return empty
        if queues is None:
            pass
        elif isinstance(queues, dict):
            # If the dictionary is composed of list-like objects or some type of wyrm message
            if all(
                isinstance(queues[_k], (_BaseMsg, deque, list)) for _k in queues.keys()
            ):
                # Iterate across each key and populate a new queue, assuming _k is a SNCL code (LOGO)
                for _k in queues.keys():
                    self.queues.update({_k: {"q": deque(queues[_k]), "age": 0}})
            # Otherwise, kick TypeError
            else:
                raise TypeError(
                    'Values of "queues" of type dict must be type "list", "deque", or some "_BaseMsg"'
                )
        else:
            raise TypeError('Input "queues" must be a dict or None')

    def __repr__(self, extended=False):
        rstr = f"DEQ_Dict containing {len(self.queues)} queues\n"
        for _i, _k in enumerate(self.queues.keys()):
            if _i < 4:
                rstr += f'{_k} | {len(self.queues[_k]["q"])} elements | age: {self.queues[_k]["age"]}\n'
            if not extended and len(self.queues) > 9:
                if _i == 4:
                    rstr += "   ...   \n"
                if _i > len(self.queues) - 5:
                    rstr += f'{_k} | {len(self.queues[_k]["q"])} elements | age: {self.queues[_k]["age"]}\n'
                if _i == len(self.queues) - 1:
                    rstr += (
                        'For a complete print, call "DEQ_Dict.__repr__(extended=True)"'
                    )
            elif _i >= 4:
                rstr += f'{_k} | {len(self.queues[_k]["q"])} elements | age: {self.queues[_k]["age"]}\n'
        return rstr

    def _append_pop(
        self, method="append", side="right", sncl="...--", msg=None, age=None
    ):
        # Compatability check for sncl
        if not isinstance(sncl, str):
            raise TypeError("sncl must be type str")
        elif len(sncl.split(".")) != 4:
            raise SyntaxError('sncl should have 4 "." delimited elements')
        else:
            pass
        # Compatability check for msg
        if not isinstance(msg, (type(None), _BaseMsg)):
            raise TypeError("msg must be None or type _BaseMsg")
        else:
            pass
        # Compatability check for age
        if not isinstance(age, (int, type(None))):
            raise TypeError("age must be type int or None")
        else:
            pass

        # If new sncl
        if sncl not in self.queues.keys():
            if method.lower() == "append":
                if isinstance(age, type(None)):
                    age = 0
                if side.lower() in ["right", "left"]:
                    self.queues.update({sncl: {"q": deque([msg]), "age": age}})
                else:
                    raise SyntaxError('side must be "left" or "right"')
            # Attempting to pop a non-existant queue returns an empty queue with sncl key
            if method.lower() == "pop":
                return {sncl: {"q": deque([]), "age": 0}}
        # If existing sncl
        else:
            qage = self.queues[sncl]["age"]
            # appending to an existing queue
            if method.lower() == "append":
                if age is None or qage == age:
                    if side.lower() == "right":
                        self.queues[sncl]["q"].append(msg)
                    elif side.lower() == "left":
                        self.queues[sncl]["q"].appendleft(msg)
                    else:
                        raise SyntaxError('side must be "left" or "right"')
                else:
                    raise ValueError("age must match current age of sncl-matched queue")

            # Popping off an existing queue
            elif method.lower() == "pop":
                if len(self.queues[sncl]["q"]) > 0:
                    if side.lower() == "right":
                        _pq = self.queues[sncl]["q"].pop()
                        return {sncl: {"q": deque(_pq), "age": qage}}
                    elif side.lower() == "left":
                        _pq = self.queues[sncl]["q"].popleft()
                        return {sncl: {"q": deque(_pq), "age": qage}}
                    else:
                        raise SyntaxError('side must be "left" or "right"')
                # Return sncl keyed queue element
                elif len(self.queues[sncl]["q"]) == 0:
                    return {sncl: self.queues.pop(sncl)}

    def append_msg(self, msg, age=None):
        if isinstance(msg, TraceMsg):
            self._append_pop(
                method="append", side="right", sncl=msg.sncl, msg=msg, age=age
            )
        elif isinstance(msg, StreamMsg):
            for _trMsg in msg:
                self._append_pop(
                    method="append", side="right", sncl=_trMsg.sncl, msg=_trMsg, age=age
                )
        else:
            raise TypeError("msg must be type TraceMsg or StreamMsg")

    def append_msg_left(self, msg, age=None):
        if isinstance(msg, TraceMsg):
            self._append_pop(
                method="append", side="left", sncl=msg.sncl, msg=msg, age=age
            )
        elif isinstance(msg, StreamMsg):
            for _trMsg in msg:
                self._append_pop(
                    method="append", side="left", sncl=_trMsg.sncl, msg=_trMsg, age=age
                )
        else:
            raise TypeError("msg must be type TraceMsg or StreamMsg")

    def pop_msg(self, sncl, bundled=False):
        x = self._append_pop(method="pop", side="right", sncl=sncl, msg=None, age=None)
        if not bundled:
            x = x[sncl]["q"]
        return x

    def pop_msg_left(self, sncl, bundled=False):
        x = self._append_pop(method="pop", side="left", sncl=sncl, msg=None, age=None)
        if not bundled:
            x = x[sncl]["q"]
        return x


class HDEQ_Dict(object):
    """
    Heirarchical Double Ended Queue Dictionary
    Message buffer data structure for wyrm.core.io.*Wyrm classes

    A SNCL-keyed dictionary containing dictionaries with structure:
    '{Network Code}'
        '{Station Code}'
            '{Location Code}'
                '{Channel Code}'
                    {'q': deque([]), 'age': int}

    'q': deque([]) for messages
    'age': int for number of pulses the queue has experienced where
            the number of elements in DEQ_Dict['sncl']['q'] is unchanged
    """

    def __init__(self, source_hdeq=None, extra_contents={"age": 0}):
        
        if not isinstance(extra_contents, dict):
            raise TypeError('extra_contents must be type dict')
        else:
            self.extra_contents = extra_contents

        if not isinstance(source_hdeq, (type(None), HDEQ_Dict)):
            raise TypeError('source_hdeq must be type None or a HDEQ_Dict')
        # If a HDEQ_Dict object is provided on input
        elif isinstance(source_hdeq, HDEQ_Dict):
            # Copy queues and codes
            self.queues = source_hdeq.copy().queues
            self.codes = source_hdeq.copy().codes
            # Iterate across contents
            for _sncl in self.codes:
                # Do validation of codes
                _target = self._get_sncl_target(_sncl)
                for _k in self.extra_contents.keys():
                    if _k not in _target.keys():
                        _target.update(self.extra_contents[_k])
                    else:
                        pass
        else:
            self.queues = {}
            self.codes = []
    
        
    def __repr__(self, extended=False):
        rstr = f"HDEQ_Dict containing {len(self.codes)} queue"
        if len(self.codes) > 1:
            rstr += 's\n'
        else:
            rstr += '\n'
        for _i, _k in enumerate(self.codes):
            _s, _n, _c, _l = _k.split(".")
            if _i < 4:
                rstr += f'{_k} | {len(self.queues[_n][_s][_l][_c]["q"])} elements | age: {self.queues[_n][_s][_l][_c]["age"]}\n'
            if not extended and len(self.codes) > 9:
                if _i == 4:
                    rstr += "   ...   \n"
                if _i > len(self.codes) - 5:
                    rstr += f'{_k} | {len(self.queues[_n][_s][_l][_c]["q"])} elements | age: {self.queues[_n][_s][_l][_c]["age"]}\n'
                if _i == len(self.codes) - 1:
                    rstr += (
                        'For a complete print, call "HDEQ_Dict.__repr__(extended=True)"'
                    )
            elif _i >= 4:
                rstr += f'{_k} | {len(self.queues[_n][_s][_l][_c]["q"])} elements | age: {self.queues[_n][_s][_l][_c]["age"]}\n'
        return rstr

    def copy(self):
        return deepcopy(self)

    def _sncl_exists(self, sncl):
        # Parse input
        if len(sncl.split(".")) != 4:
            raise SyntaxError
        else:
            _s, _n, _c, _l = sncl.split(".")

        # Check dictionary structure
        if _n in self.queues.keys():
            if _s in self.queues[_n].keys():
                if _l in self.queues[_n][_s].keys():
                    if _c in self.queues[_n][_s][_l].keys():
                        # sncl exists in indexing structure
                        # If sncl not in codes, add it
                        if sncl not in self.codes:
                            self.codes.append(sncl)
                        else:
                            pass
                        # And return true
                        return True
                    # Channel for SNCL does not exist in indexing structure
                    else:
                        return False
                # Location for SNCL does not exist in indexing strcuture
                else:
                    return False
            # Station for SNCL does not exist in indexing structure
            else:
                return False
        # Network for SNCL does not exist in indexing structure
        else:
            return False

    def _get_sncl_target(self, sncl):
        if self._sncl_exists(sncl):
            _s, _n, _c, _l = sncl.split(".")
            _target = self.queues[_n][_s][_l][_c]
            return _target
        else:
            return None

    def _add_index_branch(self, sncl):
        if not self._sncl_exists(sncl):
            self.codes.append(sncl)
            _s, _n, _c, _l = sncl.split(".")
            # If network code exists
            if _n in self.queues.keys():
                # If
                if _s in self.queues[_n].keys():
                    if _l in self.queues[_n][_s].keys():
                        self.queues[_n][_s][_l].update(
                            {_c: {"q": deque([])}}
                        )
                        state = True
                    else:
                        self.queues[_n][_s].update(
                            {_l: {_c: {"q": deque([])}}}
                        )
                        state = True
                else:
                    self.queues[_n].update(
                        {_s: {_l: {_c: {"q": deque([])}}}}
                    )
                    state = True
            else:
                self.queues.update(
                    {_n: {_s: {_l: {_c: {"q": deque([])}}}}}
                )
                state = True
        else:
            state = False
        # Only populate contents if new entry
        if state:
            self.queues[_n][_s][_l][_c].update(self.extra_contents)
        return state

    def _append_pop_queue(self, sncl, value, key="q", method="pop"):
        _k = key

        if self._sncl_exists(sncl):
            _target = self._get_sncl_target(sncl)
            if "append" in method.lower():
                if _k in _target.keys():
                    if isinstance(_target[_k], deque):
                        if method.lower() == "appendleft":
                            _target[_k].appendleft(value)
                        elif method.lower() == "append":
                            _target[_k].append(value)
                        else:
                            raise SyntaxError(
                                f'append-type "method" must be: "append" or "appendleft"'
                            )
                    else:
                        raise TypeError(
                            "Target key:value needs to be associated with a deque-type value"
                        )
                # If key is not present and running an append, append new keyed deque
                else:
                    _target.update({_k: deque([value])})

            elif "pop" in method.lower():
                if _k in _target.keys():
                    if len(_target[_k]) > 0:
                        if method.lower() == "pop":
                            try:
                                x = _target[_k].pop()
                                return x
                            except AttributeError:
                                raise AttributeError
                        elif method.lower() == "popleft":
                            try:
                                x = _target[_k].popleft()
                                return x
                            except AttributeError:
                                raise AttributeError
                else:
                    raise KeyError(
                        f"key {_k} does not exist in target sncl - no popping allowed"
                    )
        else:
            raise KeyError("Target SNCL is not present in HDEQ_Dict.queues structure")

    def append(self, sncl, value, key='q'):
        self._append_pop_queue(sncl, value, key=key, method="append")
    
    def appendleft(self, sncl, value, key='q'):
        self._append_pop_queue(sncl, value, key=key, method='appendleft')
        
    def pop(self, sncl, key='q'):
        x = self._append_pop_queue(sncl, None, key=key, method='pop')
        return x

    def popleft(self, sncl, key='q'):
        self._append_pop_queue(sncl, None, key=key, method="popleft")

    def _get_keyed_value(self, sncl, key):
        """
        Fetch a copy of a keyed value for a given SNCL entry
        """
        if self._sncl_exists(sncl):
            _target = self._get_sncl_target(sncl)
            if key in _target.keys():
                return _target[key]
            else:
                raise KeyError("Target key is not present in SNCL entry")
        else:
            raise KeyError("Target SNCL is not present in HDEQ_Dict.queues structure")

    def _replace_keyed_value(self, sncl, key, value):
        """
        Replace the value of a keyed value for a given SNCL entry
        """
        if self._sncl_exists(sncl):
            _target = self._get_sncl_target(sncl)
            if key in _target.keys():
                _target[key] = value
            else:
                _target.update({key: value})
        else:
            raise KeyError("Target SNCL is not present in HDEQ_Dict.queues structure")


    def _flatten(self):
        out = {}
        for _sncl in self.codes:
            _target = self.copy()._get_sncl_target(_sncl)
            _td = {_sncl:_target}
            out.update(_td)
        return out
    
    def _sort_channels(self, order='Z3N1E2'):
        for _sncl in self.codes:
            _s, _n, _c, _l = _sncl.split('.')
            _target = 