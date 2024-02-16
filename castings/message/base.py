"""
:module: wyrm.classes.pyew_msg
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This module contains the fundamental base class definition for Python-side 
    Earthworm messages based formatting and objects from PyEarthworm 

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
import wyrm.util.input_compatability_checks as icc

# CREATE GLOBAL VARIABLES FOR MESSAGE TYPES FOR NOW..
# TODO: HAVE SOME FUNCTIONALITY TO CROSS-REFERENCE WITH earthworm_global.d AND
#       INSTALLATION SPECIFIC MESSAGE CODE (earthworm_local.d?) BEFORE STARTUP


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

    def __init__(self, mtype="TYPE_TRACEBUF2", mcode=None):
        # Attach input compatability check methods
        self.bounded_intlike = icc.bounded_intlike
        self.bounded_floatlike = icc.bounded_floatlike
        self.none_str = icc.none_str
        self.iterable_characters = icc.iterable_characters
        # Validate message
        self._validate_basemsg(mtype, mcode)

    def _validate_basemsg(self, mtype, mcode):
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

        # Do initial value/type compatability checks
        if mcode is None:
            pass
        else:
            mcode = self.bounded_intlike(mcode, name='mcode', minimum=0, maximum=255)
            # If in earthworm_global.d range
            if 0 <= self.mcode <= 99:
                # Error if self.mcode in [0,99] and invalid
                if mcode not in EW_GLOBAL_MESSAGE_CODES:
                    raise ValueError(
                        "mcode falls into earthworm_global.d\
                                        reserved range [0,99] but is not a\
                                        valid value")

        # Validation on mtype type and syntax sniff-tests
        if mtype is None:
            pass
        else:
            mtype = self.none_str(mtype)
            # Sniff-test on all-caps syntax
            if mtype.upper() != mtype:
                raise SyntaxError("mtype must be all-caps")
            # Sniff-test on leading "TYPE_"
            if mtype[:5] != "TYPE_":
                raise SyntaxError('mtype must start with "TYPE_"')
            
        # Cross validation checks
        # If both are None
        if mtype is None and mcode is None:
            self.mtype = mtype
            self.mcode = mcode
            # raise TypeError("Must assign mtype and/or mcode as non-None-type")

        # If mtype is None, but mcode passed individual checks above
        elif mtype is None and mcode is not None:
            # If self.mcode falls into earthworm_global.d message code range
            if 0 <= self.mcode <= 99:
                self.mcode = mcode
                self.mtype = EW_GLOBAL_CT[mcode]
            # IF self.mcode falls into the installation range, kick error
            # because both mtype and mcode need to be defined for these
            else:
                raise SyntaxError(
                    "mcode in [100, 255] - installation\
                          message codes - must specify mtype"
                )

        # If mtype passed individual checks above but mcode is None
        elif mtype is not None and mcode is None:
            # if mtype is in global message types, use this to assign mcode!
            if mtype in EW_GLOBAL_MESSAGE_TYPES:
                self.mtype = mtype
                self.mcode = EW_GLOBAL_TC[self.mtype]
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
            if 0 <= mcode <= 99:
                # If mismatch
                if EW_GLOBAL_CT[mcode] != mtype:
                    raise ValueError(
                        "mcode is in earthworm_global.d\
                        reserved range [0,99] but mtype does not match"
                    )
                # Otherwise pass
                else:
                    self.mtype = mtype
                    self.mcode = mcode
            # If mcode in installation range
            else:
                print(
                    f"Assuming {self.mtype} : {self.mcode} matches\
                          installation message defs"
                )
                self.mtype = mtype
                self.mcode = mcode

    def __repr__(self):
        rstr = f"MTYPE: {self.mtype}\n"
        rstr += f"MCODE: {self.mcode}"
        return rstr
