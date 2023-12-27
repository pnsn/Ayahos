import numpy as np
from obspy import Trace
from obspy.realtime import RtTrace
from wyrm.message.base import _BaseMsg
import PyEW

### DOCUMENT SOME DATA FORMAT TRANSLATIONS AS 
# NOTE: I've seen both 's4' and 'f4' show up in PyEarthworm documentation.
#       Include both for redundance
# NPDTYPES = [np.int16, np.int32, np.float32, np.float32]
NPDTYPES = [
    np.dtype("int16"),
    np.dtype("int32"),
    np.dtype("float32"),
    np.dtype("float32"),
]

# NOTE: Numpy Understands EWDTYPES as given strings to dtype
#       so ideally EWDTYPES should just be held onto throughout
NPDTYPESSTR = ["int16", "int32", "float32", "float32"]
EWDTYPES = ["i2", "i4", "s4", "f4"]
# Form two-way look-up dictionaries
NP2EWDTYPES = dict(zip(NPDTYPES, EWDTYPES))
EW2NPDTYPES = dict(zip(EWDTYPES, NPDTYPES))

NPSTR2EWDTYPES = dict(zip(NPDTYPESSTR, EWDTYPES))


class TraceMsg(Trace, _BaseMsg):
    """
    Multiple inheritance class merging obspy.Trace and wyrm._BaseMsg classes
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
