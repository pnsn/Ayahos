from obspy import Trace, Stream, UTCDateTime
from wyrm.structures.rtbufftrace import RtBuffTrace
import wyrm.util.input_compatability_checks as icc
import seisbench.models as sbm
import numpy as np
import torch
from copy import deepcopy


class InstWindow(Stream):
    """
    Class InstWindow (Instrument-Level Window-Forming Class for ML models).
    This class is a child of the obspy.core.stream.Stream class
    used to structure slices of buffered waveform data from a single
    seismometer under the assumption that there are three channels
    and one of those is a vertical component (or somethings to that effect)

    (i.e., a specific instrument/band at a specific site) that provides the
    following additional attributes and functionalities to obspy.Stream

    :: Component Aliases ::
    :attrib Z: Vertical component trace ID
    :attrib N: First horizontal component trace ID
    :attrib E: Second horizontal component trace ID

    :: ML/DL Model Windowing Information ::
    :attrib target_sr: Target sampling rate this window will end with
    :attrib target_npts: Target number of samples this window will end with
    :attrib target_starttime: Reference starttime for first sample of final window
    :attrib target_in_channels: Target number of input channels for ML model
    :attrib model_name: case sensitive name of the ML model this window will
                        pass data to.

    :: Incomplete Data Handling Rules:
    :attrib missing_component_rule: sets the rule for converting incomplete data into
                    a 3-channel representation of the data.
            Current supported rules:
            'Zeros' - if one or both horizontal channels are missing, fill
                      with zero-vector data (after Retailleau et al., 2021)
                      and populate a 0-valued trace with slightly altered
                      metadata from vertical channel data. Specifically,
                      the SEED channel code letter for component will be set
                      to 0. I.e., HNZ --> 000
                      This will populate a second trace object in the self.traces
                      class attribute
            'CloneZ' - if one or both horizontal channels are missing, clone
                      vertical component data as horizontal channel data
                      (after Ni et al., 2023). This strictly aliases data
            'CloneHZ' - if one horizontal channel is missing, clone the other
                      horizontal component data to fill the missing data, if
                      both horizontal channels are missing, do 'cloneZ'
                      (after Lara et al., 2023). This strictly aliases data

    :: Windowing Information ::


    :: obspy.core.stream.Stream inheritance ::
    InstWindow is a child-class of Stream and its data storage structure is the same
    i.e., self.traces = list of Trace objects. To make WindowMsg objects more compact
    cloned channels are just aliases to the same Trace objects in self.traces.

    As such, any pre-/post-processing conducted on the contents of a InstWindow object
    should be conducted using obspy.core.stream.Stream class methods to avoid accidentially
    applying processing steps multiple times as might arise in the case of iterating
    across the component aliases. DO NOT ITERATE ACROSS Component Aliases.


    """

    def __init__(
        self,
        Z,
        N=None,
        E=None,
        target_starttime=None,
        fill_value=0.0,
        tolsec=2.0,
        missing_component_rule="Zeros",
        target_norm="peak",
        target_sr=100.0,
        target_npts=6000.0,
        target_channels=3,
        target_order="ZNE",
        target_overlap=1800,
        target_blinding=500,
        model_name="EQTransformer",
        index=None,
    ):
        """
        Initialize an InstWindow object

        :: INPUTS ::
        :param Z: [obspy.Trace]
                    Vertical component Trace object - REQUIRED
        :param N: [obspy.Trace] or [None]
                    Horizontal component 1 Trace object - optional
        :param E: [obspy.Trace] or [None]
                    Horizontal component 2 Trace object - optional
        :param target_starttime: [obspy.UTCDateTime] or None
                    Target starttime for output window.
                    If specified, may result in trace padding using fill_value
                    If none, uses starttime from vertical component trace.
        :param fill_value: [float-like] fill_value passed to masked arrays
                    that occur in the course of processing data in this
                    MLInstWindow object.
        :param tolsec: [float] tolerance in seconds for mismatching start
                    times for input traces
                    default is 0.03 sec (3 samples @ 100 sps)
        :param missing_component_rule: [str]
                    Empty channel fill rule. Supported options:
                    "Zeros" - make horizontals 0-traces if any N, E are None
                    "CloneZ" - clone vertical if any horizontals are None
                    "CloneHZ" - clone horizontal if one is present, clone
                                vertical if both horizontals are missing
        :param target_norm: [str] name of normalization to apply. Supported
                    'minmax' / 'peak' - normalize by maximum window amplitude
                    'std' - normalize by standard deviation of window amplitudes
        :param target_sr: [float-like] target sampling rate for processed window
        :param target_npts: [int] number of data per trace for processed window (axis 2)
        :param target_channels: [int] number of channels for processed window (axis 1)
        :param target_order: [str] case-sensitive order of components in a string
                        e.g., 'ZNE', 'Z12', 'ENZ'
        :param target_overlap: [int] number of data processed windows overlap by
        :param target_blinding: [int] number of data processed windows will have
                        blinding appiled to on either end after ML prediction

        :param model: [seisbench.models.WaveformModel]
                    from a SeisBench WaveformModel,
                        scrape the following information:
                    self._model_name = model.name
                    self._target_order = model.component_order
                    self._target_npts = model.in_samples
                        Defaults to 6000 if model has
                        no in_samples attribute
                    self._target_sr = model.sampling_rate
                        Defaults to 100 sps if model has no
                        sampling_rate attribute
                    self._normtype = model.norm
                        Defaults to 'minmax' if no model has
                        no norm attribute

        WindowMsg is a child class of obspy.Stream, inheriting all of its class methods
        structure for storing lists of obspy.Trace data objects.
        """
        # # Initialize parent class attributes (stream)
        super().__init__(self)

        # # Z compatability checks
        if isinstance(Z, Trace):
            self.Z = Z.id
            if isinstance(Z, RtBuffTrace):
                self.traces.append(Z.to_trace())
            else:
                self.traces.append(Z.copy())
        else:
            raise TypeError(
                "Z must be type obspy.core.trace.Trace or \
                             wyrm.structure.rtbufftrace.RtBuffTrace"
            )
        # # Define insturment code from full NSLC for Z-component trace
        self.inst_code = self.Z[:-1]

        # # N compatability checks
        if isinstance(N, Trace):
            self.N = N.id
            if isinstance(N, RtBuffTrace):
                self.traces.append(N.to_trace())
            else:
                self.traces.append(N.copy())
        elif N is None:
            self.N = N
        else:
            raise TypeError(
                "N must be type obspy.core.trace.Trace, \
                wyrm.structure.rtbufftrace.RtBuffTrace, or None"
            )

        # # E compatability checks
        if isinstance(E, Trace):
            self.E = E.id
            if isinstance(E, RtBuffTrace):
                self.traces.append(E.to_trace())
            else:
                self.traces.append(E.copy())
        elif E is None:
            self.E = E
        else:
            raise TypeError(
                "E must be type obspy.core.trace.Trace, \
                wyrm.structure.rtbufftrace.RtBuffTrace, or None"
            )

        # # target_starttime compatability checks
        if isinstance(target_starttime, UTCDateTime):
            # if abs(Z.stats.starttime - target_starttime) <= self.tolsec:
            self._target_starttime = target_starttime
        elif target_starttime is None:
            self._target_starttime = Z.stats.starttime
        else:
            raise TypeError("target_starttime must be type UTCDateTime or None")

        # # fill_value compatability checks
        if fill_value is None:
            self.fill_value = fill_value
        else:
            self.fill_value = icc.bounded_floatlike(
                fill_value,
                name="fill_value",
                minimum=None,
                maximum=None,
                inclusive=True,
            )

        # tolsec compatability checks
        self.tolsec = icc.bounded_intlike(
            tolsec, name="tolsec", minimum=0, maximum=None, inclusive=True
        )

        # # missing_component_rule compatability checks (with some grace on exact naming)
        if not isinstance(missing_component_rule, str):
            raise TypeError("missing_component_rule must be type str")
        elif missing_component_rule.lower() in ["zeros", "0s", "0"]:
            self._missing_component_rule = "Zeros"
        elif missing_component_rule.lower() in ["clonez", "clonez", "cz", "zs", "z"]:
            self._missing_component_rule = "CloneZ"
        elif missing_component_rule.lower() in [
            "clonehz",
            "clonezh",
            "czh",
            "chz",
            "zh",
            "hz",
        ]:
            self._missing_component_rule = "CloneHZ"
        else:
            raise ValueError(
                f'missing_component_rule {missing_component_rule} not supported. Supported rules: "Zeros", "CloneZ", or "CloneHZ"'
            )
        self._window_fill_status = None

        # # target_norm compatability checks
        if not isinstance(target_norm, (str, type(None))):
            raise TypeError("target_norm must be type str or None")
        elif isinstance(target_norm, str):
            if target_norm not in ["std", "minmax", "peak"]:
                raise ValueError(
                    f'target_norm {target_norm} not supported. Supported values: "peak", "minmax", "std"'
                )
            else:
                self._normtype = target_norm
        else:
            self._normtype = "peak"

        # # Compatability check for target_sr
        if target_sr is not None:
            _val = icc.bounded_floatlike(
                target_sr, name="target_sr", minimum=0, maximum=None, inclusive=False
            )
            self._target_sr = _val

        # # Compatability check for target_npts
        if target_npts is not None:
            _val = icc.bounded_intlike(
                target_npts,
                name="target_npts",
                minimum=0,
                maximum=None,
                inclusive=False,
            )
            self._target_npts = _val

        # Compatability check for target_channels
        if target_channels is not None:
            _val = icc.bounded_intlike(
                target_channels,
                name="target_channels",
                minimum=1,
                maximum=6,
                inclusive=True,
            )
            self._target_channels = _val

        # Compatability check for target_order:
        if not isinstance(target_order, (str, type(None))):
            raise TypeError("target_order must be type str or None")
        elif isinstance(target_order, str):
            if target_order.upper() == target_order:
                if len(target_order) == self._target_channels:
                    self._target_order = target_order
                else:
                    raise ValueError(
                        "number of elements in target order must match target_channels"
                    )
            else:
                raise SyntaxError("target order must be all capital characters")

        # Compatability check for target_overlap_npts
        if target_overlap is not None:
            _val = icc.bounded_intlike(
                target_overlap,
                name="target_overlap",
                minimum=-1,
                maximum=self._target_npts,
                inclusive=False,
            )
            self._target_overlap = _val

        # # Compatability check for target_blinding_npts
        if target_blinding is not None:
            _val = icc.bounded_intlike(
                target_blinding,
                name="target_blinding",
                minimum=0,
                maximum=self._target_npts,
                inclusive=True,
            )
            self._target_blinding = _val

        # # model_name compatability checks
        if not isinstance(model_name, (str, type(None))):
            raise TypeError("model_name must be type str or None")
        elif isinstance(model_name, str):
            self._model_name = model_name

        # # index compatability checks
        if index is None:
            self.index = None
        else:
            self.index = icc.bounded_intlike(
                index,
                name="index",
                minimum=0,
                maximum=None,
                inclusive=True,
            )

        # # Trace compatability cross-checks
        for _i, _id1 in enumerate([self.Z, self.N, self.E]):
            for _j, _id2 in enumerate([self.Z, self.N, self.E]):
                if _i > _j:
                    if _id1 is not None and _id2 is not None:
                        try:
                            self.check_trace_compatability(_id1, _id2)
                        except ValueError:
                            raise ValueError(
                                f"metadata incompatability between {_id1} and {_id2}"
                            )
        self._window_processing = ''
        # Handle non 3-C WindowMsg inputs based on missing_component_rule
        self.apply_missing_component_rule()
        

    def __str__(self, extended=False):
        keys = {"Z": self.Z, "N": self.N, "E": self.E}
        # WindowMsg parameter summary
        rstr = f"{len(self)} trace(s) in InstWindow | "
        rstr += f'{self.inst_code} | index: {self.index}\n'
        rstr += "Target > "
        rstr += f"model: {self._model_name} ð "
        rstr += f"dims: (1, {self._target_channels}, {self._target_npts}) ð "
        rstr += f"S/R: {self._target_sr} Hz | "
        rstr += f"missing comp. rule: {self._missing_component_rule}\n"
        # Unique Trace List
        for _i, _tr in enumerate(self.traces):
            # Get alias list
            aliases = [_o for _o in self._target_order if keys[_o] == _tr.id]
            # Handle extended formatting
            if len(self) - 3 > _i >= 3:
                if extended:
                    rstr += f"{_tr.__str__()}"
                    # Add alias information to trace line
                    if len(aliases) > 0:
                        rstr += " | ("
                        for _o in aliases:
                            rstr += _o
                        rstr += ") alias\n"
                    else:
                        rstr += " | [UNUSED]\n"
                else:
                    if _i == 3:
                        rstr += f"...\n({len(self)-4} other traces)\n...\n"
            else:
                rstr += f"{_tr.__str__()}"
                # Add alias information to trace line
                if len(aliases) > 0:
                    rstr += " | ("
                    for _o in aliases:
                        rstr += _o
                    rstr += ") alias\n"
                else:
                    rstr += " | [UNUSED]\n"
        rstr += f" Trim Target ð {self._target_starttime} - {self._target_starttime + self._target_npts/self._target_sr} ð"
        rstr += f'\n Window Processing: {self._window_processing}'
        return rstr

    def __repr__(self, extended=False):
        """Short format representation of a MLInstWindow"""
        rstr = self.__str__(extended=extended)
        return rstr

    # DATA LOOKUP METHODS #

    def fetch_with_id(self, nslc_id, ascopy=False):
        """
        Fetch the trace(s) that match the specified N.S.L.C formatted
        id string

        :: INPUT ::
        :param nslc_id: [str] N.S.L.C formatted channel ID string
        :param ascopy: [bool] operate on copies of traces?

        :: OUTPUT ::
        :return traces: [None] - if no traces matches nslc_id
                        [obspy.Traces] - if one trace object matches nslc_id
                        [obspy.Stream] - if multiple trace objects match nslc_id
        """
        traces = []
        if not isinstance(nslc_id, (type(None), str)):
            raise TypeError("id must be a N.S.L.C. string or None")
        elif nslc_id is None:
            return None
        else:
            for _tr in self.traces:
                if _tr.id == nslc_id:
                    if ascopy:
                        traces.append(_tr.copy())
                    else:
                        traces.append(_tr)
        if len(traces) == 0:
            traces = None
        elif len(traces) == 1:
            traces = traces[0]
        else:
            traces = Stream(traces)

        return traces

    def Zdata(self, ascopy=False):
        trs = self.fetch_with_id(self.Z, ascopy=ascopy)
        return trs

    def Ndata(self, ascopy=False):
        trs = self.fetch_with_id(self.N, ascopy=ascopy)
        return trs

    def Edata(self, ascopy=False):
        trs = self.fetch_with_id(self.E, ascopy=ascopy)
        return trs

    def apply_missing_component_rule(self):
        """
        Apply the specified self._missing_component_rule to this
        WindowMsg.
        """
        if self._missing_component_rule == "Zeros":
            self._apply_Zeros_rule()
        elif self._missing_component_rule == "CloneZ":
            self._apply_CloneZ_rule()
        elif self._missing_component_rule == "CloneHZ":
            self._apply_CloneHZ_rule()
        else:
            emsg = f"missing_component_rule {self._missing_component_rule} invalid."
            emsg += 'Supported: "Zeros", "CloneZ", "CloneHZ"'
            raise ValueError(emsg)
        self._window_processing += f'{self._missing_component_rule} applied'
        return self

    def _apply_Zeros_rule(self):
        """
        PRIVATE METHOD
        Apply "Zeros" missing_component_rule to this WindowMsg
        """
        if self.N is None or self.E is None:
            # Fetch vertical component trace(s)
            _trs = self.Zdata()
            # and overwrite self.traces to remove extraneous data
            if isinstance(_trs, Trace):
                self.traces = [_trs]
                self.N = None
                self.E = None
            # if there are multiple traces for id Z (e.g., split data)
            elif isinstance(_trs, Stream):
                self.traces = _trs.traces
                self.N = None
                self.E = None
            # Update window_fill_status
            self._window_fill_status = True
        else:
            self._window_fill_status = False

    def _apply_CloneZ_rule(self):
        """
        PRIVATE METHOD
        Apply "CloneZ" missing_component_rule to this WindowMsg
        """
        if self.N is None or self.E is None:
            self.N = self.Z
            self.E = self.Z
            # Fetch Z id match data
            _trs = self.fetch_with_id(self.Z)
            # and overwrite self.traces to remove extraneous data
            if isinstance(_trs, Trace):
                self.traces = [_trs]
            elif isinstance(_trs, Stream):
                self.traces = _trs.traces
            self._window_fill_status = "Z cloned"
        else:
            self._window_fill_status = False

    def _apply_CloneHZ_rule(self):
        """
        PRIVATE METHOD
        Apply "CloneHZ" missing_component_rule to this WindowMsg
        """
        # If both horizontals are None, apply cloneZ
        if self.N is None and self.E is None:
            self._apply_cloneZ_rule()
        # Otherwise, duplicate id for None-id horizontal channel
        elif self.N is not None and self.E is None:
            self.E = self.N
            self._window_fill_status = "N cloned"
        elif self.E is not None and self.N is None:
            self.N = self.E
            self._window_fill_status = "E cloned"
        else:
            self._window_fill_status = False

    # ###################################### #
    # Compatability and Status Check Methods #
    # ###################################### #

    def check_trace_compatability(self, id1, id2):
        """
        Check compatability of two traces supposedly from the
        same instrument for the same period of time

        :: INPUTS ::
        :param id1: [str] trace object id 1 (N.S.L.C format)
        :param id2: [str] trace object id 2 (N.S.L.C format)
        :param tolsec: [int] sample tolerance at self._target_sr
                        sampling rate for differences in starttime
                        and endtime values for id1 and id2
        :: OUTPUT ::
        If compatable, this method returns bool output 'True'
        otherwise it raises a TypeError or ValueError with some additional
        information on the specific failed compatability test
        """
        tr1 = self.fetch_with_id(id1, ascopy=True)
        tr2 = self.fetch_with_id(id2, ascopy=True)
        # Handle case where where fetch_with_id returns a Stream object
        if isinstance(tr1, Stream):
            print(f"{id1} maps to Stream --> merge()'d copy to trace")
            tr1 = tr1.merge()
        if isinstance(tr2, Stream):
            print(f"{id2} maps to Stream --> merge()'d copy to trace")
            tr2 = tr2.merge()

        # Compatability check on trace type
        if not isinstance(tr1, Trace):
            raise TypeError("id1 does not map to type Trace")
        if not isinstance(tr2, Trace):
            raise TypeError("id2 does not map to type Trace")

        # Get stats objects
        s1 = tr1.stats
        s2 = tr2.stats

        # # RUN COMPATABILITY CHECKS # #
        # Check stats that must be identical
        for _k in ["station", "network", "location", "sampling_rate", "calib"]:
            if s1[_k] != s2[_k]:
                raise ValueError(f"c1({s1[_k]}) != c2({s2[_k]}) [{_k} test]")
            else:
                pass
        # Check instrument/bands codes match
        if s1["channel"][:-1] != s2["channel"][:-1]:
            raise ValueError(
                f'c1({s1["channel"]}) != c2({s2["channel"]}) [inst/band test]'
            )
        else:
            pass
        # Check that starttime and endtime are within tolerance
        for _k in ["starttime", "endtime"]:
            if abs(s1[_k] - s2[_k]) > self.tolsec:
                raise ValueError(f"difference in {_k}'s outside tolerance")
            else:
                pass

        return True

    def is_wind_split(self):
        """
        Ascertain if the message data looks like it has been split
        i.e., duplicate trace IDs

        :: OUTPUT ::
        :return status: [bool] are there more traces than trace ids?
        """
        id_list = []
        for _tr in self.traces:
            _id = _tr.id
            if _id not in id_list:
                id_list.append(_id)
        if len(id_list) < len(self):
            status = True
        elif len(id_list) == len(self):
            status = False
        else:
            raise IndexError("more trace ids than traces...?")

        return status

    def are_traces_masked(self):
        """
        For each trace in self.trace, determine the masked status
        of its data and return a list of bools

        :: OUTPUT ::
        :return status: [list] sequential outputs from np.ma.is_masked(tr.data)
        """
        status = [np.ma.is_masked(_tr.data) for _tr in self.traces]
        return status

    def are_starttimes_outside_tolerance(self):
        bool_list = []
        for _tr in self.traces:
            trts = _tr.stats.starttime
            deltat = abs(self._target_starttime - trts)
            bool_list.append(deltat > self.tolsec)
        return bool_list

    # ############################################################ #
    # Window-style processing methods (as opposed to stream-style) #
    # ############################################################ #

    def wind_split(self):
        """
        Apply a obspy.Stream.split() in-place on data in this WindowMsg
        Warning - this will permanently alter contained data.
        """
        new_traces = Stream()
        if any(self.are_traces_masked()):
            new_traces = self.split()
            self.traces = new_traces
        self._window_processing += ' -> split'
        return self

    def wind_merge(self, method=1, interpolation_samples=-1):
        """
        Merge traces in this WindowMsg by id using obspy.Stream.merge()
        if message appears to be split (see is_wind_split())

        :: INPUTS ::
        :param pad: [bool] see obspy.core.stream.Stream.merge()
        :param fill_value:              ...
        :param method:                  ...
        :param interpolation_samples:   ...

        Merge operates in-place.
        """
        # If more traces than unique id's --> merge
        if self.is_wind_split():
            self.merge(
                method=method,
                fill_value=self.fill_value,
                interpolation_samples=interpolation_samples,
            )
            self._window_processing += ' -> merge'
        return self

    def wind_sync(self):
        """
        Use obspy.Trace.interpolate to synchronize mismatching
        trace starttimes. This method includes several compatability checks
        to ensure interpolate() operates as expected.

        Trace sampling rates are not altered by interpolation operations
        """
        # Check if message contennts are split
        if not self.is_wind_split():
            # Check if message contains masked values
            if not any(self.are_traces_masked()):
                # Check if any traces have different starttimes
                for _tr in self.traces:
                    # If a given trace has a different starttime
                    if _tr.stats.starttime != self._target_starttime:
                        # Check if samples are misaligned
                        if (
                            _tr.stats.starttime - self._target_starttime
                        ) % _tr.stats.delta != 0:
                            # Apply a non-mask generating padding that encompasses _target_starttime
                            _tr.trim(
                                starttime=self._target_starttime - _tr.stats.delta,
                                pad=True,
                                fill_value=self.fill_value,
                            )
                            # Use interpolation to sync the starttimes
                            _tr.interpolate(
                                _tr.stats.sampling_rate,
                                starttime=self._target_starttime,
                            )
                            # Remove any padding values
                            _tr.trim(starttime=self._target_starttime)
                        # if different start times is due to a gap, not sampling misalignment
                        else:
                            # do nothing to this trace
                            pass
                    # if trace and target starttimes are identical
                    else:
                        # do nothing to this trace
                        pass

            # print warning if trying to pass masked traces into sync
            else:
                raise UserWarning(
                    "WindowMsg contains masked traces - must fill values prior to sync starttimes"
                )
        else:
            raise UserWarning(
                "WindowMsg appears to have split traces - must merge before syncing"
            )
        return self

    def wind_trim(self, nearest_sample=True):
        """
        Trim the traces contained in this message to match the window specified by
        the target_starttime, target_sr, and target_npts attributes
        """
        ts = self._target_starttime
        te = ts + (self._target_npts - 1) / self._target_sr
        self.trim(
            starttime=ts,
            endtime=te,
            pad=True,
            fill_value=self.fill_value,
            nearest_sample=nearest_sample,
        )
        if any(_tr.stats.npts != self._target_npts for _tr in self.traces):
            raise UserWarning("Not all trimmed traces in WindowMsg meet target_npts")
        return self

    def wind_normalize(self):
        """
        Normalize individual traces in self.traces based on the
        specified normalization routine.
        """
        if self._normtype in ["peak", "max", "minmax"]:
            self.normalize()
        # If std normalization, use numpy methods for each trace
        elif self._normtype in ["std"]:
            for _tr in self.traces:
                # Get the std of the data
                if np.ma.is_masked(_tr.data):
                    _tnf = np.nanstd(_tr.data.data[~_tr.data.mask])
                else:
                    _tnf = np.nanstd(_tr.data)
                if np.isfinite(_tnf):
                    # Apply normalization
                    _tr.normalize(_tnf)
                else:
                    raise ValueError(
                        f"{_tr.id} | std normalization gets a non-finite scalar {_tnf}"
                    )
        else:
            pass
        return self

    # ################################## #
    # Data Format Transformation Methods #
    # ################################## #

    def to_stream(self):
        """
        Provide a copy of the contents of self.traces as an obspy.core.stream.Stream
        object in the specified order. Can produce duplicate or 0-valued traces.

        If channel codes

        :: OUTPUT ::
        :return new_stream: [obspy.core.stream.Stream]
        """
        new_stream = Stream()
        keys = {"Z": self.Z, "N": self.N, "E": self.E}
        for _o in self._target_order:
            _tr = self.fetch_with_id(keys[_o], ascopy=True)
            if isinstance(_tr, Trace):
                if _o != _tr.stats.channel[-1]:
                    _tr.stats.channel = _tr.stats.channel[:-1] + _o.lower()
                new_stream.append(_tr.copy())
            elif _tr is None:
                _tr = self.fetch_with_id(self.Z, ascopy=True)
                _tr.stats.channel = _tr.stats.channel[:-1] + _o.lower()
                _tr.data = np.zeros(_tr.stats.npts)
                new_stream.append(_tr)
        return new_stream

    def to_numpy(self):
        """
        Compose a (1, target_in_channels, target_samples) numpy array from data
        contained in this MLInstWindow.
        """
        emsg = ""
        # Run check that data sampling_rate matches target
        if any(
            [_tr.stats.sampling_rate != self._target_sr for _tr in self.to_stream()]
        ):
            emsg += "\nNot all traces have target sampling rate. Resampling required -> see obspy.core.trace.Trace.resample()"

        # Run check that data have been resampled and trimmed to the appropriate sampl
        if any([_tr.stats.npts != self._target_npts for _tr in self.to_stream()]):
            emsg += "\nNot all traces conform to target_npts. Trimming required -> see .wind_trim()"

        if any(
            [_tr.stats.starttime != self._target_starttime for _tr in self.to_stream()]
        ):
            emsg += "\nNot all traces conform to target_starttime. Time syncing required -> see .wind_sync()"
        if emsg != "":
            raise UserWarning(emsg + "\n" + self.__str__(extended=True))

        # Create holders
        array_holder = []
        ma_status = self.are_traces_masked()
        keys = {"Z": self.Z, "N": self.N, "E": self.E}
        for _i, _k in enumerate(self._target_order):
            _tr = self.fetch_with_id(keys[_k])
            if isinstance(_tr, Trace):
                _data = _tr.data
            elif isinstance(_tr, Stream):
                raise UserWarning(
                    f"key {keys[_k]} returned a Stream - Must merge split traces before export -> see .wind_merge()"
                )
            elif _tr is None:
                _data = np.zeros(self._target_npts)
            # Homogenize all arrays as masked with uniform fill value
            if any(ma_status):
                if not ma_status[_i]:
                    _data = np.ma.asarray(_data)
                    _data.fill_value = self.fill_value
                elif any(ma_status) and ma_status[_i]:
                    _data.fill_value = self.fill_value
            # Append to holder
            array_holder.append(_data)
        # Concatenate arrays and fill if needed
        if any(ma_status):
            out_array = np.ma.array(array_holder)
            out_array = out_array.filled()
        else:
            out_array = np.array(array_holder)

        # Add leading index
        out_array = out_array.reshape(1, self._target_channels, self._target_npts)
        return out_array

    def to_torch(self):
        array = self.to_numpy()
        tensor = torch.Tensor(array)
        return tensor

    def get_metadata(self):
        metadata = {
            "inst_code": self.inst_code,
            "component_order": self._target_order,
            "model_name": self._model_name,
            "weight_name": None,
            "label_codes": None,
            "samprate": self._target_sr,
            "starttime": self._target_starttime,
            "fill_rule": self._missing_component_rule,
            "fill_status": self._window_fill_status,
            "fill_value": self.fill_value,
            "index": self.index,
        }
        return metadata
    
    def export_processed_window(self, out_type='torch'):
        if not isinstance(out_type, str):
            raise TypeError('out_type must be type str')
        elif out_type.lower() == 'torch':
            data = self.to_torch()
            meta = self.get_metadata()
            return (data, meta)
        elif out_type.lower() == 'numpy': 
            data = self.to_numpy()
            meta = self.get_metadata()
            return (data, meta)
        elif out_type.lower() == 'obspy':
            data = self.to_stream()
            return data
        else:
            raise ValueError(f'Unsupported out_type {out_type}. Supported: "torch", "numpy", "obspy"')

    # def to_labeled_tensor(self):
    #     lt = LabeledTensor

    def _preproc_example(self, tapsec=0.06):
        # Split if masked gaps exist
        self.wind_split()
        # Filter
        self.filter("bandpass", freqmin=1, freqmax=45)
        # Remove mean
        self.detrend("demean")
        # Remove trend
        self.detrend("linear")
        # Resample data with Nyquist sensitive filtering
        self.resample(self._target_sr)
        # Taper traces (or segments) on both ends
        self.taper(None, max_length=tapsec, side="both")
        # Merge traces
        self.wind_merge()
        # Interpolate traces
        self.wind_sync()
        # Trim to specific length
        self.wind_trim()
        # Normalize traces
        self.wind_normalize()
        # # Convert to tensor & metadata pair
        # out = self.export_processed_window(out_type='torch')
        # return out


class SeisBenchWindow(InstWindow):
    """
    Child-class of InstWindow that populates the windowing parameters for
    InstWindow using information carried in a SeisBench WaveformModel object.

    Required inputs
    model = seisbench.models.WavefromModel
    Z = obspy.Trace
    N
    E
    target_starttime
    fill_value
    tolsec
    missing_component_rule

    """

    def __init__(
        self,
        model,
        Z,
        N=None,
        E=None,
        target_starttime=None,
        fill_value=0.0,
        tolsec=2.0,
        missing_component_rule="Zeros",
        index=None
    ):
        """
        Initialize a SeisBenchWindow object

        :: INPUTS ::
        :param model: [seisbench.models.WaveformModel]
                    WaveformModel to extract windowing information from. Non-standard
                    attributes have default values assigned as follows
                        model.in_samples -> defaults to 6000
                        model.sampling_rate -> defaults to 100
                        model.norm -> defaults to 'peak'
                        model.in_channels -> defaults to len(model.component_order)
                        model._annotate_args['overlap'][-1] -> defaults to 0
                        model._annotate_args['blinding'][-1][0] -> defaults to 0
        :param Z: [obspy.core.trace.Trace] vertical component trace data
        :param N: [obspy.core.trace.Trace] or [None] north component trace data
        :param E: [obspy.core.trace.Trace] or [None] east component trace data
        :param fill_value: [float] or [None] default fill_value for masked arrays
        :param tolsec: [float] maximum seconds mismatch between trace start/endtimes
                    for inputs Z, N, and E
        :param missing_component_rule: [str] name of missing_component_rule to use
                    see InstWindow
        """

        # Confirm that model is a seisbench.models.WaveformModel object
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError("model must be a seisbench.models.WaveformModel")
        # Provide default values for attributes not default in sbm.WaveformModel() objects
        if model.in_samples is None:
            msamp = 6000
        else:
            msamp = model.in_samples

        if model.sampling_rate is None:
            msrate = 100
        else:
            msrate = model.sampling_rate

        if "norm" not in dir(model):
            mnorm = "peak"
        else:
            mnorm = model.norm

        if "in_channels" not in dir(model):
            mchan = len(model.component_order)
        else:
            mchan = model.in_channels

        if "_annotate_args" not in dir(model):
            mover = 0
            mblind = 0
        else:
            if "overlap" in model._annotate_args.keys():
                mover = model._annotate_args["overlap"][-1]
            else:
                mover = 0
            if "blinding" in model._annotate_args.keys():
                mblind = model._annotate_args["blinding"][-1][0]
            else:
                mblind = 0

        # Initialize an InstWindow object with provided inputs
        super().__init__(
            Z=Z,
            N=N,
            E=E,
            target_starttime=target_starttime,
            fill_value=fill_value,
            tolsec=tolsec,
            missing_component_rule=missing_component_rule,
            target_sr=msrate,
            target_norm=mnorm,
            target_npts=msamp,
            target_channels=mchan,
            target_order=model.component_order,
            target_overlap=mover,
            target_blinding=mblind,
            model_name=model.name,
            index=index
        )
