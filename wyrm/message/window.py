from obspy import Trace, Stream, UTCDateTime
from wyrm.structures.rtbufftrace import RtBuffTrace
import numpy as np


class WindowMsg(Stream):
    """
    Class WindowMsg. This class is a child of the obspy.core.stream.Stream
    class type to structure slices of buffered waveform data by seismometer
    (i.e., a specific instrument/band at a specific site) that provides the
    following additional attributes and functionalities

    :: Component Aliases ::
    :attrib V0: Vertical component trace ID
    :attrib H1: First horizontal component trace ID
    :attrib H2: Second horizontal component trace ID

    :: Incomplete Data Handling Rules:
    :attrib window_fill_rule: sets the rule for converting incomplete data into
                    a 3-channel representation of the data.
            Current supported rules:
            'zeros' - if one or both horizontal channels are missing, fill
                      with zero-vector data (after Retailleau et al., 2021)
                      and populate a 0-valued trace with slightly altered
                      metadata from vertical channel data. Specifically,
                      the SEED channel code letter for component will be set
                      to 0. I.e., HNZ --> 000
                      This will populate a second trace object in the self.traces
                      class attribute
            'cloneZ' - if one or both horizontal channels are missing, clone
                      vertical component data as horizontal channel data
                      (after Ni et al., 2023). This strictly aliases data
            'cloneHZ' - if one horizontal channel is missing, clone the other
                      horizontal component data to fill the missing data, if
                      both horizontal channels are missing, do 'cloneZ'
                      (after Lara et al., 2023). This strictly aliases data

    :: Windowing Information ::
    :attrib target_sr: Target sampling rate this window will end with
    :attrib target_npts: Target number of samples this window will end with
    :attrib ref_starttime: Reference starttime for first sample of final window


    :: obspy.core.stream.Stream inheritance ::
    WindowMsg is a child-class of Stream and its data storage structure is the sam
    i.e., self.traces = list of Trace objects. To make WindowMsg objects more compact
    cloned channels are just aliases to the same Trace objects in self.traces.

    As such, any pre-/post-processing conducted on the contents of a WindowMsg object
    should be conducted using obspy.core.stream.Stream class methods to avoid accidentially
    applying processing steps multiple times as might arise in the case of iterating
    across the component aliases. DO NOT ITERATE ACROSS Component Aliases.

    TODO: 
    - update documentation for new defs of self.V0 self.H1 self.H2
    - streamline handling of zeros
    - eliminate rule "000" - it's kinda silly and introduces extra overhead.
    """

    def __init__(
        self,
        V0=None,
        H1=None,
        H2=None,
        window_fill_rule="zeros",
        target_sr=100.0,
        target_npts=6000,
        ref_starttime=None,
        tolsec=0.03,
        tapsec=0.06,
        normtype='max'
    ):
        """
        Initialize a WindowMsg object

        :: INPUTS ::
        :param V0: [obspy.Trace] or [None]
                    Vertical component Trace object
        :param H1: [obspy.Trace] or [None]
                    Horizontal component 1 Trace object
        :param H2: [obspy.Trace] or [None]
                    Horizontal component 2 Trace object
        :param window_fill_rule: [str]
                    Empty channel fill rule. Supported options:
                    "zeros" - make horizontals 0-traces if any H1, H2 are None
                    "cloneZ" - clone vertical if any horizontals are None
                    "cloneHZ" - clone horizontal if one is present, clone vertical
                                if both horizontals are missing
        :param target_sr: [float] target sampling rate used to generate windowed data slice
        :param target_npts: [int] target number of samples used to generate windowed data slice
        :param ref_starttime: [None] or [obspy.UTCDateTime]
                            Reference starttime for window. Should be within a `tolsec`
                            samples of the target window sampling rate and the starttime of
                            V0.
        :param tolsec: [float] seconds that input trace starttimes can mismatch one another
                               or that V0data() can mismatch ref_starttime
        :param tapsec: [int] number of seconds for maximum taper length
        :param normtype: [str] type of normalization to apply by default

        WindowMsg is a child class of obspy.Stream, inheriting all of its class methods
        structure for storing lists of obspy.Trace data objects.
        """
        debug = False
        # Initialize parent class attributes (stream)
        super().__init__(self)

        # V0 compatability checks
        if isinstance(V0, Trace):
            self.V0 = V0.id
            if isinstance(V0, RtBuffTrace):
                self.traces.append(V0.as_trace())
            else:
                self.traces.append(V0.copy())
        elif V0 is None:
            self.V0 = V0
        else:
            raise TypeError("V0 must be type Trace or None")
        if debug:
            print(f"init V0 {self.V0}")

        # H1 compatability checks
        if isinstance(H1, Trace):
            self.H1 = H1.id
            if isinstance(H1, RtBuffTrace):
                self.traces.append(H1.as_trace())
            else:
                self.traces.append(H1.copy())
        elif H1 is None:
            self.H1 = H1
        else:
            raise TypeError("H1 must be type Trace or None")
        if debug:
            print(f"init H1 {self.H1}")
        # H2 compatability checks
        if isinstance(H2, Trace):
            self.H2 = H2.id
            if isinstance(H2, RtBuffTrace):
                self.traces.append(H2.as_trace())
            else:
                self.traces.append(H2.copy())
        elif H2 is None:
            self.H2 = H2
        else:
            raise TypeError("H2 must be type Trace or None")
        if debug:
            print(f"init H2 {self.H2}")

        # window_fill_rule compatability checks
        if not isinstance(window_fill_rule, str):
            raise TypeError("window_fill_rule must be type str")
        elif window_fill_rule in ["zeros", '000', "cloneZ", "cloneHZ"]:
            self.window_fill_rule = window_fill_rule
        else:
            raise ValueError(
                f'window_fill_rule {window_fill_rule} not supported. Only "zeros", "000", "cloneZ", or "cloneHZ"'
            )

        # target_sr compatability checks
        if isinstance(target_sr, (int, float)):
            if target_sr > 0:
                self.target_sr = target_sr
            else:
                raise ValueError("target_sr must be positive")
        else:
            raise TypeError("target_sr must be int or float")

        # target_npts compatability checks
        if isinstance(target_npts, int):
            if target_npts > 0:
                self.target_npts = target_npts
            else:
                raise ValueError("target_npts must be a positive integer")
        elif isinstance(target_npts, float):
            if target_npts > 0:
                self.target_npts = int(target_npts)
            else:
                raise ValueError("target_npts must be a positive integer")
        else:
            raise TypeError("target_npts must be an int-like number")

        # tolsec compatability checks
        if isinstance(tolsec, (float, int)):
            if tolsec >= 0:
                self.tolsec = tolsec
            else:
                raise ValueError("tolsec must be g.e. 0")
        else:
            raise TypeError("tolsec must be float-like")

        # ref_starttime compatability checks
        if isinstance(ref_starttime, UTCDateTime):
            if V0 is None:
                self.ref_starttime = ref_starttime
            elif isinstance(V0, Trace):
                if abs(V0.stats.starttime - ref_starttime) <= self.tolsec:
                    self.ref_starttime = ref_starttime
                else:
                    emsg = f"specified ref_starttime {ref_starttime} "
                    emsg += f"mismatches V0 starttime {V0.stats.starttime}."
                    emsg += "\nDefaulting to V0 starttime"
                    print(emsg)
                    self.ref_starttime = V0.stats.starttime
            else:
                raise TypeError("Somehow got a V0 that is not type Trace or None...")
        elif ref_starttime is None:
            if isinstance(V0, Trace):
                self.ref_starttime = V0.stats.starttime
            elif V0 is None:
                self.ref_starttime = UTCDateTime(0)
            else:
                raise TypeError("Somehow got a V0 that is not type Trace or None...")
        else:
            raise TypeError("ref_starttime must be type UTCDateTime or None")

        # tapsec compatability checks
        if isinstance(tapsec, (float,int)):
            if tapsec > 0 and np.isfinite(tapsec):
                self.tapsec = float(tapsec)
            else:
                raise ValueError('tapsec must be a non-zero, finite value')
        else:
            raise TypeError('tapsec must be float-like')

        # normtype compatability checks
        if isinstance(normtype, str):
            if normtype.lower() in ['max','maximum','m','minmax','minmaxscalar']:
                self.normtype = 'max'
            elif normtype.lower() in ['std','stdev','o','standard','standardscalar']:
                self.normtype = 'std'
            else:
                raise ValueError('normtype must be "max", "std" or select aliases - see source code')
        else:
            raise TypeError('normtype must be type str')
            

        # Input trace mutual compatability checks
        for _i, _id1 in enumerate([self.V0, self.H1, self.H2]):
            for _j, _id2 in enumerate([self.V0, self.H1, self.H2]):
                if _i > _j:
                    if _id1 is not None and _id2 is not None:
                        try:
                            if debug:
                                print(f"compat {_id1} v {_id2}")
                            self.check_trace_compatability(
                                _id1, _id2, tolsec=self.tolsec
                            )
                        except ValueError:
                            raise ValueError(
                                f"metadata incompatability between {_id1} and {_id2}"
                            )

        # Handle non 3-C WindowMsg inputs based on window_fill_rule
        self.apply_window_fill_rule()
        if debug:
            print(f"post wfr apply V0: {self.V0}, H1: {self.H1}, H2: {self.H2}")

    def __str__(self, extended=False):
        # WindowMsg parameter summary
        rstr = f"{len(self)} trace(s) in WindowMsg | "
        rstr += f"target npts: {self.target_npts} | "
        rstr += f"target S/R: {self.target_sr} Hz | "
        rstr += f"channel fill rule: {self.window_fill_rule}\n"
        # Unique Trace List
        for _i, _tr in enumerate(self.traces):
            if len(self) - 3 > _i >= 3:
                if extended:
                    rstr += f"{_tr.__str__()}\n"
                else:
                    if _i == 3:
                        rstr += f"...\n({len(self)-4} other traces)\n...\n"
            else:
                rstr += f"{_tr.__str__()}\n"
        # WindowMsg channe alias directory
        rstr += "Aliases\n"
        rstr += f"V0: {self.V0} | "
        rstr += f"H1: {self.H1} | "
        rstr += f"H2: {self.H2}"
        # rstr += f"{super().__str__(extended=extended)}"
        return rstr

    def __repr__(self, extended=False):
        """Short format representation of a WindowMsg"""
        rstr = self.__str__(extended=extended)
        return rstr

    # DATA LOOKUP METHODS #

    def fetch_with_id(self, nslc_id):
        """
        Fetch the trace(s) that match the specified N.S.L.C formatted
        id string

        :: INPUT ::
        :param nslc_id: [str] N.S.L.C formatted channel ID string

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
                    traces.append(_tr)
        if len(traces) == 0:
            traces = None
        elif len(traces) == 1:
            traces = traces[0]
        else:
            traces = Stream(traces)

        return traces

    def V0data(self):
        trs = self.fetch_with_id(self.V0)
        return trs

    def H1data(self):
        trs = self.fetch_with_id(self.H1)
        return trs

    def H2data(self):
        trs = self.fetch_with_id(self.H2)
        return trs

    def apply_window_fill_rule(self):
        """
        Apply the specified self.window_fill_rule to this
        WindowMsg.
        """
        if self.window_fill_rule == "zeros":
            self._apply_zeros_fill_rule()
        elif self.window_fill_rule == '000':
            self._apply_zeros_fill_rule(explicit_000=True)
        elif self.window_fill_rule == "cloneZ":
            self._apply_cloneZ_fill_rule()
        elif self.window_fill_rule == "cloneHZ":
            self._apply_cloneHZ_fill_rule()
        else:
            emsg = f"window_fill_rule {self.window_fill_rule} invalid."
            emsg += 'Supported: "zeros", "cloneZ", "cloneHZ"'
            raise ValueError(emsg)

    def _apply_zeros_fill_rule(self, explicit_000=False):
        """
        PRIVATE METHOD
        Apply "zeros" window_fill_rule to this WindowMsg
        """
        if self.H1 is None or self.H2 is None:
            if self.V0 is None:
                self.H1 = None
                self.H2 = None
            # If vertical data trace is present
            elif self.V0 is not None:
                # Fetch this trace
                _trs = self.V0data()
                # and overwrite self.traces to remove extraneous data
                if isinstance(_trs, Trace):
                    if explicit_000:
                        _tr0 = _trs.copy()
                        _tr0.data *= 0
                        _tr0.stats.channel = "000"
                        self.H1 = _tr0.id
                        self.H2 = _tr0.id
                        self.traces = [_trs, _tr0]
                    else:
                        self.traces = [_trs]
                        self.H1 = None
                        self.H2 = None

                elif isinstance(_trs, Stream):
                    self.traces = _trs.traces
                    if explicit_000:
                        _st0 = _trs.copy()
                        for _tr0 in _st0:
                            _tr0.data *= 0
                            _tr0.stats.channel = "000"
                            self.traces.append(_tr0)
                        self.H1 = _tr0.id
                        self.H2 = _tr0.id
                    else:
                        self.H1 = None
                        self.H2 = None

    def enforce_0_trace(self, zeroval=1e-20):
        for _tr in self.traces:
            if self.window_fill_rule == 'zeros':
                if _tr.stats.channel == '000':
                    _tr.data = np.ones(_tr.stats.npts) * zeroval

    def _apply_cloneZ_fill_rule(self):
        """
        PRIVATE METHOD
        Apply "cloneZ" window_fill_rule to this WindowMsg
        """
        if self.H1 is None or self.H2 is None:
            self.H1 = self.V0
            self.H2 = self.V0
            # If vertical data trace is present
            if self.V0 is not None:
                # Fetch V0 id match data
                _trs = self.fetch_with_id(self.V0)
                # and overwrite self.traces to remove extraneous data
                if isinstance(_trs, Trace):
                    self.traces = [_trs]
                elif isinstance(_trs, Stream):
                    self.traces = _trs.traces

    def _apply_cloneHZ_fill_rule(self):
        """
        PRIVATE METHOD
        Apply "cloneHZ" window_fill_rule to this WindowMsg
        """
        # If both horizontals are None, apply cloneZ
        if self.H1 is None and self.H2 is None:
            self._apply_cloneZ_fill_rule()
        # Otherwise, duplicate id for None-id horizontal channel
        elif self.H1 is not None:
            self.H2 = self.H1
        elif self.H2 is not None:
            self.H1 = self.H2

    def check_trace_compatability(self, id1, id2, tolsec=0.03):
        """
        Check compatability of two traces supposedly from the
        same instrument for the same period of time

        :: INPUTS ::
        :param id1: [str] trace object id 1 (N.S.L.C format)
        :param id2: [str] trace object id 2 (N.S.L.C format)
        :param tolsec: [int] sample tolerance at self.target_sr
                        sampling rate for differences in starttime
                        and endtime values for id1 and id2
        :: OUTPUT ::
        If compatable, this method returns bool output 'True'
        otherwise it raises a TypeError or ValueError with some additional
        information on the specific failed compatability test
        """
        tr1 = self.fetch_with_id(id1).copy()
        tr2 = self.fetch_with_id(id2).copy()
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
            if abs(s1[_k] - s2[_k]) > tolsec:
                raise ValueError(f"difference in {_k}'s outside tolerance")
            else:
                pass

        return True

    def msg_is_split(self):
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

    def msg_masked_status(self):
        """
        For each trace in self.trace, determine the masked status
        of its data and return a list of bools

        :: OUTPUT ::
        :return status: [list] sequential outputs from np.ma.is_masked(tr.data)
        """
        status = [np.ma.is_masked(_tr.data) for _tr in self.traces]
        return status

    def msg_starttimes_outside_tolerance(self):
        bool_list = []
        for _tr in self.traces:
            trts = _tr.stats.starttime
            deltat = abs(self.ref_starttime - trts)
            bool_list.append(deltat > self.tolsec)
        return bool_list

    def split_msg_data(self):
        """
        Apply a obspy.Stream.split() in-place on data in this WindowMsg
        Warning - this will permanently alter contained data.
        """
        new_traces = Stream()
        if any(self.msg_masked_status()):
            new_traces = self.split()
            self.traces = new_traces

    def merge_msg_data(
        self, fill_value=None, method=1, interpolation_samples=-1
    ):
        """
        Merge traces in this WindowMsg by id using obspy.Stream.merge()
        if message appears to be split (see msg_is_split())

        :: INPUTS ::
        :param pad: [bool] see obspy.core.stream.Stream.merge()
        :param fill_value:              ...
        :param method:                  ...
        :param interpolation_samples:   ...

        Merge operates in-place.
        """
        # If more traces than unique id's --> merge
        if self.msg_is_split():
            self.merge(
                method=method,
                fill_value=fill_value,
                interpolation_samples=interpolation_samples,
            )

    def sync_msg_starttimes(self, fill_value=0):
        """
        Use obspy.Trace.interpolate to synchronize mismatching
        trace starttimes. This method includes several compatability checks
        to ensure interpolate() operates as expected.

        Trace sampling rates are not altered by interpolation operations
        """
        if not isinstance(fill_value, (int, float)):
            raise TypeError("fill_value must be an int or float value")

        # Check if message contennts are split
        if not self.msg_is_split():
            # Check if message contains masked values
            if not any(self.msg_masked_status()):
                # Check if any traces have differen starttimes
                if any(
                    _tr.stats.starttime != self.ref_starttime for _tr in self.traces
                ):
                    # Check if trace starttimes are outside tolerance
                    exceeds_tolerance = self.msg_starttimes_outside_tolerance()
                    for _i, _tr in enumerate(self.traces):
                        # If outside tolerance, apply a non-mask generating padding
                        if exceeds_tolerance[_i]:
                            _tr.trim(
                                starttime=self.ref_starttime,
                                pad=True,
                                fill_value=fill_value,
                            )
                        _tr.interpolate(
                            _tr.stats.sampling_rate, starttime=self.ref_starttime
                        )
                # If all starttimes are identical, do nothing
                else:
                    pass
            # print warning if
            else:
                raise UserWarning(
                    "WindowMsg contains masked traces - must fill values prior to sync starttimes"
                )
        else:
            raise UserWarning(
                "WindowMsg appears to have split traces - must merge before syncing"
            )

    def trim_msg(self, fill_value=0, nearest_sample=True):
        """
        Trim the traces contained in this message to match the window specified by
        the ref_starttime, target_sr, and target_npts attributes
        """
        ts = self.ref_starttime
        te = ts + (self.target_npts - 1) / self.target_sr
        self.trim(
            starttime=ts,
            endtime=te,
            pad=True,
            fill_value=fill_value,
            nearest_sample=nearest_sample,
        )
        if any(_tr.stats.npts != self.target_npts for _tr in self.traces):
            raise UserWarning("Not all trimmed traces in WindowMsg meet target_npts")

    def normalize_msg(self):
        if self.normtype == "max":
            for _tr in self.traces:
                # Exclude case of '000'
                if _tr.stats.channel != '000':
                    _tr.normalize()
                else:
                    continue
        # If std normalization, use numpy methods for each trace
        elif self.normtype == 'std':
            for _tr in self.traces:
                # Exclude case of '000'
                if _tr.stats.channel == '000':
                    continue
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

    def to_stream(self, copy=True):
        new_stream = Stream()
        for _tr in self.traces:
            if copy:
                new_stream.append(_tr.copy())
            else:
                new_stream.append(_tr)

    def to_numpy(self, order="012", fill_value=0):
        keys = {"0": self.V0, "1": self.H1, "2": self.H2}
        if any(_tr.stats.npts != self.target_npts for _tr in self.traces):
            raise UserWarning(
                "not all traces conform to target_npts. Preprocessing required!"
            )
        else:
            pass

        if isinstance(order, (str, list)):
            if len(order) == 3:
                if all(str(x) in ["0", "1", "2"] for x in order):
                    pass
                else:
                    raise ValueError(
                        'order must be a unique character comprising "0", "1", and "2"'
                    )
            else:
                raise SyntaxError("order must have precisely 3 elements")
        else:
            raise TypeError("order must be a 3-element string or list of characters")
        # Convert string-type order into a list of characters
        if isinstance(order, str):
            order = [x for x in order]
        # Ensure order entries are strings
        if isinstance(order, list):
            order = [str(x) for x in order]

        ma_status = self.msg_masked_status()

        array_holder = []

        for _i, _k in enumerate(order):
            _tr = self.fetch_with_id(keys[_k])
            if isinstance(_tr, Trace):
                _data = _tr.data
            elif isinstance(_tr, Stream):
                raise TypeError(f'key {keys[_k]} returned a Stream - Must merge split traces before export')
            elif _tr is None:
                _data = np.zeros(self.target_npts)
            # Homogenize all arrays as masked with uniform fill value
            if any(ma_status):
                if not ma_status[_i]:
                    _data = np.ma.asarray(_data)
                    _data.fill_value = fill_value
                elif any(ma_status) and ma_status[_i]:
                    _data.fill_value = fill_value
            # Append to holder
            array_holder.append(_data)

        if any(ma_status):
            out_array = np.ma.array(array_holder)
        else:
            out_array = np.array(array_holder)

        return out_array

    def _preproc_example(self, fill_value=0):
        # Split if masked gaps exist
        self.split_msg_data()
        # Remove mean
        self.detrend("demean")
        # Remove trend
        self.detrend("linear")
        # Resample data with Nyquist sensitive filtering
        self.resample(self.target_sr)
        # Taper traces (or segments) on both ends with a length of 6 samples
        self.taper(None, max_length=self.tapsec, side="both")
        # Merge traces
        self.merge_msg_data(fill_value=fill_value)
        # Interpolate traces
        if self.window_fill_rule == '000':
            self.enforce_0_trace(zeroval=1e-20)
        self.sync_msg_starttimes(fill_value=fill_value)
        # Trim to specific length
        self.trim_msg()
        # if self.window_fill_rule == '000':
        #     self.enforce_0_trace(zeroval=1e-20)
        ## STEPS PROBABLY BEST HANDLED IN TorchWrym.pulse(x=)
        # Normalize traces
        self.normalize_msg()
        if self.window_fill_rule == '000':
            self.enforce_0_trace(zeroval=0.)
        # Convert to tensor
        array = self.to_numpy(order="012")
        return array
