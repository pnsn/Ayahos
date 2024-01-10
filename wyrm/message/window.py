from obspy import Trace, Stream, UTCDateTime
from wyrm.structures.rtbufftrace import RtBuffTrace


class WindowMsg(Stream):
    """
    Class WindowMsg. This class is a child of the obspy.core.stream.Stream
    class type to structure slices of buffered waveform data by seismometer
    (i.e., a specific instrument/band at a specific site) that provides the
    following additional attributes and functionalities

    :: Component Aliases ::
    :attrib V0: Vertical component aliased trace
    :attrib H1: First horizontal component aliased trace
    :attrib H2: Second horizontal component aliased trace

    :: Incomplete Data Handling Rules:
    :attrib ch_fill_rule: sets the rule for converting incomplete data into
                    a 3-channel representation of the data.
            Current supported rules:
            'zeros' - if one or both horizontal channels are missing, fill
                      with zero-vector data (after Retailleau et al., 2021)
                      and populate a 0-valued trace with slightly altered
                      metadata from vertical channel data. Specifically,
                      the SEED channel code letter for component will be set
                      to 0. I.e., HNZ --> HN0
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
    """

    def __init__(
        self,
        V0=None,
        H1=None,
        H2=None,
        ch_fill_rule="zeros",
        target_sr=100.0,
        target_npts=6000,
        ref_starttime=None,
        tolsamp=3,
    ):
        """
        Initialize a WindowMsg object

        :: INPUTS ::
        :param V0: [obspy.Trace] or [None]
                    Vertical component Trace object
        :param H1: [obspy.Trace] or [None]
                    Horizontal component Trace object 1
        :param H2: [obspy.Trace] or [None]
                    Horizontal component Trace object 2
        :param ch_fill_rule: [str]
                    Empty channel fill rule. Supported options:
                    "zeros" - make horizontals 0-traces if any H1, H2 are None
                    "cloneZ" - clone vertical if any horizontals are None
                    "cloneHZ" - clone horizontal if one is present, clone vertical
                                if both horizontals are missing
        :param target_sr: [float] target sampling rate used to generate windowed data slice
        :param target_npts: [int] target number of samples used to generate windowed data slice
        :param ref_starttime: [None] or [obspy.UTCDateTime]
                            Reference starttime for window. Should be within a `tolsamp`
                            samples of the target window sampling rate and the starttime of
                            V0.
        :param tolsamp: [int] number of samples at target_sr that input trace starttimes
                            can mismatch one another or that V0 can mismatch ref_starttime
        
        WindowMsg is a child class of obspy.Stream, inheriting all of its class methods
        structure for storing lists of obspy.Trace data objects. 
        """
        # Initialize parent class attributes (stream)
        super().__init__(self)

        # V0 compatability checks
        if isinstance(V0, RtBuffTrace):
            self.V0 = V0.as_trace()
        elif isinstance(V0, Trace):
            self.V0 = V0.copy()
        elif V0 is None:
            self.V0 = V0
        else:
            raise TypeError("V0 must be type Trace or None")
        if self.V0 is not None:
            self.traces.append(V0)

        # H1 compatability checks
        if isinstance(H1, RtBuffTrace):
            self.H1 = H1.as_trace()
        elif isinstance(H1, Trace):
            self.H1 = H1.copy()
        elif H1 is None:
            self.H1 = H1
        else:
            raise TypeError("H1 must be type Trace or None")
        if self.H1 is not None:
            self.traces.append(H1)

        # H2 compatability checks
        if isinstance(H2, RtBuffTrace):
            self.H2 = H2.as_trace()
        elif isinstance(H2, Trace):
            self.H2 = H2.copy()
        elif H2 is None:
            self.H2 = H2
        else:
            raise TypeError("H2 must be type Trace or None")
        if self.H2 is not None:
            self.traces.append(H2)

        # ch_fill_rule compatability checks
        if not isinstance(ch_fill_rule, str):
            raise TypeError("ch_fill_rule must be type str")
        elif ch_fill_rule in ["zeros", "cloneZ", "cloneHZ"]:
            self.ch_fill_rule = ch_fill_rule
        else:
            raise ValueError(
                f'ch_fill_rule {ch_fill_rule} not supported. Only "zeros", "cloneZ", or "cloneHZ"'
            )

        # # model_code compatability checks
        # if not isinstance(model_code, str):
        #     raise TypeError("model_code must be type str")
        # elif model_code in ["EQT", "PN"]:
        #     self.model_code = model_code
        # else:
        #     raise ValueError(f"model code {model_code} not supported")

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

        # tolsamp compatability checks
        if isinstance(tolsamp, int):
            if tolsamp >= 0:
                self.tolsamp = tolsamp
            else:
                raise ValueError("tolsamp must be g.e. 0")
        elif isinstance(tolsamp, float):
            if int(tolsamp) >= 0:
                self.tolsamp = int(tolsamp)
            else:
                raise ValueError("tolsamp must be g.e. 0")
        else:
            raise TypeError("tolsamp must be int-like")

        # ref_starttime compatability checks
        if isinstance(ref_starttime, UTCDateTime):
            if self.V0 is None:
                self.ref_starttime = ref_starttime
            elif isinstance(self.V0, Trace):
                if abs(
                    self.V0.stats.starttime - ref_starttime
                ) / self.target_npts <= self.tolsamp:
                    self.ref_starttime = ref_starttime
                else:
                    emsg = f"specified ref_starttime {ref_starttime} "
                    emsg += f"mismatches V0 starttime {self.V0.stats.starttime}."
                    emsg += "\nDefaulting to V0 starttime"
                    print(emsg)
                    self.ref_starttime = self.V0.stats.starttime
            else:
                raise TypeError("Somehow got a V0 that is not type Trace or None...")
        elif ref_starttime is None:
            if isinstance(self.V0, Trace):
                self.ref_starttime = self.V0.stats.starttime
            elif self.V0 is None:
                self.ref_starttime = UTCDateTime(0)
            else:
                raise TypeError("Somehow got a V0 that is not type Trace or None...")
        else:
            raise TypeError("ref_starttime must be type UTCDateTime or None")

        # Input trace mutual compatability checks
        for _i, _c in enumerate([self.V0, self.H1, self.H2]):
            for _j, _k in enumerate([self.V0, self.H1, self.H2]):
                if _i > _j:
                    if _c is not None and _k is not None:
                        try:
                            self.check_inst_meta_compat(_c, _k, tolsamp=self.tolsamp)
                        except ValueError:
                            raise ValueError(f"{_c} mismatches {_k}")

        # Handle non 3-C WindowMsg inputs based on ch_fill_rule
        # setting attribute aliases to duplicate data
        # NOTE: contents of self.traces is the definitive list of unique data
        if self.ch_fill_rule == "zeros":
            if self.H1 is None or self.H2 is None:
                self.CH = self.V0.copy()
                self.CH.data *= 0
                self.CH.stats.channel = self.CH.stats.channel[:-1] + "0"
                self.H1 = self.CH
                self.H2 = self.CH
                self.traces.append(self.CH)

        elif self.ch_fill_rule == "cloneZ":
            if self.H1 is None or self.H2 is None:
                self.CH = self.V0.copy()
                self.H1 = self.CH
                self.H2 = self.CH

        elif self.ch_fill_rule == "cloneHZ":
            if self.H1 is None and self.H2 is None:
                self.CH = self.V0.copy()
                self.H1 = self.CH
                self.H2 = self.CH
            elif self.H1 is None:
                self.H1 = self.H2
            elif self.H2 is None:
                self.H2 = self.H1
        # Kick error if ch_fill_rule got changed incorrectly later
        else:
            raise ValueError(f"ch_fill_rule {ch_fill_rule} invalid")

    def __str__(self):
        """string representation of a WindowMsg"""

        rstr = f"{len(self)} trace(s) in WindowMsg | "
        rstr += f"target model: {self.model_code} | "
        rstr += f"channel fill rule: {self.ch_fill_rule} | "
        rstr += f"target S/R: {self.target_sr} Hz\n"
        # Vertical component trace display
        rstr += f"Vert:  {self.V0.__str__()} \n"
        # Horizontal component 1 trace display
        rstr += f"Hztl 1:{self.H1.__str__()}"
        # And ch_fill_rule annotations
        if self.H1.stats.channel[-1] == "0":
            rstr += " (blank)\n"
        elif self.H1.stats.channel == self.H2.stats.channel:
            rstr += " (twinned)\n"
        else:
            rstr += "\n"
        # Horizontal component 2 trace display
        rstr += f"Hztl 2:{self.H2.__str__()}"
        # And ch_fill_rule annotations
        if self.H2.stats.channel[-1] == "0":
            rstr += " (blank)\n"
        elif self.H1.stats.channel == self.H2.stats.channel:
            rstr += " (twinned)\n"
        else:
            rstr += "\n"
        return rstr

    def __repr__(self):
        """Short format representation of a WindowMsg"""
        rstr = self.__str__()
        return rstr

    def check_inst_meta_compat(self, c1, c2, tolsamp=3):
        """
        Check compatability of two traces supposedly from the
        same instrument for the same period of time

        :: INPUTS ::
        :param c1: [obspy.Trace] trace object 1
        :param c2: [obspy.Trace] trace object 2
        :param tolsamp: [int] sample tolerance at self.target_sr
                        sampling rate for differences in starttime
                        and endtime values for c1 and c2
        :: OUTPUT ::
        If compatable, this method returns bool output 'True'
        otherwise it raises a TypeError or ValueError with some additional
        information on the specific failed compatability test
        """
        if not isinstance(c1, Trace):
            raise TypeError("c1 is not type Trace")
        if not isinstance(c2, Trace):
            raise TypeError("c2 is not type Trace")
        # Get stats
        s1 = c1.stats
        s2 = c2.stats
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
            if abs(s1[_k] - s2[_k]) / self.target_sr > tolsamp:
                raise ValueError(f"difference in {_k}'s outside tolerance")
            else:
                pass

        return True

    def sync_interp(self, **kwargs):
        """
        Sync the starttimes and sampling rates of traces using
        the target_sr and ref_starttime attributes of this WindowMsg
        """
        # INTERPOLATION SECTION
        # If reference starttime matches vertical
        if self.V0.stats.starttime == self.ref_starttime:
            # If all sampling rates
            for _tr in self.traces:
                if _tr.stats.sampling_rate == self.target_sr:
                    if _tr.stats.starttime == self.ref_starttime:
                        continue
                    
                elif _tr.stats.sampling_rate < self.target_sr:
                    _tr.interpolate(self.target_sr, **kwargs)
                else:
                    _tr.filter('lowpass', freq= self.target_sr/2)
                    _tr.interpolate(self.target_sr, **kwargs)
        

        # TRIM/PAD SECTION

        for _tr in self.traces:
            if _tr.stats.starttime == self.ref_starttime:
                if _tr.stats.sampling_rate == self.target_sr:
                    pass
                elif _tr.stats.sampling_rate > self.target_sr:
                    _tr.filter('lowpass', freq=self.target_sr/2)
                    _tr.interpolate(self.target_sr)



    def to_array(self, fill_value=0, order='Z12'):
        


    def _example_processing(self):
