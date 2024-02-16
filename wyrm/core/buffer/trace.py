"""
:module: wyrm.buffer.trace
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains the class definition for TraceBuff
    which is an adaptation of the obspy.realtime.rttrace.RtTrace
    class in ObsPy that provides further development of handling
    gappy, asyncronous data packet sequencing and fault tolerance
    against station acquisition configuration changes without having
    to re-initialize the TraceBuff object.
"""

from obspy import Trace, Stream
import numpy as np
from copy import deepcopy
import wyrm.util.input_compatability_checks as icc


class TraceBuff(Trace):
    """
    Adapted version of the obspy.realtime.rttrace.RtTrace class
    that provides handling for gappy data that is not developed
    in the current release of obspy (1.4.0).

    A key difference in this class compared to the RtTrace class
    is that all trace.data attributes are handled as numpy.ma.masked_array
    objects. The data.mask sub-attribute is used to validate
    window completeness in processing steps leading into ML input-data
    window production.

    """

    def __init__(
        self,
        max_length=1.0,
        fill_value=None,
        method=1,
        interpolation_samples=-1
    ):
        """
        Initialize an empty TraceBuff object

        :: INPUTS ::
        :param max_length: [int], [float], or [None]
                            maximum length of the buffer in seconds
                            with the oldest data being trimmed from
                            the trace (left trim) in the case where
                            max_length is not None
        :param fill_value: [None], [float], or [int]
                            fill_value to assign to all masked
                            arrays associated with this TraceBuff
        :param method: method kwarg to pass to obspy.core.stream.Stream.merge()
                        internal to the TraceBuff.append() method
                        NOTE: Default here (method=1) is different from the
                        default for Stream.merge() (method=0). This default
                        was chosen such that overlapping data are handled
                        with interpolation (method=1), rather than gap generation
                        (method=0).
        :param interpolation_samples: interpolation_samples kwarg to pass to
                        obspy.core.stream.Stream.merge() internal to
                        the TraceBuff.append() method.
                        NOTE: Default here (-1) is different form Stream.merge()
                        (0). This was chosen such that all overlapping samples
                        are included in an interpolation in an attempt to suppress
                        abrupt steps in traces
        """
        super(TraceBuff, self).__init__()
        # Compatability check for max_length
        self.max_length = icc.bounded_floatlike(
            max_length, name="max_length", minimum=0.0, maximum=None, inclusive=False
        )

        # Compatability check for fill_value
        if not isinstance(fill_value, (int, float, type(None))):
            raise TypeError("fill_value must be type int, float, or None")
        else:
            self.fill_value = fill_value

        # Compatability check for method
        if method not in [0, 1, -1]:
            raise ValueError(
                "stream.merge method must be -1, 0, or 1 - see obspy.core.stream.Stream.merge"
            )
        else:
            self.method = method

        # Compatability check for interpolation_samples:
        self.interpolation_samples = icc.bounded_intlike(
            interpolation_samples,
            minimum=-1,
            maximum=self.max_length * 1e6,
            inclusive=True,
        )
        if not isinstance(interpolation_samples, int):
            raise TypeError(
                "interpolation_samples must be type int - see obspy.core.stream.Stream.merge"
            )
        else:
            self.interpolation_samples = interpolation_samples

        # Set initial state of have_appended trace
        self._have_appended_trace = False
        # Initialize buffer statistics parameters
        self.filled_fraction = 0
        self.valid_fraction = 1

    def copy(self):
        """
        Create a deepcopy of this TraceBuff
        """
        return deepcopy(self)

    def __eq__(self, other):
        """
        Implement rich comparison of TraceBuff objects for "==" operator.

        Traces are the same if their data, stats, and mask arrays are the same
        """
        if not isinstance(other, TraceBuff):
            return False
        else:
            return super(TraceBuff, self).__eq__(other)

    def to_trace(self):
        """
        Return a deepcopy obspy.core.trace.Trace representation
        sof this TraceBuff object
        """
        tr = Trace(data=self.copy().data, header=self.copy().stats)
        return tr

    def enforce_max_length(self):
        """
        Enforce max_length on data in this TraceBuff using
        the endtime of the TraceBuff as the reference datum
        and trimming using _ltrim (left-trim) if the data length
        exceeds self.max_length in seconds

        note: trim is conducted in-place, so trimmed-off data will
        be lost
        """
        te = self.stats.endtime
        sr = self.stats.sampling_rate
        max_samp = int(self.max_length * sr + 0.5)
        ts = te - max_samp / sr
        if ts > self.stats.starttime:
            self._ltrim(
                ts,
                pad=True,
                nearest_sample=True,
                fill_value=self.fill_value,
            )
        else:
            pass
        return self

    def append(self, trace):
        """
        Append a candidate trace to this RtTraceBuffer for a suite of scenarios:
            1) first ever append - self._first_append()

        IF id, stats.calib, stats.sampling_rate, data.dtype all match
            2) appending data with gaps - uses self._gappy_append()
            3) appending data without gaps - uses self._contiguous_append()
            takes away extra processing steps used to handle gaps, saving
            processing time relative to self._gappy_append().

        IF id matches
            4) appending data with gaps larger than self.max_length - uses self._first_append()
                This allows for restart of buffers in the event of data outages or
                changes in acquisition parameters without crashing the module.

        :: INPUT ::
        :param trace: [obspy.core.trace.Trace] candidate trace to append
        """
        # Compatability check that input trace is an obspy trace
        if not isinstance(trace, Trace):
            raise TypeError('input "trace" must be type obspy.core.trace.Trace')
        # If this is truly the first append to buffer, proceed directly to append
        if not self._have_appended_trace:
            self._first_append(trace)
        # Otherwise conduct checks
        else:
            # Get information on relative timing
            status, gap_sec = self._assess_relative_timing(trace)
            # If trace is fully before or after current contents reset
            if status in ["leading", "trailing"]:
                # If the gap is larger than max_length, run as _first_append
                if gap_sec > self.max_length:
                    # Do cursory check that NSLC matches
                    if self.id == trace.id:
                        self._have_appended_trace = False
                        self._first_append(trace)
                    else:
                        emsg = f"id of self ({self.id}) and trace ({trace.id}) mismatch"
                        raise ValueError(emsg)
                # If the gap is smaller than max_length, run as _gappy_append
                else:
                    self.check_trace_compatability(trace)
                    self._gappy_append(trace, method=1)
            elif status == "overlapping":
                self.check_trace_compatability(trace)
                # If either self or trace data are masked, treat as gappy
                if np.ma.is_masked(self.data) or np.ma.is_masked(trace.data):
                    self._gappy_append(trace)
                # Otherwise, conduct as a contiguous append without gap safety checks
                # - saves compute time, but exposes
                else:
                    self._contiguous_append(trace)
            else:
                raise TypeError(
                    f'_check_overlap_status retured status of {status}: must be "leading", "lagging", or "overlapping"'
                )
        return self

    # ################## #
    # Append Subroutines #
    # ################## #
    def _first_append(self, trace):
        """
        Special case append where data appended can exceed the self.max_length
        limit placed on all subsequent appends to this buffer. Data and Stats
        are copied from trace and data.

        :: INPUT ::
        :param trace: [obspy.core.trace.Trace] trace object to append

        :: ATTRIBUTES USED ::
        :attrib data: updated with trace.data
        :attrib stats: updated with trace.stats
        :attrib _have_appended_trace: update to True
        """
        if self._have_appended_trace:
            raise AttributeError(
                "Cannot execute _first_append() with self._have_appended_trace == True"
            )
        if not isinstance(trace, Trace):
            raise TypeError("trace must be type obspy.Trace")
        self.data = trace.copy().data
        self.stats = trace.copy().stats
        self._have_appended_trace = True
        self._update_buffer_stats()
        return self

    def _contiguous_append(self, trace):
        """
        Append data method for when contiguous data are anticipated
        """
        if not isinstance(trace, Trace):
            raise TypeError("trace must be type obspy.Trace")
        tr_self = self.to_trace()
        st = Stream([tr_self, trace.copy()])
        st.merge(
            fill_value=self.fill_value,
            method=self.method,
            interpolation_samples=self.interpolation_samples,
        )
        if len(st) == 1:
            tr = st[0]
            self.data = tr.data
            self.stats = tr.stats
            self.enforce_max_length()._update_buffer_stats()
        else:
            emsg = f"TraceBuff({self.id}).append "
            emsg += f"produced {len(st)} distinct traces "
            emsg += "rather than 1 expected. Canceling append."
            print(emsg)
        return self

    def _gappy_append(self, trace):
        """
        Append data method for when gappy (masked) data are anticipated
        to be present due to self or trace having gaps or the existance of
        a gap between self and trace data. This private method includes a
        split() method internally to create a stream of contiguous data
        segments and then merge to re-merge the data into (semi) contiguous
        data. This also allows for nonsequential packet loading (i.e., gap
        back-filling) that could occur in some marginal cases.

        This added split() requires additional compute resources, but is safe
        for both gappy and contiguous data.

        :: INPUT ::
        :param trace: [obspy.core.trace.Trace] candidate trace to append

        :: ATTRIBUTES USED ::
        :attrib data: TraceBuff data array
        :attrib stats: TraceBuff metadata
        :attrib method: kwarg value passed to obspy.Stream.merge()
        :attrib interpolation_samples: kwarg value passed to obspy.Stream.merge
        """
        if not isinstance(trace, Trace):
            raise TypeError("trace must be type obspy.Trace")
        # Create a trace representation
        tr_self = self.to_trace()
        # Put trace and tr_self into a stream
        st = Stream([tr_self, trace.copy()])
        # Conduct split to convert gappy data into contiguous, unmasked chunks
        st = st.split()
        # Use merge to stitch data back together
        st.merge(
            fill_value=self.fill_value,
            method=self.method,
            interpolation_samples=self.interpolation_samples,
        )
        # Check that one trace resulted rom merge
        if len(st) == 1:
            tr = st[0]
            self.data = tr.data
            self.stats = tr.stats
            self.enforce_max_length()._update_buffer_stats()
        else:
            emsg = f"TraceBuff({self.id}).append "
            emsg += f"produced {len(st)} distinct traces "
            emsg += "rather than 1 expected. Canceling append."
            print(emsg)
        return self

    def _assess_relative_timing(self, trace):
        """
        Assesss the relative start/endtimes of TraceBuff and candidate trace
        :: INPUT ::
        :param trace: [obspy.core.trace.Trace] candidate trace

        :: OUTPUT ::
        :return status: [str] 'leading' - trace is entirely before this TraceBuff
                              'trailing' - trace is entirely after this TraceBuff
                              'overlapping' - trace overlaps with this TraceBuff
                              None - in the case where something goes wrong with
                                    types of timing information
        :return gap_sec: [float] - in the case of 'leading' and 'trailing' this is
                                    the length of the gap in seconds
                                   in the case of 'overlapping', defaults to None
        """
        starts = None
        ends = None
        if not isinstance(trace, Trace):
            raise TypeError('input "trace" must be type obspy.core.trace.Trace')
        sss = self.stats.starttime
        sse = self.stats.endtime
        tss = trace.stats.starttime
        tse = trace.stats.endtime
        # if trace starts before self starts
        if tss < sss:
            starts = "before"
        # if trace starts during self
        elif sss <= tss <= sse:
            starts = "during"
        # if trace starts after self ends
        elif sse < tss:
            starts = "after"

        # If trace ends before self ends
        if tse < sse:
            ends = "before"
        elif sss <= tse <= sse:
            ends = "during"
        elif sse < tse:
            ends = "after"

        if starts == ends == "before":
            status = "leading"
            gap_sec = sss - tse
        elif starts == ends == "after":
            status = "trailing"
            gap_sec = tss - sse
        else:
            if starts is not None and ends is not None:
                status = "overlapping"
                gap_sec = None
            else:
                status = None
                gap_sec = None
        return status, gap_sec

    # ########################################## #
    # Trace Metadata / Dtype Compatability Check #
    # ########################################## #

    def check_trace_compatability(self, trace):
        """
        Check compatability of key attriubters of self and trace:
        .id
        .stats.calib
        .stats.sampling_rate
        .data.dtype

        if any mismatch, return False
        else, return True
        """
        if not isinstance(trace, Trace):
            raise TypeError("trace must be type obspy.core.trace.Trace")
        xc_attr = ["id", "stats.calib", "stats.sampling_rate", "data.dtype"]
        bool_list = []
        for _attr in xc_attr:
            try:
                bool_list.append(
                    self._check_attribute_compatability(trace, attr_str=_attr)
                )
            except ValueError:
                bool_list.append(False)
        return all(bool_list)

    def _check_attribute_compatability(self, trace, attr_str="id"):
        """
        Check compatability of an attribute using eval() statements for
        self and trace:

        i.e., eval(f'self.{attr_str}') != eval(f'trace.{attr_str}'):
        Raises ValueError if mismatch
        returns True if match
        """
        self_val = eval(f"self.{attr_str}")
        trace_val = eval(f"trace.{attr_str}")
        if self_val != trace_val:
            emsg = f'attribute ".{attr_str}" mismatches'
            emsg += f"for buffer ({self_val}) and candidate trace ({trace_val})"
            raise ValueError(emsg)
        else:
            return True

    #####################################
    # Data completeness check & display #
    #####################################
    def _update_buffer_stats(self):
        """
        Update the self.filled_fraction and self.valid_fraction
        attributes of this TraceBuff with its current contents
        """
        self.filled_fraction = self.get_filled_fraction()
        self.valid_fraction = self.get_unmasked_fraction()
        return self

    def get_filled_fraction(self):
        """
        Get the fractional amount of data (masked and unmasked)
        present in this TraceBuff relative to self.max_length
        """
        ffnum = self.stats.endtime - self.stats.starttime
        ffden = self.max_length
        return ffnum / ffden

    def get_trimmed_valid_fraction(
        self, starttime=None, endtime=None, wgt_taper_sec=0.0, wgt_taper_type="cosine"
    ):
        """
        Get the valid (unmasked) fraction of data contained within a specified
        time window, with the option of introducing downweighting functions at
        the edges of the candidate window.

        :: INPUTS ::
        :param starttime: [None] or [obspy.UTCDateTime] starttime to pass
                            to obspy.Trace.trim. None input uses starttime
                            of this TraceBuff
        :param endtime: [None] or [obspy.UTCDateTime] endtime to pass
                            to obspy.Trace.trim. None input uses endtime
                            of this TraceBuff
                            NOTE: trim operations are conducted on a temporary
                            copy of this TraceBuff and the trim uses pad=True
                            and fill_value=None to generate masked samples
                            if the specified start/endtimes lie outside the
                            bounds of data in this TraceBuff
        :param wgt_taper_sec: [float] non-negative, finite number of seconds
                            that a specified taper function should last on
                            each end of the candidate window. Value of 0
                            results in no taper applied.
        :param wgt_taper_type: [str] name of taper type to apply
                        Supported values:
                        "cosine" - apply a cosine taper to the data weighting
                                   function
                            aliases: 'cos', 'tukey'
                        "step" - set all data weights in the tapered 
                                 region to 0
                            aliases: 'h', 'heaviside'
        :: OUTPUT ::
        :return vf: [float] (weighted) fraction of non-masked data in the
                    candidate window. vf \in [0, 1]
        """

        tmp_copy = self.copy()
        tmp_copy.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=None)
        vf = tmp_copy.get_unmasked_fraction(
            wgt_taper_sec=wgt_taper_sec, wgt_taper_type=wgt_taper_type
        )
        return vf

    def get_unmasked_fraction(self, wgt_taper_sec=0.0, wgt_taper_type="cosine"):
        """
        Get the fractional amount of unmasked data present in
        this TraceBuff, or a specified time slice of the data.
        Options are provided to introduce a tapered down-weighting
        of data on either end of the buffer or specified slice of
        the data

        :: INPUTS ::
        :param wgt_taper_sec: [float] tapered weighting function length
                        for each end of (subset) trace in seconds on either
                        end of the trace.
        :param wgt_taper_type: [str] type of taper to apply for data weighting.
                    Supported Values:
                    'cosine' - cosine taper (Default)
                        aliases: 'cos', 'tukey' (case insensitive)
                    'step' - Step functions centered on the nearest sample to
                            starttime + wgt_taper_sec and endtime - wgt_taper_sec
                        aliases: 'heaviside', 'h' (case insensitive)
        :: OUTPUT ::
        :return valid_frac: [float] value \in [0,1] indicating the (weighted)
                            fraction of valid (non-masked) samples in the
                            (subset)buffer

        NOTE: The 'step' wgt_taper_type is likely useful when assessing data windows
        that will be subject to blinding (e.g., continuous, striding predictions
        with ML workflows like EQTransformer and PhaseNet). The 'cosine' wgt_taper_type
        is likely more appropriate for general assessments of data when targeted
        features are assumed to be window-centered.
        """
        # Handle empty buffer case
        if self.stats.npts == 0:
            valid_frac = 1.0
        else:
            # Compatability check for wgt_taper_sec
            wgt_taper_sec = icc.bounded_floatlike(
                wgt_taper_sec,
                name="wgt_taper_sec",
                minimum=0,
                maximum=self.stats.endtime - self.stats.starttime,
            )
            # Compatability check for wgt_taper_type
            if not isinstance(wgt_taper_type, str):
                raise TypeError("wgt_taper_type must be type str")
            elif wgt_taper_type.lower() in [
                "cosine",
                "cos",
                "tukey",
                "step",
                "h",
                "heaviside",
            ]:
                pass
            else:
                raise ValueError(
                    'wgt_taper_types supported: "cosine", "step", and select aliases'
                )

            # Taper generation section
            # Zero-length taper
            if wgt_taper_sec == 0:
                tsamp = 0
                tap = 1
            # Non-zero-length taper
            else:
                tsamp = int(wgt_taper_sec * self.stats.sampling_rate)
                # Cosine taper
                if wgt_taper_type.lower() in ["cosine", "cos", "tukey"]:
                    tap = 0.5 * (1.0 + np.cos(np.linspace(np.pi, 2.0 * np.pi, tsamp)))
                # Step taper
                elif wgt_taper_type.lower() in ["step", "h", "heaviside"]:
                    tap = np.zeros(tsamp)
            # Assess percent valid (valid_frac)
            # If unmasked data --> 100% completeness
            if not np.ma.is_masked(self.data):
                valid_frac = 1.0
            else:
                # Unmasked values = 1
                bin_array = np.ones(self.stats.npts)
                # Masked values = 0
                bin_array[self.data.mask] = 0
                # If no tapered weighting function
                if tsamp == 0:
                    valid_frac = np.mean(bin_array)
                # If tapered weighting function
                elif tsamp > 0:
                    # Compose weight array
                    wt_array = np.ones(self.stats.npts)
                    wt_array[:tsamp] *= tap
                    wt_array[-tsamp:] *= tap[::-1]
                    # Calculate weighted average
                    num = np.sum(wt_array * bin_array)
                    den = np.sum(wt_array)
                    valid_frac = num / den
        return valid_frac

    # ##################### #
    # Display class methods #
    # ##################### #

    def get_sncl(self):
        """
        Fetch string for
        Sta.Net.Chan.Loc
        """
        sta = self.stats.station
        net = self.stats.network
        cha = self.stats.channel
        loc = self.stats.location
        sncl = f"{sta}.{net}.{cha}.{loc}"
        return sncl

    def get_snil_c(self):
        """
        Fetch strings for tuple
        ('Sta.Net.Inst.Loc', 'Comp')
        """
        sta = self.stats.station
        net = self.stats.network
        cha = self.stats.channel
        inst = cha[:-1]
        comp = cha[-1]
        loc = self.stats.location
        snil = f"{sta}.{net}.{inst}.{loc}"
        return (snil, comp)

    def get_nsli_c(self):
        """
        Fetch strings for tuple
        ('Net.Sta.Loc.Inst', 'Comp')
        """
        sta = self.stats.station
        net = self.stats.network
        cha = self.stats.channel
        inst = cha[:-1]
        comp = cha[-1]
        loc = self.stats.location
        nsli = f"{net}.{sta}.{loc}.{inst}"
        return (nsli, comp)

    def __repr__(self, compact=False):
        """
        Return short summary string of the current TraceBuff
        with options for displaying buffer status graphically

        :: INPUT ::
        :param compact: [bool] show a compact representation of buffer fill fraction
                        and masked fraction?
        :: OUTPUT ::
        :return rstr: [str] representative string
        """
        # If non-compact representation 
        if not compact:
            # Start Trace-inherited __str__ 
            rstr = f"{super().__str__()}"
            # Add buffer, mask, and max length metrics in pretty formatting
            rstr += f" | buffer {100.*(self.filled_fraction):.0f}%"
            rstr += f" | masked {100.*(1. - self.valid_fraction):.0f}%"
            rstr += f" | max {self.max_length} sec"
        # If compact representation (used with TieredBuffer)
        else:
            rstr = f"B:{self.filled_fraction: .1f}|M:{(1. - self.valid_fraction):.1f}"
        return rstr

    def __str__(self):
        """
        Return a repr string for this TraceBuff
        """
        rstr = 'wyrm.buffer.trace.TraceBuff('
        rstr += f'max_length={self.max_length}, fill_value={self.fill_value}, '
        rstr += f'method={self.method}, interpolation_samples={self.interpolation_samples})'
        return rstr


