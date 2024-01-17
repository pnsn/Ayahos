"""
:module: wyrm.structures.rtbufftrace
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains the class definition for RtBuffTrace
    which is an adaptation of the obspy.realtime.rttrace.RtTrace
    class in ObsPy that provides further development of handling
    gappy, asyncronous data packet sequencing and fault tolerance
    against station acquisition configuration changes without having
    to re-initialize the RtBuffTrace object.
"""

from obspy import Trace, Stream, UTCDateTime
import numpy as np
from copy import deepcopy
import wyrm.util.input_compatability_checks as icc


class RtBuffTrace(Trace):
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

    def __init__(self, max_length=1.0, fill_value=None, method=1, interpolation_samples=-1):
        """
        Initialize an empty RtBuffTrace object

        :: INPUTS ::
        :param max_length: [int], [float], or [None]
                            maximum length of the buffer in seconds
                            with the oldest data being trimmed from
                            the trace (left trim) in the case where
                            max_length is not None
        :param fill_value: [None], [float], or [int]
                            fill_value to assign to all masked
                            arrays associated with this RtBuffTrace
        """
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
            raise ValueError('stream.merge method must be -1, 0, or 1 - see obspy.core.stream.Stream.merge')
        else:
            self.method = method

        # Compatability check for interpolation_samples:
        if not isinstance(interpolation_samples, int):
            raise TypeError('interpolation_samples must be type int - see obspy.core.stream.Stream.merge')
        else:
            self.interpolation_samples = interpolation_samples

        # Set holder for dtype preservation
        self.dtype = None
        # Set initial state of have_appended trace
        self.have_appended_trace = False
        # Initialize with an empty, masked trace
        super(RtBuffTrace, self).__init__(
            data=np.ma.masked_array([], mask=False, fill_value=self.fill_value),
            header=None,
        )

    def copy(self):
        """
        Create a deepcopy of this RtBuffTrace
        """
        return deepcopy(self)

    def __eq__(self, other):
        """
        Implement rich comparison of RtBuffTrace objects for "==" operator.

        Traces are the same if their data, stats, and mask arrays are the same
        """
        if not isinstance(other, RtBuffTrace):
            return False
        else:
            return super(RtBuffTrace, self).__eq__(other)

    def to_trace(self):
        """
        Return a deepcopy obspy.core.trace.Trace representation
        sof this RtBuffTrace object
        """
        tr = Trace(data=self.copy().data, header=self.copy().stats)
        return tr


    
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
            self._first_append(self, trace)
        # Otherwise conduct checks 
        else:
            # Get information on relative timing
            status, gap_sec = self._check_overlap_status(self, trace):
            # If trace is fully before or after current contents reset
            if status in ['leading','trailing']:
                # If the gap is larger than max_length, run as _first_append
                if gap_sec > self.max_length:
                    # Do cursory check that NSLC matches
                    if self.id == trace.id:
                        self._first_append(self, trace)
                    else:
                        raise ValueError(f'id of self ({self.id}) and trace ({trace.id}) mismatch')
                # If the gap is smaller than max_length, run as _gappy_append
                else:
                    self._check_trace_compatability(self, trace)
                    self._gappy_append(self, trace, method=1)
            elif status == 'overlapping':
                self._check_trace_compatability(self, trace)
                # If either self or trace data are masked, treat as gappy
                if np.ma.is_masked(self.data) or np.ma.is_masked(trace.data):
                    self._gappy_append(self, trace)
                # Otherwise, conduct as a contiguous append without gap safety checks
                # - saves compute time, but exposes 
                else:
                    self._contiguous_append(self, trace)
            else:
                raise TypeError(f'_check_overlap_status retured status of {status}: must be "leading", "lagging", or "overlapping"')
                
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
        """
        self.data = trace.copy().data
        self.stats = trace.copy().stats
        self._have_appended_trace = True
            
    def _gappy_append(self, trace):
        """
        Append data method for when gappy (masked) data are anticipated
        to be present due to 

        """
        # Create a trace representation
        tr_self = self.to_trace()
        # Put trace and tr_self into a stream
        st = Stream([tr_self, trace.copy()])
        # Conduct split to convert gappy data into contiguous, unmasked chunks
        st = st.split()
        # Use merge to stitch data back together
        st.merge(
            fill_value=self.fill_value,
            method=self.merge_method,
            interpolation_samples=self.interpolation_samples)
        # Check that one trace resulted rom merge
        if len(st) == 1:
            tr = st[0]
            tr = self._enforce_max_length(self, tr)
            self.data = tr.data
            self.stats = tr.stats
        else:
            emsg = f"RtBuffTrace({self.id}).append "
            emsg += f"produced {len(st)} distinct traces "
            emsg += "rather than 1 expected. Canceling append."
            print(emsg)   
    
    def _contiguous_append(self, trace):
        """
        Append data method for when contiguous data are anticipated
        """
        tr_self = self.to_trace()
        st = Stream([tr_self, trace.copy()])
        st.merge(
            fill_value=self.fill_value,
            method=self.method,
            interpolation_samples=self.interpolation_samples)
        if len(st) == 1:
            tr = st[0]
            tr = self._enforce_max_length(self, tr)
            self.data = tr.data
            self.stats = tr.stats
        else:
            emsg = f"RtBuffTrace({self.id}).append "
            emsg += f"produced {len(st)} distinct traces "
            emsg += "rather than 1 expected. Canceling append."
            print(emsg)

    def _enforce_max_length(self, trace):
        """
        Enforce maximum_length on target trace using
        the endtime of the trace as the reference datum
        and return a left-trimmed version of input trace
        if the length of the input trace exceeds maximum_length

        note: trim is conducted in-place, so data will be lost
        unless using trace.copy() as an input argument

        :: INPUT ::
        :param trace: [obspy.core.trace.Trace] input trace
        
        :: OUTPUT ::
        :return trace: [obspy.core.trace.Trace] left-trimmed trace
                    if trace is longer than self.max_length seconds,
                    otherwise return the unaltered trace.
        """
        if not isinstance(trace, Trace):
            raise TypeError('trace must be type obspy.core.trace.Trace')
        te = trace.stats.endtime
        sr = trace.stats.sampling_rate
        max_samp = int(self.max_length * sr + 0.5)
        ts = te - max_samp / sr
        if ts > trace.stats.starttime:
            trace._ltrim(
                ts,
                pad=True,
                nearest_sample=True,
                fill_value=self.fill_value,
            )
        else:
            pass
        return trace

    def _check_attrib_compatability(self, trace, attribute_string='id'):
        """
        Check compatability of an attribute using eval() statemetnts for
        self and trace:

        i.e., eval(f'self.{attribute_string}') != eval(f'trace.{attribute_string}'):
        Raises ValueError if mismatch
        returns True if match
        """
        self_val = eval(f'self.{attribute_string}')
        trace_val = eval(f'trace.{attribute_string}')
        if self_val != trace_val:
            emsg = f'attribute ".{attribute_string}" mismatches'
            emsg += f'for buffer ({self_val}) and candidate trace ({trace_val})'
            raise ValueError(emsg)
        else:
            return True
        
    def _check_trace_compatability(self, trace):
        """
        Check compatability of key attriubters of self and trace:
        .id
        .stats.calib
        .stats.sampling_rate
        .data.dtype

        if any mismatch, return False
        else, return True
        """
        if not isinstance(trace):
            raise TypeError('trace must be type obspy.core.trace.Trace')
        xc_attr = ['id','stats.calib','stats.sampling_rate','data.dtype']
        bool_list = []
        for _attr in xc_attr:
            try:
                bool_list.append(self._check_attrib_compatability(self,trace, attribute_string=_attr))
            except ValueError:
                bool_list.append(False)
        return all(bool_list)   

    def _assess_relative_timing(self, trace):
        """
        Assesss the relative start/endtimes of RtBuffTrace and candidate trace
        :: INPUT ::
        :param trace: [obspy.core.trace.Trace] candidate trace

        :: OUTPUT ::
        :return status: [str] 'leading' - trace is entirely before this RtBuffTrace
                              'trailing' - trace is entirely after this RtBuffTrace
                              'overlapping' - trace overlaps with this RtBuffTrace
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
            starts = 'before'
        # if trace starts during self
        elif sss <= tss <= sse:
            starts = 'during'
        # if trace starts after self ends
        elif sse < tss:
            starts = 'after'
        
        # If trace ends before self ends
        if tse < sse:
            ends = 'before'
        elif sss <= tse <= sse:
            ends = 'during'
        elif sse < tse:
            ends = 'after'
        
        if starts == ends == 'before': 
            status = 'leading'
            gap_sec = sss - tse
        elif starts == ends == 'after':
            status = 'trailing'
            gap_sec = tss - sse
        else:
            if starts is not None and ends is not None:
                status = 'overlapping'
                gap_sec = None
            else:
                status = None
                gap_sec = None
        return status, gap_sec    


    def get_unmasked_fraction(
        self, starttime=None, endtime=None, pad=True, wgt_taper_sec=0.0, wgt_taper_type="cosine"
    ):
        """
        Get the (weighted) fraction value of valid (unmasked)
        data in a buffer or a specified slice of the buffer

        :: INPUTS ::
        :param starttime: [None] or [UTCDateTime]
                        starttime passed to self.copy().trim()
        :param endtime [None] or [UTCDateTime]
                        endtime passed to self.copy().trim()
        :param pad: [bool] pad argument passed to self.copy().trim()
        :param wgt_taper_sec: [float] tapered weighting function length
                        for each end of (subset) trace in seconds
        :param wgt_taper_type: [str] type of taper to apply. Supported:
                    'cosine' - cosine taper
                    'step' - Step functions centered on the nearest sample to
                            starttime + wgt_taper_sec and endtime - wgt_taper_sec
        :: OUTPUT ::
        :return pct_val: [float] value \in [0,1] indicating the (weighted)
                            fraction of valid (non-masked) samples in the
                            (subset)buffer

        NOTE: The 'step' wgt_taper_type is likely useful when assessing data windows
        that will be subject to blinding (e.g., continuous, striding predictions
        with ML workflows like EQTransformer and PhaseNet). The 'cosine' wgt_taper_type
        is likely more appropriate for general assessments of data when targeted
        features are assumed to be window-centered.
        """

        # Compatability check for starttime
        if isinstance(starttime, UTCDateTime):
            ts = starttime
        elif isinstance(starttime, type(None)):
            ts = None
        else:
            raise TypeError("starttime must be UTCDateTime or None")
        # Compatability check for endtime
        if isinstance(endtime, UTCDateTime):
            te = endtime
        elif isinstance(endtime, type(None)):
            te = None
        else:
            raise TypeError("endtime must be UTCDateTime or None")
        # Cmopatability check for pad
        if not isinstance(pad, bool):
            raise TypeError("pad must be bool")
        else:
            pass
        # Compatability check for wgt_taper_sec
        if isinstance(wgt_taper_sec, float):
            pass
        elif isinstance(wgt_taper_sec, int):
            wgt_taper_sec = float(wgt_taper_sec)
        else:
            raise TypeError("wgt_taper_sec must be type float or int")
        # Compatability check for wgt_taper_type
        if not isinstance(wgt_taper_type, str):
            raise TypeError("wgt_taper_type must be type str")
        else:
            pass
        # Taper generation section
        if wgt_taper_sec <= 0:
            tsamp = 0
            tap = 1
        else:
            tsamp = int(wgt_taper_sec * self.stats.sampling_rate)
            if wgt_taper_type.lower() in ["cosine", "cos"]:
                tap = 0.5 * (1.0 + np.cos(np.linspace(np.pi, 2.0 * np.pi, tsamp)))
            elif wgt_taper_type.lower() in ["step", "h", "heaviside"]:
                tap = np.zeros(tsamp)
            else:
                raise ValueError(
                    f"wgt_taper_type {wgt_taper_type} not supported. Supported values: 'cosine','step'"
                )
        # Apply trim function with cleared arguments
        _tr = self.copy().trim(
            starttime=ts,
            endtime=te,
            pad=pad,
            nearest_sample=True,
            fill_value=self.fill_value,
        )
        # Assess percent valid (pct_val)
        # If unmasked slice of buffer --> 100% completeness
        if not np.ma.is_masked(_tr.data):
            pct_val = 1.0
        else:
            # Compose binary array from bool mask
            mask_array = _tr.data.mask
            if len(mask_array) != _tr.npts:
                raise TypeError(
                    "expected _tr.data.mask and _tr.data to have the same length"
                )
            # Unmasked values = 1
            bin_array = np.ones(_tr.stats.npts)
            # Masked values = 0
            bin_array[mask_array] = 0
            # If no tapered weighting function
            if tsamp == 0:
                pct_val = np.mean(bin_array)
            # If tapered weighting function
            elif tsamp > 0:
                # Compose weight array
                wt_array = np.ones(_tr.stats.npts)
                wt_array[:tsamp] *= tap
                wt_array[-tsamp:] *= tap[::-1]
                # Calculate weighted average
                num = np.sum(wt_array * bin_array)
                den = np.sum(wt_array)
                pct_val = num / den

        return pct_val

    def get_window_stats(
        self,
        starttime=None,
        endtime=None,
        pad=True,
        taper_sec=0.0,
        taper_type="cosine",
        vert_codes="Z3",
        hztl_codes="N1E2",
    ):
        """
        Produce a dictionary of key statistics / bool results for this rtbufftrace for a specified window

        """
        nsli, comp = self.get_nsli_c()
        if comp in vert_codes:
            stats = {"comp_type": "Vertical", "comp_code": comp}
        elif comp in hztl_codes:
            stats = {"comp_type": "Horizontal", "comp_code": comp}
        else:
            stats = {"comp_type": "Unassigned", "comp_code": comp}

        # Compatability check for starttime
        if isinstance(starttime, UTCDateTime):
            ts = starttime
        elif isinstance(starttime, type(None)):
            ts = self.stats.starttime
        else:
            raise TypeError("starttime must be UTCDateTime or None")

        stats.update({"starttime": ts})

        # Compatability check for endtime
        if isinstance(endtime, UTCDateTime):
            te = endtime
        elif isinstance(endtime, type(None)):
            te = self.stats.endtime
        else:
            raise TypeError("endtime must be UTCDateTime or None")

        stats.update({"endtime": te})

        # Cmopatability check for pad
        if not isinstance(pad, bool):
            raise TypeError("pad must be bool")
        else:
            pass
        # Compatability check for taper_sec
        if isinstance(taper_sec, float):
            pass
        elif isinstance(taper_sec, int):
            taper_sec = float(taper_sec)
        else:
            raise TypeError("taper_sec must be type float or int")
        # Compatability check for taper_type
        if not isinstance(taper_type, str):
            raise TypeError("taper_type must be type str")
        else:
            pass

        # Get percent valid
        pct_val = self.get_valid_pct(
            starttime=ts,
            endtime=te,
            pad=pad,
            taper_sec=taper_sec,
            taper_type=taper_type,
        )

        stats.update({"percent_valid": pct_val})

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

    def __str__(self, extended=True, disc=30, showkey=False):
        """
        Return short summary string of the current RtBuffTrace
        with options for displaying buffer status graphically

        :: INPUTS ::
        :param extended: [bool] show buffer status?

        ..see RtBuffTrace._display_buff_status() for
        :param disc: [int]
        :param showkey: [bool]
        """
        rstr = super().__str__()
        rstr += f" | masked {100.*(1. - self.get_valid_pct()):.0f}%"
        rstr += f" | max {self.max_length} sec"
        if extended:
            rstr += "\n"
            rstr += self._display_buff_status(disc=disc, showkey=showkey, asprint=False)
        return rstr

    def _display_buff_status(self, disc=100, showkey=False, asprint=True):
        """
        Display a grapic representation of the data validity
        """
        if not isinstance(disc, int):
            raise TypeError("disc must be type int")
        elif disc < 0:
            raise ValueError("disc must be positive")
        else:
            pass
        if not isinstance(showkey, bool):
            raise TypeError("showkey must be type bool")
        else:
            pass

        rstr = "Buffer |"
        dt = self.max_length / disc
        if dt < self.stats.delta:
            dt = self.stats.delta
            disc = int(self.max_length / dt)
        ts0 = self.stats.starttime
        te0 = self.stats.endtime
        for _i in range(disc):
            ts = self.stats.starttime + _i * dt
            te = self.stats.starttime + (_i + 1) * dt
            # If window ends before start of buffered data
            if te <= ts0:
                rstr += "_"
            # If window starts after end of buffered data
            elif te0 <= ts:
                rstr += "_"
            # Otherwise
            else:
                try:
                    _tr = self.copy().trim(starttime=ts, endtime=te, pad=True)
                except:
                    breakpoint()
                if not np.ma.is_masked(_tr.data):
                    rstr += "D"
                # If all masked
                elif all(_tr.data.mask):
                    rstr += "m"
                # If any masked
                elif any(_tr.data.mask):
                    rstr += "p"
                # If none masked
                elif not any(_tr.data.mask):
                    rstr += "D"
        rstr += "|"
        if showkey:
            rstr += "\n (D)ata (p)artial (m)asked (_)none"
        if asprint:
            print(rstr)
        else:
            return rstr




# def _fussy_append(self, trace, merge_method=1, merge_interp_samp=-1, verbose=True):
#         """
#         Earlier version of self.append() that has much more picky rules and syntax



#         Appends a Trace-like object to this RtBuffTrace
#         with a series of compatability checks

#         :: INPUTS ::
#         :param trace: [obspy.core.trace.Trace] or child-class
#                         Trace or Trace-inheriting class object
#                         to be appended to this RtBuffTrace
#         :param merge_method: [int] `method` kwarg input for
#                         obspy.core.stream.Stream.merge()
#         :param merge_interp_samp: [int] `interpolation_samples`
#                         kwarg input for Stream.merge()
#         :param verbose: [bool] print warning messages?
#         """
#         # Compatability check for trace
#         if not isinstance(trace, Trace):
#             raise TypeError("trace must be a Trace-like object")
#         else:
#             pass

#         # Enforce masked data array for candidate data packet
#         if not np.ma.is_masked(trace.data):
#             trace.data = np.ma.masked_array(
#                 trace.data, fill_value=self.fill_value, mask=[False] * trace.stats.npts
#             )
#         elif trace.data.fill_value != self.fill_value:
#             trace.data.fill_value = self.fill_value
#         else:
#             pass

#         # If first append to RtBuffTrace, scrape data from input trace
#         if not self.have_appended_trace:
#             self.data = trace.data
#             self.dtype = trace.data.dtype
#             self.stats = trace.stats
#             self.have_appended_trace = True
#             if not np.ma.is_masked(self.data):
#                 self.data = np.ma.masked_array(
#                     self.data, mask=[False] * len(self), fill_value=self.fill_value
#                 )

#         elif self.have_appended_trace:
#             # IN ALL CASES, ID MUST MATCH
#             # ID check
#             if self.id != trace.id:
#                 raise TypeError(f"Trace ID differs: {self.id} vs {trace.id}")
#             else:
#                 pass

#             # # Time Checks # #
#             # Alias starttimes and endtimes for convenience
#             ts0 = self.stats.starttime
#             te0 = self.stats.endtime
#             ts1 = trace.stats.starttime
#             te1 = trace.stats.starttime
#             # Check for overlap
#             # ... if trace starttime after self starttime
#             if ts0 <= ts1 <= te0:
#                 starts_inside = True
#             else:
#                 starts_inside = False
#             # ... if trace endtime before self endtime
#             if ts0 <= te1 <= te0:
#                 ends_inside = True
#             else:
#                 ends_inside = False

#             if any([starts_inside, ends_inside]):
#                 overlapping = True
#             else:
#                 overlapping = False

#             # Check for too-large data gap (oversize_gap)
#             # to prevent generation of temporary, large masked arrays in memory
#             # if a given station was offline for awhile and then comes back
#             # online, potentially with a different digitization configuration.
#             # This should also enable adaptation to digitizer configuration
#             # changes while data are streaming within ~max_length sec.
#             if not overlapping and self.max_length is not None:
#                 gap_size = ts1 - te0
#                 if gap_size < 0:
#                     if gap_size < -1.0 * self.max_length:
#                         oversize_gap = True
#                         predates = True
#                     else:
#                         oversize_gap = False
#                         predates = True

#                 elif 0 <= gap_size:
#                     if gap_size < self.max_length:
#                         oversize_gap = False
#                         predates = False
#                     else:
#                         oversize_gap = True
#                         predates = False

#             else:
#                 oversize_gap = False
#                 predates = False

#             # S/R check
#             if self.stats.sampling_rate != trace.stats.sampling_rate:
#                 emsg = f"{self.id} | Sampling rate differs "
#                 emsg += f"{self.stats.sampling_rate} vs "
#                 emsg += f"{trace.stats.sampling_rate}"
#                 if not oversize_gap:
#                     raise TypeError(emsg)
#                 elif oversize_gap and not predates:
#                     emsg += f"after oversize gap: {gap_size:.3f} sec"
#                     emsg += f" (> {self.max_length} sec max_length)"
#                     if verbose:
#                         print(emsg)
#                 elif oversize_gap and predates:
#                     emsg += "preceeding RtBuffTrace with gap of "
#                     emsg += f"{gap_size:.3f} sec"
#                     if verbose:
#                         print(emsg)
#             else:
#                 pass

#             # Calib check
#             if self.stats.calib != trace.stats.calib:
#                 emsg = f"{self.id} | Calibration factor differs"
#                 emsg += f"{self.stats.calib} vs {trace.stats.calib}"
#                 if not oversize_gap:
#                     raise TypeError(emsg)
#                 elif oversize_gap and not predates:
#                     emsg += f"after oversize gap: {gap_size:.3f} sec"
#                     emsg += f" (> {self.max_length} sec max_length)"
#                     if verbose:
#                         print(emsg)
#                 elif oversize_gap and predates:
#                     emsg += "preceeding RtBuffTrace with gap of "
#                     emsg += f"{gap_size:.3f} sec"
#                     if verbose:
#                         print(emsg)
#             else:
#                 pass

#             # Check dtype
#             if self.data.dtype != trace.data.dtype:
#                 emsg = f"{self.id} | Data type differs "
#                 emsg += f"{self.data.dtype} vs {trace.data.dtype}"
#                 if not oversize_gap:
#                     raise TypeError(emsg)
#                 elif oversize_gap and not predates:
#                     emsg += f"after oversize gap: {gap_size:.3f} sec"
#                     emsg += f" (> {self.max_length} sec max_length)"
#                     if verbose:
#                         print(emsg)
#                 elif oversize_gap and predates:
#                     emsg += "preceeding RtBuffTrace with gap of "
#                     emsg += "{gap_size:.3f} sec"
#                     if verbose:
#                         print(emsg)
#             else:
#                 pass

#             # PROCESSING FOR OVERSIZE_GAP APPEND FROM PAST DATA
#             if oversize_gap and predates:
#                 emsg = f"{self.id} | trace significantly pre-dates "
#                 emsg += "data contained in this RtBufftrace.\n"
#                 emsg += "Assuming timing is bad on candidate packet.\n"
#                 emsg += f"tr({trace.stats.endtime}) -> "
#                 emsg += f"RtBuffTrace({self.stats.starttime}).\n"
#                 emsg += "Canceling append"
#                 if verbose:
#                     print(emsg)

#             # PROCESSING FOR OVERSIZE_GAP APPEND FROM FUTURE DATA
#             elif oversize_gap and not predates:
#                 # read in data from trace
#                 self.data = trace.data
#                 # get new stats information
#                 self.stats = trace.stats
#                 # back-pad data to max_length
#                 sr = self.stats.sampling_rate
#                 te = self.stats.endtime
#                 max_samp = int(self.max_length * sr + 0.5)
#                 ts = te - max_samp / sr
#                 self._ltrim(
#                     ts,
#                     pad=True,
#                     nearest_sample=True,
#                     fill_value=self.fill_value,
#                 )

#                 if not np.ma.is_masked(self.data):
#                     self.data = np.ma.masked_array(
#                         self.data, mask=[False] * len(self), fill_value=self.fill_value
#                     )
#                 # Allows for changes in data precision
#                 self.dtype = trace.data.dtype

#             # PROCESSING FOR NON-OVERSIZE_GAP APPEND
#             # ALLOWING FOR ASYNCHRONOUS LOADING WITHIN BOUNDS
#             # I.e., predates is not used as a descriminator here.
#             elif not oversize_gap:
#                 # # Use Merge to combine traces # #
#                 # Create Trace representation of self
#                 tr = Trace(data=deepcopy(self.data), header=deepcopy(self.stats))
#                 # Place self and trace into a stream
#                 st = Stream([tr, trace.copy()])
#                 # if overlap check flagged an overlap, run split
#                 # NOTE: split helps gracefully merge in delayed packets
#                 # in the case that the arrive in asynchronously.
#                 # NOTE: This behavior becomes useful in reconstituting ML
#                 # outputs into traces.
#                 if overlapping:
#                     st = st.split()
#                 # Merge in either case (overlapping or not overlapping)
#                 st.merge(
#                     fill_value=self.fill_value,
#                     method=merge_method,
#                     interpolation_samples=merge_interp_samp,
#                 )
#                 # Check that one trace resulted from merge...
#                 if len(st) == 1:
#                     tr = st[0]
#                 # ...raise a warning if it didnt and cancel append
#                 else:
#                     emsg = f"RtBuffTrace({self.id}).append "
#                     emsg += f"produced {len(st)} distinct traces "
#                     emsg += "rather than 1 expected. Canceling append."
#                     print(emsg)

#                 # # Use _ltrim to enforce maximum buffer length (if needed) # #
#                 if self.max_length is not None:
#                     # Get metadata
#                     merge_samples = len(tr.data)
#                     sr = tr.stats.sampling_rate
#                     max_samples = int(self.max_length * self.stats.sampling_rate + 0.5)
#                     # Assess if buffer at capacity
#                     if merge_samples > max_samples:
#                         ltst = tr.stats.starttime + (merge_samples - max_samples) / sr
#                         tr._ltrim(
#                             ltst,
#                             pad=True,
#                             nearest_sample=True,
#                             fill_value=self.fill_value,
#                         )

#                 # Enforce masked array if merge unmasked data
#                 if not np.ma.is_masked(tr.data):
#                     self.data = np.ma.masked_array(
#                         tr.data,
#                         mask=[False] * tr.stats.npts,
#                         fill_value=self.fill_value,
#                     )
#                 # Otherwise transfer from tr to self as-is
#                 else:
#                     self.data = tr.data
#                     # ... with a check that fill_value is preserved
#                     if tr.data.fill_value != self.fill_value:
#                         self.data.fill_value = self.fill_value
#                     # Explicit pass for completeness if everything checks out
#                     else:
#                         pass
#             # END not oversize_gap
#         # END self.have_appended_trace

#     # END of RtBuffTrace.append()