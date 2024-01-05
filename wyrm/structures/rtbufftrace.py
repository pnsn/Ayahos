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

    def __init__(self, max_length=None, fill_value=None):
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
        if not isinstance(max_length, (int, float, type(None))):
            raise TypeError("max_length must be type int, float or None")
        elif max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive")
        else:
            self.max_length = max_length

        # Compatability check for fill_value
        if not isinstance(fill_value, (int, float, type(None))):
            raise TypeError("fill_value must be type int, float, or None")
        else:
            self.fill_value = fill_value
        # Set holder for dtype preservation
        self.dtype = None
        # Set initial state of have_appended trace
        self.have_appended_trace = False
        # Initialize with an empty, masked trace
        super(RtBuffTrace, self).__init__(
            data=np.ma.masked_array([], mask=False, fill_value=self.fill_value),
            header=None,
        )

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

    def __eq__(self, other):
        """
        Implement rich comparison of RtBuffTrace objects for "==" operator.

        Traces are the same if their data, stats, and mask arrays are the same
        """
        if not isinstance(other, RtBuffTrace):
            return False
        else:
            return super(RtBuffTrace, self).__eq__(other)

    def append(self, trace, merge_method=1, merge_interp_samp=-1, verbose=True):
        """
        Appends a Trace-like object to this RtBuffTrace
        with a series of compatability checks

        :: INPUTS ::
        :param trace: [obspy.core.trace.Trace] or child-class
                        Trace or Trace-inheriting class object
                        to be appended to this RtBuffTrace
        :param merge_method: [int] `method` kwarg input for
                        obspy.core.stream.Stream.merge()
        :param merge_interp_samp: [int] `interpolation_samples`
                        kwarg input for Stream.merge()
        :param verbose: [bool] print warning messages?
        """
        # Compatability check for trace
        if not isinstance(trace, Trace):
            raise TypeError("trace must be a Trace-like object")
        else:
            pass

        # Enforce masked data array for candidate data packet
        if not np.ma.is_masked(trace.data):
            trace.data = np.ma.masked_array(
                trace.data, fill_value=self.fill_value, mask=[False] * trace.stats.npts
            )
        elif trace.data.fill_value != self.fill_value:
            trace.data.fill_value = self.fill_value
        else:
            pass

        # If first append to RtBuffTrace, scrape data from input trace
        if not self.have_appended_trace:
            self.data = trace.data
            self.dtype = trace.data.dtype
            self.stats = trace.stats
            self.have_appended_trace = True
            if not np.ma.is_masked(self.data):
                self.data = np.ma.masked_array(
                    self.data, mask=[False] * len(self), fill_value=self.fill_value
                )

        elif self.have_appended_trace:
            # IN ALL CASES, ID MUST MATCH
            # ID check
            if self.id != trace.id:
                raise TypeError(f"Trace ID differs: {self.id} vs {trace.id}")
            else:
                pass

            # # Time Checks # #
            # Alias starttimes and endtimes for convenience
            ts0 = self.stats.starttime
            te0 = self.stats.endtime
            ts1 = trace.stats.starttime
            te1 = trace.stats.starttime
            # Check for overlap
            # ... if trace starttime after self starttime
            if ts0 <= ts1 <= te0:
                starts_inside = True
            else:
                starts_inside = False
            # ... if trace endtime before self endtime
            if ts0 <= te1 <= te0:
                ends_inside = True
            else:
                ends_inside = False

            if any([starts_inside, ends_inside]):
                overlapping = True
            else:
                overlapping = False

            # Check for too-large data gap (oversize_gap)
            # to prevent generation of temporary, large masked arrays in memory
            # if a given station was offline for awhile and then comes back
            # online, potentially with a different digitization configuration.
            # This should also enable adaptation to digitizer configuration
            # changes while data are streaming within ~max_length sec.
            if not overlapping and self.max_length is not None:
                gap_size = ts1 - te0
                if gap_size < 0:
                    if gap_size < -1.0 * self.max_length:
                        oversize_gap = True
                        predates = True
                    else:
                        oversize_gap = False
                        predates = True

                elif 0 <= gap_size:
                    if gap_size < self.max_length:
                        oversize_gap = False
                        predates = False
                    else:
                        oversize_gap = True
                        predates = False

            else:
                oversize_gap = False
                predates = False

            # S/R check
            if self.stats.sampling_rate != trace.stats.sampling_rate:
                emsg = f"{self.id} | Sampling rate differs "
                emsg += f"{self.stats.sampling_rate} vs "
                emsg += f"{trace.stats.sampling_rate}"
                if not oversize_gap:
                    raise TypeError(emsg)
                elif oversize_gap and not predates:
                    emsg += f"after oversize gap: {gap_size:.3f} sec"
                    emsg += f" (> {self.max_length} sec max_length)"
                    if verbose:
                        print(emsg)
                elif oversize_gap and predates:
                    emsg += "preceeding RtBuffTrace with gap of "
                    emsg += f"{gap_size:.3f} sec"
                    if verbose:
                        print(emsg)
            else:
                pass

            # Calib check
            if self.stats.calib != trace.stats.calib:
                emsg = f"{self.id} | Calibration factor differs"
                emsg += f"{self.stats.calib} vs {trace.stats.calib}"
                if not oversize_gap:
                    raise TypeError(emsg)
                elif oversize_gap and not predates:
                    emsg += f"after oversize gap: {gap_size:.3f} sec"
                    emsg += f" (> {self.max_length} sec max_length)"
                    if verbose:
                        print(emsg)
                elif oversize_gap and predates:
                    emsg += "preceeding RtBuffTrace with gap of "
                    emsg += f"{gap_size:.3f} sec"
                    if verbose:
                        print(emsg)
            else:
                pass

            # Check dtype
            if self.data.dtype != trace.data.dtype:
                emsg = f"{self.id} | Data type differs "
                emsg += f"{self.data.dtype} vs {trace.data.dtype}"
                if not oversize_gap:
                    raise TypeError(emsg)
                elif oversize_gap and not predates:
                    emsg += f"after oversize gap: {gap_size:.3f} sec"
                    emsg += f" (> {self.max_length} sec max_length)"
                    if verbose:
                        print(emsg)
                elif oversize_gap and predates:
                    emsg += "preceeding RtBuffTrace with gap of "
                    emsg += "{gap_size:.3f} sec"
                    if verbose:
                        print(emsg)
            else:
                pass

            # PROCESSING FOR OVERSIZE_GAP APPEND FROM PAST DATA
            if oversize_gap and predates:
                emsg = f"{self.id} | trace significantly pre-dates "
                emsg += "data contained in this RtBufftrace.\n"
                emsg += "Assuming timing is bad on candidate packet.\n"
                emsg += f"tr({trace.stats.endtime}) -> "
                emsg += f"RtBuffTrace({self.stats.starttime}).\n"
                emsg += "Canceling append"
                if verbose:
                    print(emsg)

            # PROCESSING FOR OVERSIZE_GAP APPEND FROM FUTURE DATA
            elif oversize_gap and not predates:
                # read in data from trace
                self.data = trace.data
                # get new stats information
                self.stats = trace.stats
                # back-pad data to max_length
                sr = self.stats.sampling_rate
                te = self.stats.endtime
                max_samp = int(self.max_length * sr + 0.5)
                ts = te - max_samp / sr
                self._ltrim(
                    ts,
                    pad=True,
                    nearest_sample=True,
                    fill_value=self.fill_value,
                )

                if not np.ma.is_masked(self.data):
                    self.data = np.ma.masked_array(
                        self.data, mask=[False] * len(self), fill_value=self.fill_value
                    )
                # Allows for changes in data precision
                self.dtype = trace.data.dtype

            # PROCESSING FOR NON-OVERSIZE_GAP APPEND
            # ALLOWING FOR ASYNCHRONOUS LOADING WITHIN BOUNDS
            # I.e., predates is not used as a descriminator here.
            elif not oversize_gap:
                # # Use Merge to combine traces # #
                # Create Trace representation of self
                tr = Trace(data=deepcopy(self.data), header=deepcopy(self.stats))
                # Place self and trace into a stream
                st = Stream([tr, trace.copy()])
                # if overlap check flagged an overlap, run split
                # NOTE: split helps gracefully merge in delayed packets
                # in the case that the arrive in asynchronously.
                # NOTE: This behavior becomes useful in reconstituting ML
                # outputs into traces.
                if overlapping:
                    st = st.split()
                # Merge in either case (overlapping or not overlapping)
                st.merge(
                    fill_value=self.fill_value,
                    method=merge_method,
                    interpolation_samples=merge_interp_samp,
                )
                # Check that one trace resulted from merge...
                if len(st) == 1:
                    tr = st[0]
                # ...raise a warning if it didnt and cancel append
                else:
                    emsg = f"RtBuffTrace({self.id}).append "
                    emsg += f"produced {len(st)} distinct traces "
                    emsg += "rather than 1 expected. Canceling append."
                    print(emsg)

                # # Use _ltrim to enforce maximum buffer length (if needed) # #
                if self.max_length is not None:
                    # Get metadata
                    merge_samples = len(tr.data)
                    sr = tr.stats.sampling_rate
                    max_samples = int(self.max_length * self.stats.sampling_rate + 0.5)
                    # Assess if buffer at capacity
                    if merge_samples > max_samples:
                        ltst = tr.stats.starttime + (merge_samples - max_samples) / sr
                        tr._ltrim(
                            ltst,
                            pad=True,
                            nearest_sample=True,
                            fill_value=self.fill_value,
                        )

                # Enforce masked array if merge unmasked data
                if not np.ma.is_masked(tr.data):
                    self.data = np.ma.masked_array(
                        tr.data,
                        mask=[False] * tr.stats.npts,
                        fill_value=self.fill_value,
                    )
                # Otherwise transfer from tr to self as-is
                else:
                    self.data = tr.data
                    # ... with a check that fill_value is preserved
                    if tr.data.fill_value != self.fill_value:
                        self.data.fill_value = self.fill_value
                    # Explicit pass for completeness if everything checks out
                    else:
                        pass
            # END not oversize_gap
        # END self.have_appended_trace

    # END of RtBuffTrace.append()

    # Copying and indexing-support methods

    def copy(self, *args, **kwargs):
        """
        Returns a deepcopy of this RtBuffTrace
        """
        new = deepcopy(self, *args, **kwargs)
        return new
    
    def as_trace(self):
        """
        returns a deepcopy of the data and header attributes
        as a plain-vanilla obspy.core.trace.Trace
        """
        new = Trace(data=self.copy().data, header=self.copy().header)
        return new

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

    def get_valid_pct(
        self, starttime=None, endtime=None, pad=True, taper_sec=0.0, taper_type="cosine"
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
        :param taper_sec: [float] tapered weighting function length
                        for each end of (subset) trace in seconds
        :param taper_type: [str] type of taper to apply. Supported:
                    'cosine' - cosine taper
                    'step' - Step functions centered on the nearest sample to
                            starttime + taper_sec and endtime - taper_sec
        :: OUTPUT ::
        :return pct_val: [float] value \in [0,1] indicating the (weighted)
                            fraction of valid (non-masked) samples in the
                            (subset)buffer

        NOTE: The 'step' taper_type is likely useful when assessing data windows
        that will be subject to blinding (e.g., continuous, striding predictions
        with ML workflows like EQTransformer and PhaseNet). The 'cosine' taper_type
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
        # Taper generation section
        if taper_sec <= 0:
            tsamp = 0
            tap = 1
        else:
            tsamp = int(taper_sec * self.stats.sampling_rate)
            if taper_type.lower() in ["cosine", "cos"]:
                tap = 0.5 * (1.0 + np.cos(np.linspace(np.pi, 2.0 * np.pi, tsamp)))
            elif taper_type.lower() in ["step", "h", "heaviside"]:
                tap = np.zeros(tsamp)
            else:
                raise ValueError(
                    f"taper_type {taper_type} not supported. Supported values: 'cosine','step'"
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
        vert_codes='Z3',
        hztl_codes='N1E2'
    ):
        """
        Produce a dictionary of key statistics / bool results for this rtbufftrace for a specified window
        """
        nsli, comp = self.get_nsli_c()
        if comp in vert_codes:
            stats = {'comp_type': 'Vertical', 'comp_code': comp}
        elif comp in hztl_codes:
            stats = {'comp_type': 'Horizontal', 'comp_code': comp}
        else:
            stats = {'comp_type': 'Unassigned', 'comp_code': comp}

        # Compatability check for starttime
        if isinstance(starttime, UTCDateTime):
            ts = starttime
        elif isinstance(starttime, type(None)):
            ts = self.stats.starttime
        else:
            raise TypeError("starttime must be UTCDateTime or None")

        stats.update({'starttime': ts})

        # Compatability check for endtime
        if isinstance(endtime, UTCDateTime):
            te = endtime
        elif isinstance(endtime, type(None)):
            te = self.stats.endtime
        else:
            raise TypeError("endtime must be UTCDateTime or None")

        stats.update({'endtime': te})        

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
        pct_val = self.get_valid_pct(starttime=ts, endtime=te, pad=pad, taper_sec=taper_sec, taper_type=taper_type)

        stats.update({'percent_valid': pct_val})