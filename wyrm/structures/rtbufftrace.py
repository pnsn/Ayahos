from obspy import Trace, Stream
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
        """ """
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

        self.dtype = None

        self.have_appended_trace = False
        super(RtBuffTrace, self).__init__(
            data=np.ma.masked_array([], mask=False, fill_value=self.fill_value),
            header=None,
        )

    def __eq__(self, other):
        """
        Implement rich comparison of RtBuffTrace objects for "==" operator.

        Traces are the same if their data, stats, and mask arrays are the same
        """
        if not isinstance(other, RtBuffTrace):
            return False
        else:
            return super(RtBuffTrace, self).__eq__(other)

    def append(
        self,
        trace,
        merge_method=1,
        merge_interp_samp=-1,
        verbose=True
    ):
        """
        Appends a Trace-like object to this RtBuffTrace
        with a series of compatability checks
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
                    esmg += f"preceeding RtBuffTrace with gap of {gap_size:.3f} sec"
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
                    esmg += f"preceeding RtBuffTrace with gap of {gap_size:.3f} sec"
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
                    esmg += f'preceeding RtBuffTrace with gap of {gap_size:.3f} sec'
                    if verbose:
                        print(emsg)
            else:
                pass

            # PROCESSING FOR OVERSIZE_GAP APPEND FROM PAST DATA
            if oversize_gap and predates:
                emsg = f"{self.id} | trace significantly pre-dates "
                emsg += f"data contained in this RtBufftrace "
                emsg += f"assuming timing is bad on candidate packet "
                emsg += f"tr({trace.stats.endtime}) -> "
                emsg += f"RtBuffTrace({self.stats.starttime}). "
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

    def copy(self, *args, **kwargs):
        """
        Returns a deepcopy of this RtBuffTrace
        """
        new = deepcopy(self, *args, **kwargs)
        return new
