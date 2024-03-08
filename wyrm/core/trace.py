"""
:module: wyrm.core.trace
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module provides class definitions that extend ObsPy's
    Trace and Stats (in module obspy.core.trace) classes to add
    these key features:

    Trace -> MLTrace - adds a 'fold' vector that tracks the number
                of observations contributing to a given time-indexed
                sample. Details on 'fold' and updated/new methods
                are detailed in the class header documentation

                Methods for saving MLTraces as miniSEED files and
                reading these save files from disk are provided, built
                on top of the obspy.core.stream.Stream.write API

    Stats -> MLStats - adds the attribute/key values 'model' and
                'weight' to document the names of ML models and 
                pretrained weights during processing. These are
                also appended to the MLTrace.id property with
                the format:
                    N.S.L.C.{Model}.{Weight}
                Further details in the MLStats class header.
                
                Additionally, this class modifies the `info`
                appended to MLStats.processing into a 4-list
                comprising:
                    info = [timestamp, 'Package Version', 'Method Name', '(*args, **kwargs)']
                to capture information on data packets' processing times
"""

import inspect, time ,os
import numpy as np
from decorator import decorator
from copy import deepcopy
from obspy import Stream, read, UTCDateTime
from obspy.core.trace import Trace, Stats
from obspy.core.util.misc import flat_not_masked_contiguous
from wyrm.util.compatability import bounded_floatlike

###################################################################################
# Machine Learning Stats Class Definition #########################################
###################################################################################

class MLStats(Stats):
    """
    This child class of the obspy.core.trace.Stats class adds in trace
    ID labels 'model' and 'weight' to capture different ML model architecture
    and pretrained weight names in the course of data processing. It also
    adjusts the default 'location' value to '--' to conform with Earthworm
    conventions for no-location channel codes.
    """
    # set of read only attrs
    readonly = ['endtime']
    # add additional default values to obspy.core.trace.Stats's defaults
    defaults = Stats.defaults
    defaults.update({
        'location': '--',
        'model': '',
        'weight': ''
    })
    # keys which need to refresh derived values
    _refresh_keys = {'delta', 'sampling_rate', 'starttime', 'npts'}
    # dict of required types for certain attrs
    _types = Stats._types
    _types.update({
        'model': str,
        'weight': str
    })

    def __init__(self, header={}):
        """
        """
        super(Stats, self).__init__(header)
        if self.location == '':
            self.location = self.defaults['location']

    def __str__(self):
        """
        Return better readable string representation of Stats object.
        """
        prioritized_keys = ['model','weight','station','channel', 'location', 'network',
                          'starttime', 'endtime', 'sampling_rate', 'delta',
                          'npts', 'calib']
        return self._pretty_str(prioritized_keys)
    
###################################################################################
# REVISED _add_processing_info DECORATOR DEFINITION ###############################
###################################################################################

@decorator
def _add_processing_info(func, *args, **kwargs):
    """
    This is a decorator that attaches information about a processing call as a string
    to the MLTrace.stats.processing and MLStream.stats.processing lists

    Attribution: Directly adapted from the obspy.core.trace function of the same name.
    """
    callargs = inspect.getcallargs(func, *args, **kwargs)
    callargs.pop("self")
    kwargs_ = callargs.pop("kwargs", {})
    info = [time.time(),"Wyrm 0.0.0:","{function}".format(function=func.__name__),"(%s)"]
    arguments = []
    arguments += \
        ["%s=%s" % (k, repr(v)) if not isinstance(v, str) else
         "%s='%s'" % (k, v) for k, v in callargs.items()]
    arguments += \
        ["%s=%s" % (k, repr(v)) if not isinstance(v, str) else
         "%s='%s'" % (k, v) for k, v in kwargs_.items()]
    arguments.sort()
    info[-1] = info[-1] % "::".join(arguments)
    self = args[0]
    result = func(*args, **kwargs)
    # Attach after executing the function to avoid having it attached
    # while the operation failed.
    self._internal_add_processing_info(info)
    return result


###################################################################################
# Machine Learning Trace Class Definition #########################################
###################################################################################

class MLTrace(Trace):
    """
    Extension of the obspy.core.trace.Trace class that adds in a 
    "fold" data attribute that records the number of obersvations contributing
    to a particular sample of the same index in the "data" attribute. Additionally
    it uses the MLStats class above in the place of the obspy.core.trace.Stats
    class to track additional information regarding ML model architectures and
    pretrained weights 

    This class provides updated definitions for the following obspy.core.trace.Trace
    methods to include an update to the "fold" attribute to match changes in the scaling
    of the "data" attribute

    :: UPDATED METHODS ::
    .__add__     - aliases to the .merge() method that is similar to, albiet with fewer options,
                    to the obspy.core.trace.Trace.__add__ method
    ._ltrim      - updates self.fold with the same truncation as self.data
    ._rtrim                         ""
    .merge                          ""
    .slice                          ""
    .slide                          ""

    NOTE: Consequently, these new definitions update the following Trace method behaviors
        Trace.trim, Trace.slice, Trace.slide, Trace.__div__, Trace.__mod__

    .resample    - wraps super().<method>(*args, **kwargs) and interpolates
                    fold using the numpy.interp1d
    .interpolate                    ""
    .decimate                       ""

    :: ADDITIONAL METHODS ::
    .merge - provides methods 0) discard overlapping values
                              1) linear interpolation across data overlaps
                              2) max value of overlapping data samples
                              3) fold-weighted average value of overlapping data samples
    .blind - set npts values on either end of self.fold to 0
    .as_trace - 
    :: UPDATED PROPERTIES ::
    .id     - "NET.STA.LOC.CHAN.MODEL.WEIGHT"
    .site   - "NET.STA"
    .inst   - "LOC.BandInstrument" (channel, minus the component code)
    .mod    - "MODEL.WEIGHT"
    """

    @_add_processing_info
    def __init__(self, data=np.array([], dtype=np.float32), fold=None, header=None):
        """
        Initialize an MLTrace object
        
        :: INPUTS ::
        :param data: [numpy.ndarray] data vector to write to this MLTrace
        :param header: [None] or [dict] initialization data for the
                        MLTrace.stats attribute, which is an MLStats object
        
        :: ATTRIBUTES ::
        :attr data: data vector, same as obspy.core.trace.Trace
        :attr fold: fold vector, same size as self.data, denotes the number of
                    observtions associated with a given element in MLTrace.data

                    This becomes important when using the MLTrace.merge() method
                    when conducting stacking (method = 2 or 3)
        :attr stats: obspy.core.trace.Stats child-class MLStats object
        """
        # If a trace is passed as data, do super().__init__ with it's data & header
        if type(data) == Trace:
            trace = data
            data = trace.data
            header = trace.stats
        
        # Initialize Trace inheritance
        super().__init__(data=data, header=None)
        if header is None:
            header = {}
        header.setdefault('npts', len(data))
        # Assign MLStats
        self.stats = MLStats(header=header)

        # If fold is None, assign uniform fold
        if fold is None:
            self.fold = np.ones(shape=self.data.shape, dtype=self.data.dtype)
        # If fold is a numpy array
        elif isinstance(fold, np.ndarray):  
            # And has the same shape as data
            if self.data.shape == fold.shape:
                # Import specified fold with data's dtype
                self.fold = fold.astype(self.data.dtype)
            # Otherwise kick errors
            else:
                raise ValueError
        else:
            raise TypeError
        # Handle case where data are masked - set mask values to 0 fold
        if isinstance(self.data, np.ma.masked_array):
            if np.ma.is_masked(self.data):
                self.fold[self.data.mask] = 0

    ###############################################################################
    # FOLD METHODS ################################################################
    ###############################################################################        
    
    def get_fold_trace(self):
        """
        Return an obspy.core.trace.Trace object with data=fold vector for this MLTrace,
        replacing the componet character with an 'f' and discarding additional properties
        in the trace stats
        """
        header = Stats()
        for _k in header.defaults.keys():
            header.update({_k: deepcopy(self.stats[_k])})
        ft = Trace(data=self.fold, header=header)
        ft.stats.channel += 'f'        
        return ft

    @_add_processing_info
    def apply_blinding(self, blinding=(500,500)):
        """
        Apply blinding to this MLTrace, which is defined by
        setting `blinding` samples on either end of the fold
        array to 0

        :: INPUT ::
        :param blinding: [2-tuple] of [int-like] or [int-like]
                        2-tuple: positive int-like number of samples to
                                blind on the left (blinding[0]) and 
                                right (blinding[1]) ends of fold
                        int-like: positive number of samples to blind
                                on either end of fold
        """
        if isinstance(blinding, (list, tuple)):
            if len(blinding) != 2:
                raise SyntaxError
            elif any(int(_b) != _b for _b in blinding):
                raise ValueError
            elif any(_b < 0 for _b in blinding):
                raise ValueError
            elif any(_b > self.stats.npts for _b in blinding):
                raise ValueError
            
            if blinding[0] > 0:
                self.fold[:blinding[0]] = 0
            if blinding[1] > 0:
                self.fold[-blinding[1]:] = 0

        elif isinstance(blinding, (int, float)):
            if int(blinding) != blinding:
                raise ValueError
            elif blinding < 0:
                raise ValueError
            elif blinding > self.stats.npts//2:
                raise ValueError
            else:
                self.fold[:blinding] = 0
                self.fold[-blinding:] = 0
        return self
    
    @_add_processing_info
    def to_zero(self, method='both'):
        """
        Convert this MLTrace IN-PLACE to a 0-valued vector for self.data and self.fold
        with the same shape and dtype
        """
        if method not in ['both','data','fold']:
            raise ValueError(f'method {method} not supported. Supported: "both", "data", "fold"')

        shape = self.data.shape
        dtype = self.data.dtype
        v0 = np.zeros(shape=shape, dtype=dtype)
        if method in ['both','data']:
            self.data = v0
        if method in ['both','fold']:
            self.fold = v0
        return self

    def enforce_zero_mask(self, **options):
        """
        Enforce equivalence of masking in data and 0-values in fold

        e.g., 

        data ->   1, 3, 4, 5, 2, 6, 7      --, 3, 4,--, 2, 6, 7
        mask ->   F, F, F, T, F, F, F  ==>  T, F, F, T, F, F, F
        fold ->   0, 1, 1, 1, 1, 1, 1       0, 1, 1, 0, 1, 1, 1
        changes:  *        *               *        *
        :: INPUTS ::
        :param **options: [kwargs] optional inputs to pass to np.ma.masked_array
                        in the event that the input data is not masked, but the
                        input fold has 0-valued elements.
        """
        # Update any places with 0 fold to masked values
        if any(self.fold == 0):
            # If not already a masked array, convert and update
            if not isinstance(self.data, np.ma.masked_array):
                self.data = np.ma.masked_array(data=self.data, mask=self.fold == 0, **options)
            # Otherwise ensure all fold=0 values are True in mask
            else:
                self.data.mask = self.fold==0
        # Update any places with masked values to 0 fold
        if isinstance(self.data, np.ma.MaskedArray):
            if np.ma.is_masked(self.data):
                self.fold[self.data.mask] = 0
        return self

    ###############################################################################
    # UPDATED MAGIC METHODS #######################################################
    ###############################################################################

    def __add__(self, other, method=1, fill_value=None, sanity_checks=True):
        """
        Alias to MLTrace.merge
        """
        self.merge(other, method=method, fill_value=fill_value, sanity_checks=sanity_checks)
        return self
    
    def __repr__(self, id_length=None):
        rstr = super().__str__(id_length=id_length)
        if self.stats.npts > 0:
            rstr += f' | Fold:'
            for _i in range(self.fold.max() + 1):
                ff = sum(self.fold == _i)/self.stats.npts
                if ff > 0:
                    rstr += f' [{_i}] {ff:.2f}'
        return rstr
    
    ###############################################################################
    # UPDATED TRIMMING METHODS ####################################################
    ###############################################################################

    def _ltrim(self, starttime, pad=False, nearest_sample=True, fill_value=None):
        """
        Run the obspy.core.trace.Trace._ltrim() method on this MLTrace and trim
        the start of the fold array if starttime > self.stats.endtime or pad the end
        of the fold array if starttime < self.stats.endtime and pad = True
        """
        old_fold = self.fold
        old_npts = self.stats.npts
        super()._ltrim(starttime, pad=pad, nearest_sample=nearest_sample, fill_value=fill_value)
        if old_npts < self.stats.npts:
            self.fold = np.full(shape=self.data.shape, fill_value=0, dtype=self.data.dtype)
            self.fold[-old_npts:] = old_fold
        elif old_npts > self.stats.npts:
            self.fold = old_fold[:self.stats.npts]
        else:
            self.fold = self.fold.astype(self.data.dtype)
        self.enforce_zero_mask()
        return self

    def _rtrim(self, endtime, pad=False, nearest_sample=True, fill_value=None):
        """
        Run the obspy.core.trace.Trace._rtrim() method on this MLTrace and trim
        the end of the fold array if endtime < self.stats.endtime or pad the end
        of the fold array if endtime > self.stats.endtime and pad = True
        """
        old_fold = self.fold
        old_npts = self.stats.npts
        super()._rtrim(endtime, pad=pad, nearest_sample=nearest_sample, fill_value=fill_value)
        if old_npts < self.stats.npts:
            self.fold = np.full(shape=self.data.shape, fill_value=0, dtype=self.data.dtype)
            self.fold[:old_npts] = old_fold
        elif old_npts > self.stats.npts:
            self.fold = old_fold[:self.stats.npts]
        else:
            self.fold = self.fold.astype(self.data.dtype)
        self.enforce_zero_mask()
        return self

    # @_add_processing_info
    def merge(self, trace, method='avg', fill_value=None, sanity_checks=True):
        """
        Conduct a "max" or "avg" stacking of a new trace into this MLTrace object under the
        following assumptions
            1) Both self and trace represent prediction traces output by the same continuous 
                predicting model with the same pretrained weights. 
                i.e., stats.model and stats.weight are not the default value and they match.
            2) The the prediction window inserted into this trace when it was initialized
                also had blinding applied to it
        
        :: INPUTS ::
        :param trace: [wyrm.core.trace.MLTrace] trace to append to this MLTrace
        :param blinding_samples: [int] number of samples to blind on either side of `trace`
                        prior to stacking
        :param method: [str] stacking method flag
                        'interpolate' - 
                        'avg' - (recursive) average stacking
                            new_data[t] = (self.data[t]*self.fold[t] + trace.data[t]*trace.fold[t])/
                                            (self.fold[t] + trace.fold[t])
                        'max' - maximum value stacking
                            new_data[t] = max([self.data[t], trace.data[t]])
                        for time-indexed sample "t"
        :param fill_value: [None] or [float-like] - fill_value to assign to a
                        numpy.ma.masked_array in the event that the stacking
                        operation produces a gap
        :param sanity_checks: [bool] should sanity checks be run on self and trace?
        
        """
        if sanity_checks:
            if isinstance(trace, Trace):
                if not isinstance(trace, MLTrace):
                    trace = MLTrace(data=trace.data, header=trace.stats)
                else:
                    raise TypeError
            # NOTE: If model/weight are specified in self.stats, 
            # the update above will raise the error below
            if self.id != trace.id:
                raise TypeError
            if self.stats.sampling_rate != trace.stats.sampling_rate:
                raise TypeError
            if self.stats.calib != trace.stats.calib:
                raise TypeError
            if self.data.dtype != trace.data.dtype:
                raise TypeError
        # Ensure fold matches data dtype
        if self.fold.dtype != self.data.dtype:
            self.fold = self.fold.astype(self.data.dtype)
        if trace.fold.dtype != trace.data.dtype:
            trace.fold = trace.fold.astype(trace.data.dtype)

        # Get data and fold vectors
        sdata = self.data
        sfold = self.fold
        tdata = trace.data
        tfold = trace.fold
        # Apply NaN masks to data for subsequent operations
        if isinstance(sdata, np.ma.masked_array):
            if np.ma.is_masked(sdata):
                sdata.fill_value = np.nan
                sfold[sdata.mask] = 0
                sdata = sdata.filled()
            else:
                sdata = sdata.data
        if isinstance(tdata, np.ma.masked_array):
            if np.ma.is_masked(tdata):
                tdata.fill_value = np.nan
                tfold[tdata.mask] = 0
                tdata = tdata.filled()
            else:
                tdata = tdata.data
        
        # Get relative indexing
        idx = self._relative_indexing(trace)

        # Create temporary data array
        tmp_data_array = np.full(shape=(2, max(idx) + 1),
                                 fill_value=np.nan,
                                 dtype=self.data.dtype)
        # Create temporary fold array for 
        tmp_fold_array = np.zeros(shape=(2, max(idx) + 1),
                                 dtype = self.fold.dtype)

        # Place self data into joined indexing positions (data being appended to)
        tmp_data_array[0, idx[0]:idx[1]+1] = self.data
        tmp_fold_array[0, idx[0]:idx[1]+1] = self.fold
        # Place trace data into joined indexing positions (data to be appended)
        tmp_data_array[1, idx[2]:idx[3]] = trace.data
        tmp_fold_array[1, idx[2]:idx[3]] = trace.fold

        if method in ['discard', 'dis', 0]:
            overlap_mask = np.isfinite(np.sum(tmp_data_array, axis=0))
            tmp_data = np.nansum(tmp_data_array, axis=0)
            tmp_fold = np.nansum(tmp_fold_array, axis=0)
            if overlap_mask.any():
                tmp_data[overlap_mask] = np.nan
                # Set eliminated sample fold to 0
                tmp_fold[overlap_mask] = 0

        # Interpolation method (effectively the same as obspy.core.trace.Trace.__add__(method=1))
        elif method in ['int', 'interpolate', 1]:
            # NaNsum folds for each element
            tmp_fold = np.nansum(tmp_fold_array, axis=0)
            # Check if there are overlaps using NaN padding to mask non-overlaps as False
            overlap_mask = np.isfinite(np.sum(tmp_data_array, axis=0))
            if overlap_mask.any():
                # self leading trace
                if idx[0] < idx[2] < idx[1] < idx[3]:
                    ls = tmp_data_array[0, idx[2]]
                    rs = tmp_data_array[1, idx[1]]
                    ds = idx[1] - idx[2]
                    tmp_data_array[:,idx[2]:idx[1]] = np.linspace(ls, rs, ds)
                    tmp_data = np.nansum(tmp_data_array, axis=0)
                # trace leading self
                elif idx[2] < idx[0] < idx[3] < idx[1]:
                    ls = tmp_data_array[1, idx[1]]
                    rs = tmp_data_array[0, idx[3]]
                    ds = idx[3] - idx[0]
                    tmp_data_array[:, idx[0]:idx[3]] = np.linspace(ls, rs, ds)
                    tmp_data = np.nansum(tmp_data_array, axis=0)

                # trace in self
                else:
                    tmp_data = tmp_data_array[0, :]
            # For gaps or contiguous, use nansum to get 
            else:
                tmp_data = np.nansum(tmp_data_array, axis=0)
                                      
        # (2) Use maximum value of overlapping elements(max stacking)
        elif method in ['max', 'maximum', 2]:
            # Create masking to NaN any 0 fold data (safety catch if enforce_zero_mask is somehow missed...)
            tmp_fold_mask = np.full(shape=tmp_data_array.shape, fill_value=np.nan, dtype=tmp_data_array.dtype)
            tmp_fold_mask[tmp_fold_array > 0] = 1
            tmp_data = np.nanmax(tmp_data_array*tmp_fold_mask, axis=0)
            tmp_fold = np.nansum(tmp_fold_array, axis=0)

        # (3) Use fold-weighted average of overlapping elements (average stacking)
        elif method in ['avg', 'average', 'mean', 3]:
            tmp_fold = np.nansum(tmp_fold_array, axis=0)
            tmp_data = np.nansum(tmp_data_array*tmp_fold_array, axis=0)
            idx = tmp_fold > 0
            tmp_data[idx] /= tmp_fold[idx]

        # If not all data are finite (i.e., gaps exist)
        if not np.isfinite(tmp_data).all():
            # Convert tmp_data into a masked array
            mask = ~np.isfinite(tmp_data)
            tmp_data = np.ma.masked_array(data=tmp_data,
                                          mask=mask,
                                          fill_value=fill_value)
            tmp_fold[mask] = 0
        # Update starttime
        new_t0 = self.stats.starttime + idx[0]/self.stats.sampling_rate
        # Overwrite starttime
        self.stats.starttime = new_t0
        # Overwrite data
        self.data = tmp_data
        # Overwrite fold
        if tmp_fold.dtype != self.data.dtype:
            tmp_fold = tmp_fold.astype(self.data.dtype)
        self.fold = tmp_fold
        self.enforce_zero_mask()
        return self
    
    def why(self):
        print('True love is the greatest thing on Earth - except for an MLT...')

    why = property(why)

    def _relative_indexing(self, other):
        """
        Helper method for merge() - calculates the integer index positions
        of the first and last samples of self and other on a uniformly
        sampled time index vector

        :: INPUTS ::
        :param other: [obspy.core.trace.Trace] or child class
        
        :: OUTPUT ::
        :return index: [list] with 4 elements
                        index[0] = relative position of self.data[0]
                        index[1] = relative position of self.data[1]
                        index[2] = relative position of other.data[0]
                        index[3] = relative position of other.data[1]
        """
        if not isinstance(other, Trace):
            raise TypeError
        if self.stats.sampling_rate != other.stats.sampling_rate:
            raise AttributeError('sampling_rate mismatches between this trace and "other"')
        self_t0 = self.stats.starttime
        self_t1 = self.stats.endtime
        other_t0 = other.stats.starttime
        other_t1 = other.stats.endtime
        sr = self.stats.sampling_rate
        if self_t0 <= other_t0:
            t0_ref = self_t0
        else:
            t0_ref = other_t0

        self_n0 = int((self_t0 - t0_ref)*sr)
        self_n1 = int((self_t1 - t0_ref)*sr) + 1
        other_n0 = int((other_t0 - t0_ref)*sr)
        other_n1 = int((other_t1 - t0_ref)*sr) + 1
        
        index = [self_n0, self_n1, other_n0, other_n1]
        return index

    def split(self):
        """
        Slight modification to the obspy.core.trace.Trace.split() method
        wherein a Stream of MLTrace objects are returned, which includes 
        a trimmed version of the MLTrace.fold attribute
        """
        # Not a masked array.
        if not isinstance(self.data, np.ma.masked_array):
            # no gaps
            return Stream([self.copy()])
        # Masked array but no actually masked values.
        elif isinstance(self.data, np.ma.masked_array) and \
                not np.ma.is_masked(self.data):
            _tr = self.copy()
            _tr.data = np.ma.getdata(_tr.data)
            return Stream([_tr])

        slices = flat_not_masked_contiguous(self.data)
        trace_list = []
        for slice in slices:
            if slice.step:
                raise NotImplementedError("step not supported")
            stats = self.stats.copy()
            tr = MLTrace(header=stats)
            tr.stats.starttime += (stats.delta * slice.start)
            # return the underlying data not the masked array
            tr.data = self.data.data[slice.start:slice.stop]
            tr.fold = self.fold[slice.start:slice.stop]
            trace_list.append(tr)
        return Stream(trace_list)
    
    ###############################################################################
    # WRAPPED RESAMPLING METHODS ##################################################
    ###############################################################################
    def resample(self, sampling_rate, window='hann', no_filter=True, strict_length=False):
        """
        Run the obspy.core.trace.Trace.resample() method on this MLTrace, interpolating
        the self.fold attribute with numpy.interp using relative times to the original
        MLTrace.stats.starttime value

        also see obpsy.core.trace.Trace.resample
        """
        tmp_fold = self.fold
        tmp_times = self.times()
        super().resample(sampling_rate, window=window, no_filter=no_filter, strict_length=strict_length)
        tmp_fold = np.interp(self.times(), tmp_times, tmp_fold)
        # Match data dtype
        if self.data.dtype != tmp_fold.dtype:
            tmp_fold = tmp_fold.astype(self.data.dtype)
        self.fold = tmp_fold
        return self

    def interpolate(
            self,
            sampling_rate,
            method='weighted_average_slopes',
            starttime=None,
            npts=None,
            time_shift=0.0,
            *args,
            **kwargs
    ):
        """
        Run the obspy.core.trace.Trace.interpolate() method on this MLTrace, interpolating
        the self.fold attribute using relative times if `starttime` is None or POSIX times
        if `starttime` is specified using numpy.interp

        also see obspy.core.trace.Trace.interpolate for detailed descriptions of inputs

        :: INPUTS ::
        :param sampling_rate: [float] new sampling rate
        :param method: [str] interpolation method - default is 'weighted_average_slopes'
                            also see obspy.core.trace.Trace.interpolate for other options
        :param starttime: [None] or [obspy.core.utcdatetime.UTCDateTime]
                          alternative starttime for interpolation (must be in the current
                          bounds of the original MLTrace)
        :param npts: [None] or [int] new number of samples to interpolate to
        :param time_shift: [float] seconds to shift the self.stats.starttime metadata by
                        prior to interpolation
        :param *args: [args] additional positional arguments to pass to the interpolator
                        in obspy.core.trace.Trace.interpolate
        :param **kwargs: [kwargs] additional key word arguments to pass to the interpolator
                        in obspy.core.trace.Trace.interpolate
        """
        tmp_fold = self.fold
        if starttime is None:
            tmp_times = self.times()
        else:
            tmp_times = self.times(type='timestamp')
        super().interpolate(
            sampling_rate,
            method=method,
            starttime=starttime,
            npts=npts,
            time_shift=time_shift,
            *args,
            **kwargs)
        if starttime is None:
            tmp_fold = np.interp(self.times(), tmp_times, tmp_fold)
        else:
            tmp_fold = np.interp(self.times(type='timestamp'), tmp_times, tmp_fold)
        # Length check
        if self.stats.npts < len(tmp_fold):
            # Trim off end values in the odd case where np.interp doesn't correctly scale tmp_fold
            tmp_fold = tmp_fold[:self.stats.npts]
        # dtype check
        if self.data.dtype != tmp_fold.dtype:
            tmp_fold = tmp_fold.astype(self.data.dtype)
        # Write update to self
        self.fold = tmp_fold
        return self
        
    def sync_to_window(self, starttime=None, endtime=None, fill_value=None, **kwargs):
        """
        Syncyronize the time sampling index
        """
        if starttime is not None:
            try:
                starttime = UTCDateTime(starttime)
            except TypeError:
                raise TypeError
            
        if endtime is not None:
            try:
                endtime = UTCDateTime(endtime)
            except TypeError:
                raise TypeError
        # If no bounds are stated, return self
        if endtime is None and starttime is None:
            return self
        # If only an endtime is stated
        elif starttime is None:
            # Nearest sample aligned with endtime that falls after starttime
            dn = (endtime - self.stats.starttime)//self.delta
            # So do not apply as (dn + 1) * delta
            starttime = endtime - dn*self.stats.delta
        else:
            dn = (self.stats.starttime - starttime)*self.stats.sampling_rate

        # Sampling points align, use MLTrace.trim() to pad to requisite size
        if int(dn) == dn:
            self.trim(starttime=starttime,endtime=endtime,pad=True, nearest_sample=True, fill_value=fill_value)
        # If samplign points don't align use MLTRace.interpolate() and .trim() to re-index and trim to size
        else:
            # If starttime is inside this MLTrace
            if self.stats.starttime < starttime < self.stats.endtime:
                self.interpolate(sampling_rate=self.stats.sampling_rate,
                                 starttime=starttime,**kwargs)
            # If starttime is before the start of the MLTrace
            elif starttime < self.stats.starttime:
                # Get time of the first aligned sample inside MLTrace
                dn = 1 + (self.stats.starttime - starttime)//self.stats.delta
                istart = starttime + dn*self.stats.delta
                # Interpolate on that sample
                self.interpolate(sampling_rate=self.stats.sampling_rate,
                                 starttime=istart,**kwargs)
            
            # If endtime is specified
            if endtime is not None:
                # If interpoalted data don't match the endtime, get there with trim()
                if self.stats.endtime != endtime:
                    self.trim(starttime=starttime,endtime=endtime, pad=True, nearest_sample=True,fill_value=fill_value)
        return self

    def decimate(self, factor, no_filter=False, strict_length=False):
        """
        Run the obspy.core.trace.Trace.decimate() method on this MLTrace, interpolating
        the self.fold attribute with numpy.interp using relative times to the original
        MLTrace.stats.starttime value

        also see obpsy.core.trace.Trace.decimate
        """
        tmp_fold = self.fold
        tmp_times = self.times()
        super().decimate(factor, no_filter=no_filter, strict_length=strict_length)
        tmp_fold = np.interp(self.times(), tmp_times, tmp_fold)
        # Match data dtype
        if self.data.dtype != tmp_fold.dtype:
            tmp_fold = tmp_fold.astype(self.data.dtype)
        self.fold = tmp_fold
        return self        



    ###############################################################################
    # I/O METHODS #################################################################
    ###############################################################################
    
    def to_trace(self, fold_threshold=0, output_mod=False):
        """
        Convert this MLTrace into an obspy.core.trace.Trace,
        masking values that have fold less than fold_threshold

        :: INPUTS ::
        :param fold_threshold: [int-like] threshold value, below which
                        samples are masked
        :param output_mod: [bool] should the model and weight strings
                        be included in the output?

        :: OUTPUT ::
        :return tr: [obspy.core.trace.Trace] trace copy with fold_threshold 
                        applied for masking
        :return model: [str] (if output_mod == True) model name
        :return weight: [str] (if output_mod == True) weight name
        """
        tr = Trace()
        for _k in tr.stats.keys():
            if _k not in tr.stats.readonly:
                tr.stats.update({_k: self.stats[_k]})
        data = self.data
        mask = self.fold < fold_threshold
        if any(mask):
            data = np.ma.masked_array(data=data,
                                      mask=mask)
        tr.data = data
        if output_mod:
            return tr, self.stats.model, self.stats.weight
        else:
            return tr
        
    def write(self, filename, **kwargs):
        """
        Wrapper around the root obspy.core.trace.Trace.write class
        method that appends the model and/or weight names to
        the end of the file name if they are not empty strings
        (i.e., default values '').

        This allows for preservation of model/weight information
        without having to 

        E.g., 
        mltr
            UW.GNW.--.BHP.EqT.pnw | 2023-10-09T02:20:15.42000Z - 2023-10-09T02:22:5.41000Z | 100.Hz, 15000 samples
        
        mltr.write('./myfile.mseed')
        ls
            myfile.EqT.pnw.mseed
        """
        fp, ext = os.path.splitext(filename)
        if self.stats.model != '':        
            fp += f'.{self.stats.model}'
        if self.stats.weight != '':
            fp += f'.{self.stats.weight}'
        filename = fp + ext
        st = Stream()
        st += self.to_trace()
        st += self.get_fold_trace()
        st.write(filename, format='MSEED', **kwargs)


    def from_trace(self, trace, fold_trace=None, model=None, weight=None, blinding=0):

        if not isinstance(trace, Trace):
            raise TypeError('trace must be an obspy.core.trace.Trace')
        
        self.data = trace.data
        for _k, _v in trace.stats.items():
            if _k == 'location' and _v == '':
                _v = '--'
            self.stats.update({_k: _v})
        if fold_trace is None:
            self.fold = np.ones(shape=self.data.shape, dtype=self.data.dtype)
            self.apply_blinding(blinding=blinding)
        
        elif isinstance(fold_trace, Trace):
            if trace.stats.starttime != fold_trace.stats.starttime:
                raise ValueError
            if trace.stats.sampling_rate != fold_trace.stats.sampling_rate:
                raise ValueError
            if trace.stats.npts != fold_trace.stats.npts:
                raise ValueError
            self.fold = fold_trace.data

        if model is not None:
            if not isinstance(model, str):
                raise TypeError('model must be type str or NoneType')
            else:
                self.stats.model = model
        
        if weight is not None:
            if not isinstance(weight, str):
                raise TypeError('weight must be type str or NoneType')
            else:
                self.stats.weight = weight

        return self 

    def from_file(self, filename, model=None, weight=None):
        st = read(filename)
        holder = {}
        if len(st) != 2:
            raise TypeError('this file does not contain the right number of traces (2)')
        for _tr in st:
            if _tr.stats.component == 'F':
                holder.update({'fold': _tr})
            else:
                holder.update({'data': _tr})
        if 'fold' not in holder.keys():
            raise TypeError('this file does not contain a fold trace')
        else:
            self.from_trace(holder['data'], fold_trace=holder['fold'], model=model, weight=weight)
        return self

    ###############################################################################
    # ID PROPERTY ASSIGNMENT METHODS ##############################################
    ###############################################################################
    def get_site(self):
        hdr = self.stats
        site = f'{hdr.network}.{hdr.station}'
        return site
    
    site = property(get_site)

    def get_inst(self):
        hdr = self.stats
        if len(hdr.channel) > 0:
            inst = f'{hdr.location}.{hdr.channel[:-1]}'
        else:
            inst = f'{hdr.location}.{hdr.channel}'
        return inst
    
    inst = property(get_inst)

    def get_comp(self):
        hdr = self.stats
        if len(hdr.channel) > 0:
            comp = hdr.channel[-1]
        else:
            comp = ''
        return comp
    
    def set_comp(self, other):
        char = str(other).upper()[0]
        self.stats.channel = self.stats.channel[:-1] + char            
    
    comp = property(get_comp)

    def get_mod(self):
        hdr = self.stats
        mod = f'{hdr.model}.{hdr.weight}'
        return mod
    
    def set_mod(self, model=None, weight=None):
        if model is not None:
            self.stats.model = model
        if weight is not None:
            self.stats.weight = weight

    mod = property(get_mod)

    def get_id(self):
        id = f'{self.site}.{self.inst}{self.comp}.{self.mod}'
        return id

    id = property(get_id)

    def get_instrument_id(self):
        id = f'{self.site}.{self.inst}'
        return id
    
    instrument = property(get_instrument_id)

    def get_valid_fraction(self, thresh=1):
        npts = self.stats.npts
        nv = np.sum(self.fold >= thresh)
        return nv/npts
    
    fvalid = property(get_valid_fraction)

    def get_id_element_dict(self):
        key_opts = {'id': self.id,
                    'instrument': self.instrument,
                    'site': self.site,
                    'inst': self.inst,
                    'component': self.comp,
                    'mod': self.mod,
                    'network': self.stats.network,
                    'station': self.stats.station,
                    'location': self.stats.location,
                    'channel': self.stats.channel,
                    'model': self.stats.model,
                    'weight': self.stats.weight
                    }
        # If id is not in traces.keys() - use dict.update
        return key_opts
    
    key_opts = property(get_id_element_dict)

###################################################################################
# Machine Learning Trace Buffer Class Definition ##################################
###################################################################################
    
class MLTraceBuffer(MLTrace):

    def __init__(self, max_length=1, blinding=None, restrict_past_append=True, **merge_kwargs):

        # Initialize as an MLTrace object
        super().__init__()
        # Compatability checks for max_length
        self.max_length = bounded_floatlike(
            max_length,
            name='max_length',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # Blinding compatability
        if blinding is None:
            self.blinding = False
        elif isinstance(blinding, (int, float, list, tuple)):
            self.blinding = blinding
        else:
            raise TypeError

        # Compatability check for restrict past appends
        if not isinstance(restrict_past_append, bool):
            raise TypeError
        else:
            self.RPA = restrict_past_append

        # Compatability checks for default merge(**kwargs)
        if merge_kwargs:
            self.merge_kwargs = merge_kwargs
        else:
            self.merge_kwargs = {}
        # breakpoint()
        # for _k, _v in inspect.signature(super().merge).items():
        #     if _k in merge_kwargs.keys():
        #         self.merge_kwargs.update({_k: merge_kwargs[_k]})
        #     elif _v.default != inspect._empty:
        #         self.merge_kwargs.update({_k: _v.default})
        # self._has_data = False

    def __add__(self, other):
        self.append(other)
        return self

    def append(self, other):
        if not isinstance(other, MLTrace):
            raise NotImplementedError
        
        # Apply blinding (if specified) to incoming trace
        if self.blinding:
            other.apply_blinding(blinding=self.blinding)

        # If this is a first append
        if not self._has_data:
            self._first_append(other)
        # If this is a subsequent append 
        else:
            # (FUTURE APPEND) If other ends at or after self (FUTURE APPEND)
            if other.stats.endtime >= self.stats.endtime:
                # If other starts within buffer range of self end
                if other.stats.starttime - self.max_length < self.stats.endtime:
                    # Conduct as a merge - future append (always unrestricted)
                    self.merge(self, other, **self.merge_kwargs)
                    self.enforce_max_length(reference='endtime')
                # If other starts later that self end + max_length - big gap
                else:
                    # Run as a first append if id matches
                    if self.id == other.id:
                        self._has_data = False
                        self._first_append(other)

            # (PAST APPEND) If other starts at or before self (PAST APPEND)
            elif other.stats.starttime <= self.stats.starttime:
                # If big past gap
                if self.stats.starttime - other.stats.endtime >= self.max_length:
                    # IF restriction in place
                    if self.RPA:
                        # Return self (cancel append)
                        pass
                    # IF restriction is not in place, run as first_append
                    else:
                        if self.id == other.id:
                            self._has_data = False
                            self._first_append(other)
                # If small past gap
                else:
                    if self.RPA:
                        self.merge(other, **self.merge_kwargs)
                        self.enforce_max_length(reference='endtime')
                    else:
                        self.merge(other, **self.merge_kwargs)
                        self.enforce_max_length(reference='starttime')

            # (INNER APPEND) - only allow merge if there is some masking
            else:
                # TODO: Make ssure this is a copy
                ftr = self.get_fold_trace().trim(starttime=other.stats.starttime, endtime=other.stats.endtime)
                # If there are any 0-fold data in self that have information from other
                if (ftr.data == 0 & other.fold >0).any():
                    self.merge(other, **self.merge_kwargs)
                else:
                    pass
            
        return self
    
    @_add_processing_info          
    def _first_append(self, other):
        if not self._has_data:
            self.stats = other.stats.copy()
            self.data = other.data
            self.enforce_max_length(reference='starttime')
            self._has_data = True
        else:
           raise AttributeError('This MLTraceBuffer already contains data - canceling _first_append()')
       
    def enforce_max_length(self, reference='endtime'):
        """
        Enforce the maximum length of the buffer using information
        """
        sr = self.stats.sampling_rate
        max_samp = int(self.max_length * sr + 0.5)
        if reference == 'endtime':
            te = self.stats.endtime
            ts = te - max_samp/sr
        elif reference == 'starttime':
            ts = self.stats.starttime
            te = ts + max_samp/sr
        self.trim(starttime=ts, endtime=te, pad=True, fill_value=None, nearest_sample=True)

    @_add_processing_info
    def trim_copy(self, starttime=None, endtime=None, **options):
        out = MLTrace(data=deepcopy(self.data), fold=deepcopy(self.fold), header=deepcopy(self.header))
        out.trim(starttime=starttime, endtime=endtime, **options)
        return out
    

        
# JUNKYARD ########################################################################

# class MLTrace(Trace):
#     """
#     Extend the obspy.core.trace.Trace class with additional stats entries for
#             model [str] - name of ML model architecture (method)
#             weight [str] - name of ML model weights (parameterization)
#     that are incorporated into the trace id attribute as :
#         Network.Station.Location.Channel.Model.Weight
#     and provides additional options for how overlapping MLTrace objects are merged
#     to conduct stacking of prediction time-series on the back-end of 
#     """
#     def __init__(
#             self,
#             data=np.array([], dtype=np.float32),
#             fold=np.array([], dtype=np.float32),
#             default_add_style='merge',
#             header=None):
#         super().__init__(data=data)
#         if header is None:
#             header = {}
#         header = deepcopy(header)
#         header.setdefault('npts', len(data))
#         self.stats = MLStats(header)
#         # super(MLTrace, self).__setattr__('data', data)
#         if self.data.shape != fold.shape or self.data.dtype != fold.dtype:
#             self.fold = np.ones(shape=self.data.shape, dtype=self.data.dtype)
#         if default_add_style not in ['merge','stack']:
#             raise ValueError(f'default_add_style {default_add_style} not supported. Supported: "merge", "stack"')

#     ###############################
#     # Property-supporting methods #
#     ############################### 
#     def get_ml_id(self):
#         id_str = super().get_id()
#         if self.stats.model is not None:
#             id_str += f'.{self.stats.model}'
#         else:
#             id_str += '.'
#         if self.stats.weight is not None:
#             id_str += f'.{self.stats.weight}'
#         else:
#             id_str += '.'
#         return id_str
    
#     def get_id(self):
#         return super().get_id()

#     def get_instrument_code(self):
#         """
#         Return instrument code
#         Location.BandInst

#         e.g. for UW.GNW.--.BHZ.EqT.pnw
#         return '--.BH'
#         """
#         inst_code = f'{self.stats.location}.{self.stats.channel[:-1]}'
#         return inst_code

#     def get_site_code(self):
#         """
#         Retrun site code:
#         Network.Station

#         e.g., for UW.GNW.--.BHZ.EqT.pnw
#         return 'UW.GNW'
#         """
#         site_code = f'{self.stats.network}.{self.stats.station}'
#         return site_code
    
#     def get_model_code(self):
#         """
#         Retrun model code:
#         Model.Weight

#         e.g., for UW.GNW.--.BHZ.EqT.pnw
#         return 'EqT.pnw'
#         """
#         model_code = f'{self.stats.model}.{self.stats.weight}'
#         return model_code
    
#     # Set properties for convenient access
#     trace_id = property(get_id)
#     id = property(get_ml_id)
#     site = property(get_site_code)
#     inst = property(get_instrument_code)
#     mod = property(get_model_code)



#     # Change the component code in self.stats
#     @_add_processing_info
#     def set_component(self, new_label):
#         """
#         Change the component character in self.stats.channel
#         to new_label

#         e.g., self.set_component(new_label = 'Detection')
#               self.stats.channel
#                 'BHD'
#         """
#         if isinstance(new_label, str):
#             if len(new_label) == 1:
#                 self.stats.channel = self.instrument + new_label
#             elif len(new_label) > 1:
#                 self.stats.channel = self.instrument + new_label[0]
#             else:
#                 raise ValueError('str-type new_label must be a non-empty string')
#         elif isinstance(new_label, int):
#             if new_label < 10:
#                 self.stats.channel = f'{self.instrument}{new_label}'
#             else:
#                 raise ValueError('int-type new_label must be a single-digit integer') 
#         else:
#             raise TypeError('new_label must be type str or int')


#     def split(self):
#         """
#         Slight modification to the obspy.core.trace.Trace.split() method
#         wherein returned 
#         """
#         # Not a masked array.
#         if not isinstance(self.data, np.ma.masked_array):
#             # no gaps
#             return Stream([self.copy()])
#         # Masked array but no actually masked values.
#         elif isinstance(self.data, np.ma.masked_array) and \
#                 not np.ma.is_masked(self.data):
#             _tr = self.copy()
#             _tr.data = np.ma.getdata(_tr.data)
#             return Stream([_tr])

#         slices = flat_not_masked_contiguous(self.data)
#         trace_list = []
#         for slice in slices:
#             if slice.step:
#                 raise NotImplementedError("step not supported")
#             stats = self.stats.copy()
#             tr = MLTrace(header=stats)
#             tr.stats.starttime += (stats.delta * slice.start)
#             # return the underlying data not the masked array
#             tr.data = self.data.data[slice.start:slice.stop]
#             trace_list.append(tr)
#         return Stream(trace_list)



#     @_add_processing_info
#     def apply_blinding(self, npts_blinding):
#         """
#         Apply blinding (i.e., set fold to 0) to samples
#         on either end of the contents of this MLTrace
        
#         :: INPUT ::
#         :param npts_blinding: [int] non-negative 
#         """
#         if self.stats.npts//2 > npts_blinding > 0:
#             self.fold[:npts_blinding] = 0
#             self.fold[-npts_blinding:] = 0
#         elif npts_blinding == 0:
#             pass
#         else:
#             raise ValueError('npts_blinding must be an integer in [0, self.stats.npts//2)')
#         return self
    
#     # def __add__(self, trace, add_style='merge', **options):
#     #     if add_style == 'merge':
#     #         self._merge(trace, **options)
#     #     elif add_style == 'stack':
#     #         self._stack(trace, **options)
#     #     else:
#     #         raise ValueError(f'add_style "{add_style}" not supported. Only "merge" and "stack"')

#     @_add_processing_info
#     def __add__(self, trace, method=1, fill_value=None, interpolation_samples=-1, sanity_checks=True):
#         """
#         Wrapper around obspy.core.trace.Trace.__add__()
#         """
#         super().__add__(
#             trace,
#             method=method,
#             fill_value=fill_value,
#             interpolation_samples=interpolation_samples,
#             sanity_checks=sanity_checks)
        
#     @_add_processing_info
#     def __mul__(self, trace, blinding_samples=0, method='avg', fill_value=0., sanity_checks=True):
#         """
#         Conduct a "max" or "avg" stacking of a new trace into this MLTrace object under the
#         following assumptions
#             1) Both self and trace represent prediction traces output by the same continuous 
#                 predicting model with the same pretrained weights. 
#                 i.e., stats.model and stats.weight are not the default value and they match.
#             2) The the prediction window inserted into this trace when it was initialized
#                 also had blinding applied to it
        
#         :: INPUTS ::
#         :param trace: [wyrm.core.trace.MLTrace] trace to append to this MLTrace
#         :param blinding_samples: [int] number of samples to blind on either side of `trace`
#                         prior to stacking
#         :param method: [str] stacking method flag
#                         'avg' - (recursive) average stacking
#                             new_data[t] = (self.data[t]*self.fold[t] + trace.data[t]*trace.fold[t])/
#                                             (self.fold[t] + trace.fold[t])
#                         'max' - maximum value stacking
#                             new_data[t] = max([self.data[t], trace.data[t]])
#                         for time-indexed sample "t"
#         :param fill_value: [None] or [float-like] - fill_value to assign to a
#                         numpy.ma.masked_array in the event that the stacking
#                         operation produces a gap
#         :param sanity_checks: [bool] should sanity checks be run on self and trace?
        
#         """
#         if sanity_checks:
#             if not isinstance(trace, MLTrace):
#                 raise TypeError
#             if self.id != trace.id:
#                 raise TypeError(f'MLTrace ID differs: {self.id} vs {trace.id}')
#             if self.stats.sampling_rate != trace.stats.sampling_rate:
#                 raise TypeError(f'Sampling rate differs: {self.stats.sampling_rate} vs {trace.stats.sampling_rate}')
#             if self.stats.calib != trace.stats.calib:
#                 raise TypeError
#             if self.data.dtype != trace.data.dtype:
#                 raise TypeError
#             if self.stats.model == '' or self.stats.weight == '':
#                 raise AttributeError('Stacking should only be applied to prediction-containing MLTrace objects')
#             if np.ma.is_masked(self.data) or np.ma.is_masked(trace.data):
#                 raise TypeError('Can only stack on unmasked data')
#         # Get relative timing status
#         # rt_status = self._assess_relative_timing(trace)
#         stack_indices = self.get_relative_indexing(trace)
#         tmp_data_array = np.full(shape=(max(stack_indices) + 1, 2),
#                                  fill_value=np.nan,
#                                  dtype=self.data.dtype)
#         tmp_fold_array = np.zeros(shape=(max(stack_indices) + 1, 2),
#                                  dtype = self.fold.dtype)
#         tmp_data_array[stack_indices[0]:stack_indices[1]+1, 0] = self.data
#         tmp_fold_array[stack_indices[0]:stack_indices[1]+1, 0] = self.fold
#         tmp_data_array[stack_indices[2]+blinding_samples:stack_indices[3]-blinding_samples, 1] = trace.data[blinding_samples:-blinding_samples]
#         tmp_fold_array[stack_indices[2]+blinding_samples:stack_indices[3]-blinding_samples, 1] = trace.fold[blinding_samples:-blinding_samples]

#         tmp_fold = np.nansum(tmp_fold_array, axis=1)
#         if method == 'max':
#             tmp_data = np.nanmax(tmp_data_array, axis=1)
        
#         elif method == 'avg':
#             tmp_data = np.nansum(tmp_data_array*tmp_fold_array, axis=1)
#             idx = tmp_fold > 0
#             tmp_data[idx] /= tmp_fold[idx]
#         # If not all data are finite
#         if not np.isfinite(tmp_data).all():
#             mask = ~np.isfinite(tmp_data)
#             tmp_data = np.ma.masked_array(data=tmp_data,
#                                           mask=mask,
#                                           fill_value=fill_value)
        
#         new_t0 = self.stats.starttime + stack_indices[0]/self.stats.sampling_rate
#         self.stats.starttime = new_t0
#         self.data = tmp_data
#         self.fold = tmp_fold

#     def _assess_relative_timing(self, other):
#         """
#         Assess the relative timing and sampling alignment of this MLTrace
#         and another Trace-type object with the same sampling_rate

#         keys in output `status` should be read as:
#             self {key_string} other
#             e.g,. self 'starts_before' other : True/False
#         and delta npts values are calculated as:
#             (self - other)*sampling_rate
#         """

#         if not isinstance(other, Trace):
#             raise TypeError('trace must be type obspy.core.trace.Trace')
#         if self.stats.sampling_rate != other.stats.sampling_rate:
#             raise TypeError('other sampling_rate does not match this trace\'s sampling_rate')
#         self_ts = self.stats.starttime
#         other_ts = other.stats.starttime
#         self_te = self.stats.endtime
#         other_te = other.stats.endtime
#         delta_ts = self_ts - other_ts
#         delta_te = self_te - other_te
#         dnpts_ts = delta_ts*self.stats.sampling_rate
#         dnpts_te = delta_te*self.stats.sampling_rate
#         # overlap_status = any(self_ts <= other_ts <= self_te,
#         #                       self_ts <= other_te <= self_te)
#         # # gap_status
#         #                   'overlaps': overlap_status,
#         #           'creates_gap': gap_status,

#         status = {'starts_before': self_ts < other_ts,
#                   'dnpts_starttime': dnpts_ts,
#                   'ends_before': self_te < other_te,
#                   'dnpts_endtime': dnpts_te,
#                   'sampling_aligns': int(dnpts_ts) == dnpts_ts}
#         return status
    
#     def get_relative_indexing(self, other):
#         if not isinstance(other, MLTrace):
#             raise TypeError
#         if self.stats.sampling_rate != other.stats.sampling_rate:
#             raise AttributeError('sampling_rate mismatches between this trace and "other"')
#         self_t0 = self.stats.starttime
#         self_t1 = self.stats.endtime
#         other_t0 = other.stats.starttime
#         other_t1 = other.stats.endtime
#         sr = self.stats.sampling_rate
#         if self_t0 <= other_t0:
#             t0_ref = self_t0
#         else:
#             t0_ref = other_t0

#         self_n0 = int((self_t0 - t0_ref)*sr)
#         self_n1 = int((self_t1 - t0_ref)*sr) + 1
#         other_n0 = int((other_t0 - t0_ref)*sr)
#         other_n1 = int((other_t1 - t0_ref)*sr) + 1
        
#         return [self_n0, self_n1, other_n0, other_n1]


    







# class MLTraceBuffer(MLTrace):

#     def __init__(self, max_length=1., add_style='merge', restrict_add_past=True, **options):
#         # Inherit from MLTrace
#         super().__init__()
#         self.max_length = wuc.bounded_floatlike(
#             max_length,
#             name='max_length',
#             minimum=0.0,
#             maximum=None,
#             inclusive=False
#         )
#         if not isinstance(add_style, str):
#             raise TypeError
#         elif add_style not in ['merge', 'stack']:
#             raise ValueError('add_style {add_style} not supported. Supported values: "merge", "stack"')
#         else:
#             add_kwargs = {}
#             if add_style == 'merge':
#                 merge_kwarg_keys = ['method', 'interpolation_samples','fill_value','sanity_checks']
#                 for _mkk in merge_kwarg_keys:
#                     if _mkk in options.keys():
#                         add_kwargs.update({_mkk: options[_mkk]})
#             elif add_style == 'stack':
#                 stack_kwarg_keys = ['method', 'blinding_samples','fill_value','sanity-checks']
#                 for _skk in stack_kwarg_keys:
#                     if _skk in options.keys():
#                         add_kwargs.update({_skk: options[_skk]})

#             self.add_kwargs = add_kwargs

#         if not isinstance(restrict_add_past, bool):
#             raise TypeError('restrict_add_past must be type bool')
#         else:
#             self.restrict_add_past = restrict_add_past

    


#     def trim(self, starttime=None, endtime=None, pad=True, fill_value=None, nearest_sample=True):
#         """
#         Produce a MLTrace object that is a copy of data trimmed out of this MLTraceBuffer
#         using the MLTrace.trim() method - this trims both the self.data and self.fold arrays
#         """
#         out = MLTrace(data = self.data, fold=self.fold, header=self.stats)
#         out.trim(
#             starttime=starttime,
#             endtime=endtime,
#             pad=pad,
#             fill_value=fill_value,
#             nearest_sample=nearest_sample)
#         return out
    
#     def enforce_max_length(self, **options):
#         if self.stats.endtime - self.stats.starttime > self.max_length:
#             self._ltrim(starttime=self.stats.endtime - self.max_length, **options)
#         else:
#             pass
#         return self

    # def append(self, trace, )
        

    # def treat_gaps(
    #         self,
    #         merge_kwargs={},
    #         detrend_kwargs={'type': 'linear'},
    #         filter_kwargs=False,
    #         taper_kwargs={'max_percentage': None, 'max_length': 0.06, 'side':'both'}
    #         ):
    #     self.stats.processing.append('vvv Wyrm 0.0.0: treat_gaps vvv')
    #     if np.ma.is_masked(self.data):
    #         self = self.split()
    #         if detrend_kwargs:
    #             self.detrend(**detrend_kwargs)
    #         if filter_kwargs:
    #             self.filter(**filter_kwargs)
    #         if taper_kwargs:
    #             self.taper(**taper_kwargs)
    #         self.merge(**merge_kwargs)
    #         if isinstance(self, Stream):
    #             self = self[0]
    #     else:
    #         if detrend_kwargs:
    #             self.detrend(**detrend_kwargs)
    #         if filter_kwargs:
    #             self.filter(**filter_kwargs)
    #         if taper_kwargs:
    #             self.taper(**taper_kwargs)
    #     self.stats.processing.append('^^^ Wyrm 0.0.0: treat_gaps ^^^')
    #     return self

    # def treat_gappy_trace(
    #         self,
    #         label,
    #         merge_kwargs={},
    #         detrend_kwargs=False,
    #         filter_kwargs=False,
    #         taper_kwargs=False
    #         ):

    #     _x = self.traces[label].copy()
    #     if np.ma.is_masked(_x.data):
    #         _x = _x.split()
    #         if detrend_kwargs:
    #             _x.detrend(**detrend_kwargs)
    #         if filter_kwargs:
    #             _x.filter(**filter_kwargs)
    #         if taper_kwargs:
    #             _x.taper(**taper_kwargs)
    #         _x.merge(**merge_kwargs)
    #         if isinstance(_x, Stream):
    #             _x = _x[0]
    #     else:
    #         if detrend_kwargs:
    #             _x.detrend(**detrend_kwargs)
    #         if filter_kwargs:
    #             _x.filter(**filter_kwargs)
    #         if taper_kwargs:
    #             _x.taper(**taper_kwargs)
    #     self.traces.update({label: _x})
    #     return self




    # def check_sync(self, level='summary'):
    #     # Use sync t0, if available, as reference
    #     if self.sync_t0:
    #         ref_t0 = self.sync_t0
    #     # Otherwise use specified ref_t0 if initially stated (or updated)
    #     elif self.ref_t0 is not None:
    #         ref_t0 = self.ref_t0
    #     # Otherwise use the starttime of the first trace in the Stream
    #     else:
    #         ref_t0 = list(self.traces.values())[0].stats.starttime
    #     # Use sync sampling_rate, if available, as reference
    #     if self.sync_sampling_rate:
    #         ref_sampling_rate = self.sync_sampling_rate
    #     # Otherwise use specified ref_sampling_rate if initially stated (or updated)
    #     elif self.ref_sampling_rate is not None:
    #         ref_sampling_rate = self.ref_sampling_rate
    #     # Otherwise use the starttime of the first trace in the Stream
    #     else:
    #         ref_sampling_rate = list(self.traces.values())[0].stats.starttime
        
    #     # Use sync npts, if available, as reference
    #     if self.sync_npts:
    #         ref_npts = self.sync_npts
    #     # Otherwise use specified ref_npts if initially stated (or updated)
    #     elif self.ref_npts is not None:
    #         ref_npts = self.ref_npts
    #     # Otherwise use the starttime of the first trace in the Stream
    #     else:
    #         ref_npts = list(self.traces.values())[0].stats.npts

    #     trace_bool_holder = {}
    #     for _l, _tr in self.traces.items():
    #         attr_bool_holder = [
    #             _tr.stats.starttime == ref_t0,
    #             _tr.stats.sampling_rate == ref_sampling_rate,
    #             _tr.stats.npts == ref_npts
    #         ]
    #         trace_bool_holder.update({_l: attr_bool_holder})

    #     df_bool = pd.DataFrame(trace_bool_holder, index=['t0','sampling_rate','npts']).T
    #     if level.lower() == 'summary': 
    #         status = df_bool.all(axis=0).all(axis=0)
    #     elif level.lower() == 'trace':
    #         status = df_bool.all(axis=1)
    #     elif level.lower() == 'attribute':
    #         status = df_bool.all(axis=0)
    #     elif level.lower() == 'debug':
    #         status = df_bool
    #     return status








# class MLStreamStats(AttribDict):

#     # readonly = ['sync_status','common_id']
#     defaults = {
#         'common_id': None,
#         'ref_starttime': None,
#         'ref_sampling_rate': None,
#         'ref_npts': None,
#         'ref_model': None,
#         'ref_weight': None,
#         'sync_status': {'starttime': False, 'sampling_rate': False, 'npts': False},
#         'processing': []
#     }

#     def __init__(self, header={}):
#         super(MLStreamStats, self).__init__(header)

#     def __repr__(self):
#         rstr = '----Stats----'
#         for _k, _v in self.items():
#             rstr += f'\n{_k:>18}: '
#             if _k == 'syncstatus':
#                 for _k2, _v2 in _v.items():
#                     rstr += f'\n{_k2:>26}: {_v2}'
#             else:
#                 rstr += f'{_v}'
#         return rstr















# class MLStream(Stream):
#     """
#     Adapted version of the obspy.core.stream.Stream class that uses a dictionary as the holder
#     for Trace-type objects to leverage hash-table search acceleration for large collections of
#     traces.
#     """
#     def __init__(
#             self,
#             traces=None,
#             header={},
#             **options):
#         """
#         Initialize a MLStream object

#         :: INPUTS ::
#         :param traces: [None] - returns an empty dict as self.traces
#                        [obspy.core.stream.Stream] or [list-like] of [obspy.core.trace.Trace]
#                             populates self.traces using each constituient
#                             trace's trace.id attribute as the trace label 
#                             (dictionary key).

#                             If there are multiple traces with the same trace.id
#                             the self.traces['trace.id'] subsequent occurrences
#                             of traces with the same id are appended to the
#                             initial trace using the initial Trace object's 
#                             __add__ ~magic~ method (i.e,. )

#                             In the case of an obspy.core.trace.Trace object, this
#                             results in an in-place execution of the __add__ method
#                             that calls Trace.merge() that is passed **options

#                             In the case of a wyrm.core.data.TraceBuffer, this 
#                             results in an application of the TraceBuffer.append()
#                             method, which has some additional restrictions on the
#                             timing of packets.

#         :param header: [dict] - dictionary passed to WindowStats.__init__ for
#                             defining reference values for this collection of traces
#                             i.e., target values for data attributes
#                             Supported keys:
#                                 'ref_starttime' [float] or [UTCDateTime] target starttime for each trace
#                                 'ref_sampling_rate' [float] target sampling rate for each trace
#                                 'ref_npts' [int] target number of samples in each trace
#                                 'ref_model' [str] name of model architecture associated with thi MLStream
#                                 'ref_weight': [str] name of model parameterization associated with this MLStream

#         :param **options: [kwargs] key-word arguments passed to Trace-like's __add__ method in the
#                                 event of multiple matching trace id's
        
#         """

#         super().__init__(traces=traces)

#         self.stats = MLStreamStats(header=header)
#         self.traces = {}
#         self.options = options

#         if traces is not None:
#             self.__add__(traces, **options)
        
#         if self.is_syncd(run_assessment=True):
#             print('Traces ready for tensor conversion')
#         else:
#             print('Steps required to sync traces prior to tensor conversion')

#     def __add__(self, other, **options):
#         """
#         Add a Trace-type or Stream-type object to this MLStream
#         """
#         # If adding a list-like
#         if isinstance(other, (list, tuple)):
#             # Check that each entry is a trace
#             for _tr in other:
#                 # Raise error if not
#                 if not isinstance(_tr, Trace):
#                     raise TypeError('all elements in traces must be type obspy.core.trace.Trace')
#                 # Use Trace ID as label
#                 else:
#                     iterant = other
#         # Handle case where other is Stream-like
#         elif isinstance(other, Stream):
#             # Particular case for MLStream
#             if isinstance(other, MLStream):
#                 iterant = other.values()
#             else:
#                 iterant = other
#         # Handle case where other is type-Trace (includes TraceBuffer)
#         elif isinstance(other, Trace):
#             # Make an iterable list of 1 item
#             iterant = [other]
#         # Iterate across list-like of traces "iterant"
#         for _tr in iterant:
#             if _tr.id not in self.traces.keys():
#                 self.traces.update({_tr.id: MLTrace(data=_tr.data, header=_tr.stats)})
#             else:
#                 self.traces[_tr.id].__add__(MLTrace(data=_tr.data, header=_tr.stats), **options)


    
#     @_add_processing_info
#     def fnselect(self, fnstr='*'):
#         """
#         Create a copy of this MLStream with trace ID's (keys) that conform to an input
#         Unix wildcard-compliant string. This updates the self.stats.common_id attribute

#         :: INPUT ::
#         :param fnstr: [str] search string to search with
#                         fnmatch.filter(self.traces.keys(), fnstr)

#         :: OUTPUT ::
#         :return lst: [wyrm.core.data.MLStream] subset copy
#         """
#         matches = fnmatch.filter(self.traces.keys(), fnstr)
#         lst = self.copy_dataless()
#         for _m in matches:
#             lst.traces.update({_m: self.traces[_m].copy()})
#         lst.stats.common_id = fnstr
#         return lst
    
#     def __repr__(self, extended=False):
#         rstr = self.stats.__repr__()
#         if len(self.traces) > 0:
#             id_length = max(len(_tr.id) for _tr in self.traces.values())
#         else:
#             id_length=0
#         rstr += f'\n{len(self.traces)} Trace(s) in LabelStream:\n'
#         if len(self.traces) <= 20 or extended is True:
#             for _l, _tr in self.items():
#                 rstr += f'{_l:} : {_tr.__str__(id_length)}\n'
#         else:
#             _l0, _tr0 = list(self.items())[0]
#             _lf, _trf = list(self.items())[-1]
#             rstr += f'{_l0:} : {_tr0.__str__(id_length)}\n'
#             rstr += f'...\n({len(self.traces) - 2} other traces)\n...'
#             rstr += f'{_lf:} : {_trf.__str__(id_length)}\n'
#             rstr += f'[Use "print(MLStream.__repr__(extended=True))" to print all labels and traces]'
#         return rstr
    
#     def __str__(self):
#         rstr = 'wyrm.core.data.MLStream()'
#         return rstr


#     def _assess_starttime_sync(self, new_ref_starttime=False):
#         if isinstance(new_ref_starttime, UTCDateTime):
#             ref_t0 = new_ref_starttime
#         else:
#             ref_t0 = self.stats.ref_starttime
        
#         if all(_tr.stats.starttime == ref_t0 for _tr in self.list_traces()):
#             self.stats.sync_status['starttime'] = True
#         else:
#             self.stats.sync_status['starttime'] = False

#     def _assess_sampling_rate_sync(self, new_ref_sampling_rate=False):
#         if isinstance(new_ref_sampling_rate, float):

#             ref_sampling_rate = new_ref_sampling_rate
#         else:
#             ref_sampling_rate = self.stats.ref_sampling_rate
        
#         if all(_tr.stats.sampling_rate == ref_sampling_rate for _tr in self.list_traces()):
#             self.stats.sync_status['sampling_rate'] = True
#         else:
#             self.stats.sync_status['sampling_rate'] = False

#     def _assess_npts_sync(self, new_ref_npts=False):
#         if isinstance(new_ref_npts, int):
#             ref_npts = new_ref_npts
#         else:
#             ref_npts = self.stats.ref_npts
        
#         if all(_tr.stats.npts == ref_npts for _tr in self.list_traces()):
#             self.stats.sync_status['npts'] = True
#         else:
#             self.stats.sync_status['npts'] = False

#     def is_syncd(self, run_assessment=True, **options):
#         if run_assessment:
#             if 'new_ref_starttime' in options.keys():
#                 self._assess_starttime_sync(new_ref_starttime=options['new_ref_starttime'])
#             else:
#                 self._assess_starttime_sync()
#             if 'new_ref_sampling_rate' in options.keys():
#                 self._assess_sampling_rate_sync(new_ref_sampling_rate=options['new_ref_sampling_rate'])
#             else:
#                 self._assess_sampling_rate_sync()
#             if 'new_ref_npts' in options.keys():
#                 self._assess_npts_sync(new_ref_npts=options['new_ref_npts'])
#             else:
#                 self._assess_npts_sync()
#         if all(self.stats.sync_status.values()):
#             return True
#         else:
#             return False

#     def diagnose_sync(self):
#         """
#         Produce a pandas.DataFrame that contains the sync status
#         for the relevant sampling reference attriburtes:
#             starttime
#             sampling_rate
#             npts
#         where 
#             True indicates a trace (id in index) matches the
#                  reference attribute (column) value
#             False indicates the trace mismatches the reference
#                     attribute value
#             'NoRefVal' indicates no reference value was available

#         :: OUTPUT ::
#         :return out: [pandas.dataframe.DataFrame] with sync
#                     assessment values.
#         """

#         index = []; columns=['starttime','sampling_rate','npts'];
#         holder = []
#         for _l, _tr in self.items():
#             index.append(_l)
#             line = []
#             for _f in columns:
#                 if self.stats['ref_'+_f] is not None:
#                     line.append(_tr.stats[_f] == self.stats['ref_'+_f])
#                 else:
#                     line.append('NoRefVal')
#             holder.append(line)
#         out = pd.DataFrame(holder, index=index, columns=columns)
#         return out


#     def check_sync(self, level='summary'):
#         # Use sync t0, if available, as reference
#         if self.sync_t0:
#             ref_t0 = self.sync_t0
#         # Otherwise use specified ref_t0 if initially stated (or updated)
#         elif self.ref_t0 is not None:
#             ref_t0 = self.ref_t0
#         # Otherwise use the starttime of the first trace in the Stream
#         else:
#             ref_t0 = list(self.traces.values())[0].stats.starttime
#         # Use sync sampling_rate, if available, as reference
#         if self.sync_sampling_rate:
#             ref_sampling_rate = self.sync_sampling_rate
#         # Otherwise use specified ref_sampling_rate if initially stated (or updated)
#         elif self.ref_sampling_rate is not None:
#             ref_sampling_rate = self.ref_sampling_rate
#         # Otherwise use the starttime of the first trace in the Stream
#         else:
#             ref_sampling_rate = list(self.traces.values())[0].stats.starttime
        
#         # Use sync npts, if available, as reference
#         if self.sync_npts:
#             ref_npts = self.sync_npts
#         # Otherwise use specified ref_npts if initially stated (or updated)
#         elif self.ref_npts is not None:
#             ref_npts = self.ref_npts
#         # Otherwise use the starttime of the first trace in the Stream
#         else:
#             ref_npts = list(self.traces.values())[0].stats.npts

#         trace_bool_holder = {}
#         for _l, _tr in self.traces.items():
#             attr_bool_holder = [
#                 _tr.stats.starttime == ref_t0,
#                 _tr.stats.sampling_rate == ref_sampling_rate,
#                 _tr.stats.npts == ref_npts
#             ]
#             trace_bool_holder.update({_l: attr_bool_holder})

#         df_bool = pd.DataFrame(trace_bool_holder, index=['t0','sampling_rate','npts']).T
#         if level.lower() == 'summary': 
#             status = df_bool.all(axis=0).all(axis=0)
#         elif level.lower() == 'trace':
#             status = df_bool.all(axis=1)
#         elif level.lower() == 'attribute':
#             status = df_bool.all(axis=0)
#         elif level.lower() == 'debug':
#             status = df_bool
#         return status

#     @_add_processing_info

    
#     @_add_processing_info
#     def treat_gappy_traces(
#         self, 
#         detrend_kwargs={'type':'linear'},
#         merge_kwargs={'method': 1, 'fill_value': 0, 'interpolation_samples': -1},
#         filter_kwargs={'type': 'bandpass', 'freqmin': 1, 'freqmax': 45},
#         taper_kwargs={ 'max_percentage': None, 'max_length': 0.06, 'side': 'both'},
#     ):
#         for _tr in self.traces.values():
#             _tr.treat_gaps(merge_kwargs=merge_kwargs,
#                            detrend_kwargs=detrend_kwargs
#                            filter_kwargs=filter_kwargs,
#                            taper_kwargs=taper_kwargs)
#         self.stats.processing.append('Wyrm 0.0.0: treat_gappy_traces')
#         return self

#     @_add_processing_info



    # def append_trace(self, trace, label, merge_kwargs={'method': 1, 'fill_value': None, 'pad': True, 'interpolation_samples': -1}):
    #     if label in self.traces.keys():
    #         _tr = self.traces[label].copy()
    #         if _tr.id == trace.id:
    #             try:
    #                 _st = Stream(traces=[_tr, trace]).merge(**merge_kwargs)
    #                 if len(_st) == 1:
    #                     _tr = _st[0]
    #                     self.traces.update({label: _tr})
    #                 else:
    #                     raise AttributeError('traces failed to merge, despite passing compatability tests')
    #             except:
    #                 raise AttributeError('traces fail merge compatability tests')
    #         else:
    #             raise AttributeError('trace.id and self.trace[label].id do not match')
    #     else:
    #         self.traces.update({label: trace})
    #     return self
    
    # def append_stream(self, stream, labels, merge_kwargs={'method': 1, 'fill_value': None, 'pad': True, 'interpolation_samples': -1})


    # def sync_data(self):

        


    # def treat_gaps(self, )

    # def apply_stream_method(self, method, *args, **kwargs)