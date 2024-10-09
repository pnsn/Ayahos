
import warnings, copy
import numpy as np
# from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.trace import Trace, Stats
from PULSE.data.header import MLStats
from obspy.core.util.misc import flat_not_masked_contiguous

# FIXME: Track down processing metadata bleed

class FoldTrace(Trace):

    def __init__(self, data=None, fold=None, header=None, dtype=None):
        """Create a :class:`~PULSE.data.foldtrace.FoldTrace` object

        :param data: Input data array, defaults to np.array([])
            Note: will also accept a :class:`~obspy.core.trace.Trace` object
        :type data: numpy.ndarray or Trace, optional
        :param fold: fold vector of equal size as data, defaults to None
            None results in fold values of 1 for all data, except masked values
            which have a fold value of 0
        :type fold: None or numpy.ndarray, optional
        :param header: non-default metadata values, defaults to None
        :type header: None or dict, optional
        :param dtype: user-specified dtype to use for **data** and **fold**, defaults to None.
            None uses the native **dtype** of data
            Note: During initialization is the only time a FoldTrace object can
            have it's dtype set via a **dtype** argument. All subsequent changes are
            affected by changing the dtype of **data**.
        :type dtype: None or type, optional
        """
        # Compatability checks on data   
        if data is None:
            if dtype is None:
                self.dtype = np.float64
                data = np.array([], dtype=self.dtype) 
        elif isinstance(data, np.ndarray):
            self._data_sanity_checks(data)
            # Catch initial dtype - only time dtype can supercede data dtype
            if dtype is None:
                self.dtype = data.dtype
            else:
                self.dtype = dtype
                data = data.astype(dtype)
        # Handle case where data is Trace
        elif isinstance(data, Trace):
            trace = data
            if dtype is not None:
                self.dtype = dtype
                data = trace.data.astype(self.dtype)
            else:
                self.dtype = trace.data.dtype
                data = trace.data
            header = trace.stats
        else:
            raise TypeError('data must be a NumPy ndarray or ObsPy Trace-like object')
        # Init to inherit from Trace
        if header is None:
            header = {}
        elif isinstance(header, dict):
            pass
        elif isinstance(header, Stats):
            header = dict(header)
        else:
            raise TypeError('header must be type dict, ObsPy Stats, or NoneType')

        if fold is None:
            fold = np.ones(data.shape, dtype=self.dtype)
        elif isinstance(fold, np.ndarray):
            fold = fold.astype(self.dtype)
        else:
            raise TypeError('fold must be type NumPy ndarray or NoneType')

        # Initialize as empty Trace
        super().__init__()
        # Upgrade stats class & set values
        self.stats = MLStats(header)
        # Set data
        self.data = data
        # Check fold against self.data and self.dtype
        self._fold_sanity_checks(fold)
        # Set fold
        self.fold = self._enforce_fold_masking_rules(fold)
       
        
            
    def __setattr__(self, key, value):
        """__setattr__ method of FoldTrace object

        dtype heirarchy
        1) data.dtype
        2) dtype
        3) fold.dtype

        :param key: _description_
        :type key: _type_
        :param value: _description_
        :type value: _type_
        """
        if key == 'data':
            # Compatability check
            self._data_sanity_checks(value)
            # Ensure contiguous if on
            if self._always_contiguous:
                value = np.require(value, requirements=['C_CONTIGUOUS'])
            # Ensure dtype match
            if hasattr(self,'dtype'):
                self.dtype = value.dtype
            if hasattr(self,'fold'):
                if self.fold.dtype != value.dtype:
                    self.fold = self.fold.astype(value.dtype)
            if hasattr(self, 'stats'):
                self.stats.npts = len(value)
        # If setting fold
        elif key == 'fold':
            # Compatability check
            self._fold_sanity_checks(value)
            # Ensure contiguous if on
            if self._always_contiguous:
                value = np.require(value, requirements=['C_CONTIGUOUS'])
            # Ensure dtype match
            if self.dtype != value.dtype:
                value = value.astype(self.dtype)
        # Ensure that stats.npts remains consistent with len(data)
        elif key == 'stats':
            if hasattr(self, 'data'):
                value.npts = len(self.data)
            
        return super(FoldTrace, self).__setattr__(key,value)
    
    def __eq__(self, other):
        if not isinstance(other, FoldTrace):
            return False
        if not self.stats == other.stats:
            return False
        if not np.array_equal(self.data, other.data):
            return False
        if not np.array_equal(self.fold, other.fold):
            return False
        if not self.dtype == other.dtype:
            return False
        return True

    def __add__(self, other, method=0, fill_value=None, idtype=np.float64):
        """Add another Trace-like object to this FoldTrace
        Overwrites the functionalities of :meth:`~obspy.core.trace.Trace.__add__`
            
        :param other: FoldTrace object
        :type other: :class:`~PULSE.data.foldtrace.FoldTrace`
        :param method: method to apply, see above, defaults to 0
        :type method: int, optional
        :param fill_value: fill value passed to numpy.ma.MaskedArray if there are
            masked values, defaults to None
        :type fill_value: scalar, optional
        :param idtype: data type to use inside this operation to allow use
            of np.nan values, defaults to numpy.float64.
            Supported values:
                numpy.float32
                numpy.float64
        :type idtype: type, optional

        .. rubric:: Notable differences from Trace.__add__
         - Requires **other** to pass sanity checks
         - Updates **fold** attribute
         - Conducts an in-place modification to self
         - Fill value only populates the fill_value attribute of **data**
         if it is a :class:`~numpy.ma.MaskedArray`
         - method = 1 is not supported
         
        .. rubric:: Supported Methods

            All methods - gaps produce samples with masked data and 0-valued fold samples

            method = 0: drop overlapping samples
             - Drops overlapping data samples (become masked values)
             - thus overlapping data -> 0 fold
             - follows behavior method=0 for obspy Trace.__add__
            
            method = 1: simple interpolation for overlapping samples
             - Not supported - raises NotImplementedError

            method = 2: maximum-value stacking
             - Larger value of overlapping data samples is retained
             - fold of samples is added

            method = 3: fold-weighted average stacking
             - Fold-weighted mean of samples is evaluated for overlapping samples
             - fold of overlapping samples is added

        """
        # internal dtype catch
        if idtype in [np.float32, np.float64]:
            pass
        else:
            raise ValueError(f'idtype {idtype} not supported. Only numpy.float32 and numpy.float64') 
        # Run sanity checks
        if not isinstance(other, FoldTrace):
            raise TypeError
        if self.fold.dtype != other.fold.dtype:
            raise TypeError
        if self.data.dtype != other.data.dtype:
            raise TypeError
        if self.get_id() != other.get_id():
            raise TypeError
        if self.stats.sampling_rate != other.stats.sampling_rate:
            raise TypeError
        if self.stats.calib != other.stats.calib:
            raise TypeError
        
        # Create output holder
        output = self.__class__()
        # Get relative temporal positions of traces
        # Self leads other by starttime
        if self.stats.starttime < other.stats.starttime:
            lt = self
            rt = other
        elif self.stats.starttime > other.stats.starttime:
        # Other leads self by starttime
            lt = other
            rt = self
        # Starttimes match
        else:
            # Self leads other by endtime
            if self.stats.endtime < other.stats.endtime:
                lt = self
                rt = other
            # other leads self by endtime
            elif self.stats.endtime > other.stats.endtime:
                lt = other
                rt = self
            # starttime and endtime match (and sanity checks pass)
            else:
                # If data are identical, sum folds and return
                if all(self.data == other.data):
                    # Stats first, so npts can be updated by __setattr__('data')
                    output.stats = self.stats
                    output.data = self.data
                    # fold next, so masked samples can have fold set to 0 by __setattr__('data')
                    output.fold = output._enforce_fold_masking_rules(np.sum(np.c_[self.fold, other.fold], axis=1))
                    # __setattr__('data')
                    # output.data = self.data
                    # Make sure we didn't goof
                    output.verify()
                    return output
                # If data are not identical, but have all the same sample times
                # Apply merge rules as normal
                else:
                    lt = self
                    rt = other

        # Get relative indices for numpy array slicing & scaling (2, new_npts) data and fold
        i0 = None
        i1 = lt.stats.npts
        i2 = lt.stats.utc2nearest_index(rt.stats.starttime,ref='starttime')
        # Add one to conform to numpy slicing syntax
        i3 = lt.stats.utc2nearest_index(rt.stats.endtime,ref='starttime') + 1
        # Get maximum index value
        imax = np.max([i1, i3])
        # If lt has last sample, assign it's end index as None
        if i1 == imax:
            i1 = None
            new_npts = lt.stats.npts
        # If rt has last sample, assign it's end index as None
        elif i3 == imax:
            i3 = None
            new_npts = imax
        # If rt has first sample along with lt, set it's start index as None
        if i2 == 0:
            i2 = None
        # Create (2, new_npts) data array with default value of NaN
        data = np.full(shape=(2,new_npts), fill_value=np.nan, dtype=idtype)
        add_data = np.full(shape=(new_npts,), fill_value=np.nan, dtype=idtype)
        # Extract data vectors, convert to idtype, and pad masked values with NaN
        if isinstance(lt.data, np.ma.MaskedArray):
            ldata = lt.data.astype(idtype).filled(fill_value=np.nan)
        else:
            ldata = lt.data.astype(idtype)
        # Fold should already reflect the masked/not-masked
        lfold = lt.fold.astype(idtype)

        if isinstance(rt.data, np.ma.MaskedArray):
            rdata = rt.data.astype(idtype).filled(fill_value=np.nan)
        else:
            rdata = rt.data.astype(idtype)
        rfold = rt.fold.astype(idtype)

        data[0,i0:i1] = ldata
        data[1,i2:i3] = rdata
        # Create (2 by new_npts) fold array with default value of 0
        fold = np.full(shape=data.shape, fill_value=0, dtype=idtype)
        add_fold = np.full(shape=add_data.shape, fill_value=0, dtype=idtype)
        # Insert fold from both FoldTraces
        fold[0,i0:i1] = lfold
        fold[1,i2:i3] = rfold

        # Find gaps in data (including masked values)
        gaps = ~np.isfinite(data).any(axis=0)
        # Find overlaps in data
        overlaps = np.isfinite(data).all(axis=0)
        
        ## PROCESS FOLD IDENTICALLY FOR (almost) ALL METHODS
        # Find where there are gaps and set fold to 0, sum overlaps
        add_fold = np.sum(fold, axis=0)
        ## PROCESS DATA - DIFFERENTLY DEPENDING ON METHOD
        # FOR ALL - assign single values as values, otherwise assign nan
        add_data = np.where(gaps | overlaps, np.nan, np.nansum(data, axis=0))

        # Method 0) Behavior of Trace.__add__(method=0)
        if method == 0:
            # Merge the finite values where there aren't overlaps or gaps
            # add_data = np.where(gaps | overlaps, np.nan, np.nansum(data, axis=0))
            pass
        # Method 1) Behavior of Trace.__add__(method=1)
        if method == 1:
            raise NotImplementedError('ObsPy simple overlap interpolation not used for PULSE')

        # Method 2) Max stacking
        if method == 2:
            # Fill overlaps with max values
            if any(overlaps):
                add_data = np.where(overlaps, np.nanmax(data,axis=0), add_data)

        # Method 3) Fold-weighted stacking
        if method == 3:
            # Find where there are overlaps and use fold to get the weighted average
            if any(overlaps):
                add_data = np.where(overlaps, np.sum(data*fold, axis=0)/add_fold, add_data)
        
        # CLEANUP
        # Enforce masked array if gaps exist
        if ~np.isfinite(add_data).all():
            add_data = np.ma.MaskedArray(data=add_data,
                                         mask=~np.isfinite(add_data),
                                         fill_value=fill_value)
        # Assign data with reverted dtype & fold (with implicity dtype reversion)
        # Assign stats first so stats.npts can be updated with __setattr__('data')
        output.stats = lt.stats.copy()
        output.data = add_data.astype(lt.dtype)
        # Assign fold next so fold for masked samples can be updated to 0 with __setattr__('data')
        output.fold = output._enforce_fold_masking_rules(add_fold)
        # # Finally, update data & apply knock-on changes from __setattr__
        # output.data = add_data.astype(lt.dtype)
        return output

    # POLYMORPHIC METHODS AFFECTING DATA AND FOLD
    def taper(self, max_percentage, type='hann', max_length=None, side='both', **kwargs):
        fold_tr = self._get_fold_trace()
        Trace.taper(fold_tr, max_percentage, type=type, max_length=max_length, side=side, **kwargs)
        Trace.taper(self, max_percentage, type=type, max_length=max_length, side=side, **kwargs)
        return self

    def interpolate(self, sampling_rate, method='weighted_average_slopes',
                    starttime=None, npts=None, time_shift=0.0,
                    *args, **kwargs):
        # Grab initial stats info
        if np.ma.is_masked(self.data):
            raise Exception('masked data - try using FoldTrace.apply_to_gappy.')
        old_stats = self.stats.copy()
        Trace.interpolate(self,
                          sampling_rate,
                          starttime=starttime,
                          method=method,
                          npts=npts,
                          time_shift=time_shift,
                          *args,
                          **kwargs)
        self._interp_fold(old_stats.starttime, old_stats.sampling_rate)
        return self

    def resample(self, sampling_rate, window='hann', no_filter=True, strict_length=False):
        if np.ma.is_masked(self.data):
            raise Exception('masked data - try using FoldTrace.apply_to_gappy.')
        old_stats = self.stats.copy()
        Trace.resample(self,
                       sampling_rate,
                       window=window,
                       no_filter=no_filter,
                       strict_length=strict_length)
        self._interp_fold(old_stats.starttime, old_stats.sampling_rate)
        return self
    
    def decimate(self, factor, no_filter=False, strict_length=False):
        if np.ma.is_masked(self.data):
            raise Exception('masked data - try using FoldTrace.apply_to_gappy')
        old_stats = self.stats.copy()
        Trace.decimate(self, factor, no_filter=no_filter, strict_length=strict_length)
        self._interp_fold(old_stats.starttime, old_stats.sampling_rate)
        return self

    def _interp_fold(self, old_starttime, old_sampling_rate):
        """Use linear interpolation and 0 padding to resample the **fold** of a FoldTrace
        that has already had it's data resampled via an ObsPy resampling method.

        :param old_starttime: the start time of the FoldTrace before **data** was resampled
        :type old_starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param old_sampling_rate: the sampling rate of the FoldTrace before **data** was resampled
        :type old_sampling_rate: float-like
        """        
        # Create an obspy trace to house fold data
        trf = Trace(data=self.fold.copy(), header={'starttime': old_starttime,
                                            'sampling_rate': old_sampling_rate})
        # Reconstitute necessary metadata values
        old = trf.stats
        new = self.stats
        # old_npts = trf.stats.npts
        # old_endtime = old_starttime + old_npts/old_sampling_rate
        # old_delta = 1./old_sampling_rate
        # new_starttime = self.stats.starttime
        # new_endtime = self.stats.endtime
        # new_delta = self.stats.delta
        # new_npts = self.stats.npts
        # Apply 0-padding if needed
        if old.starttime > new.starttime:
            trf._ltrim(new.starttime - 1./old_sampling_rate,
                       pad=True, fill_value=0,
                       nearest_sample=False)
        if old.endtime < new.endtime:
            trf._rtrim(new.endtime + 1./old_sampling_rate,
                       pad=True, fill_value=0,
                       nearest_sample=False)
        # Just make dt vectors referenced to old_starttime and do numpy interpolation
        new_dt_vect = np.array([new.starttime - old.starttime + x*new.delta for x in range(new.npts)])
        old_dt_vect = np.array([x*old.delta for x in range(old.npts)])
        # Make sure the array isn't masked & set masked values to 0
        if isinstance(trf, np.ma.MaskedArray):
            old_fold = trf.data.filled(0)
        else:
            old_fold = trf.data

        self.fold = np.interp(new_dt_vect,old_dt_vect, old_fold)

        return self
        


        


    # # def resample_fold(self, sampling_rate, starttime, endtime, method='linear', **kwargs):
    # #     fold_tr = self._get_fold_trace().copy()
    # #     old_delta = self.stats.delta
    # #     new_delta = 1./sampling_rate
    # #     # If new starttime is before original starttime
    # #     if fold_tr.stats.starttime > starttime:
    # #         # pad out by one extra sample with 0 fold
    # #         fold_tr._ltrim(starttime - old_delta,
    # #                        nearest_sample=False,
    # #                        pad=True, fill_value=0)
    # #     # If new endtime is after original endtime
    # #     if fold_tr.stats.endtime < endtime:
    # #         # pad out by one extra sample with 0 fold
    # #         fold_tr._rtrim(endtime + old_delta,
    # #                        nearest_sample=False,
    # #                        pad=True,
    # #                        fill_value=0)
    # #     # Interpolate
    # #     Trace.interpolate(fold_tr,
    # #                       sampling_rate,
    # #                       starttime=starttime,
    # #                       method=method,
    # #                       **kwargs)


    # # def interpolate_fold(self, sampling_rate, starttime=None, npts=None):
    # #     """Resample the **fold** of this FoldTrace object using linear
    # #     interpolation - used uniformly with all resampling methods inherited
    # #     from ObsPy Trace
    # #      - :meth:`~PULSE.data.foldtrace.FoldTrace.interpolate`
    # #      - :meth:`~PULSE.data.foldtrace.FoldTrace.resample`
    # #      - :meth:`~PULSE.data.foldtrace.FoldTrace.decimate`

    # #     :param sampling_rate: new reference sampling rate
    # #     :type sampling_rate: float-like
    # #     :param starttime: reference start time for interpolation, default is None.
    # #         None input uses the starttime of this FoldTrace
    # #     :type starttime: :class:`~obspy.core.utctdatetime.UTCDateTime`
    # #     :param npts: number of points for the interpolated data, default is None.
    # #         None input uses the endtime of this FoldTrace to estimate npts
    # #         such that the nearest sample occurs at or before the endtime
    # #     :return:
    # #      - **int_fold** (*numpy.ndarray*) -- interpolated fold vector
    # #     """
    # #     # sampling_rate compatability checks
    # #     if not isinstance(sampling_rate, (int, float)):
    # #         raise TypeError
    # #     elif sampling_rate <= 0:
    # #         raise ValueError
    # #     elif not np.isfinite(sampling_rate):
    # #         raise ValueError
    # #     else:
    # #         old_sr = self.stats.sampling_rate
    # #         old_delta = 1./old_sr
    # #         new_sr = sampling_rate
    # #         new_delta = 1./new_sr
    # #     # starttime compatability checks
    # #     if starttime is None:
    # #         starttime = self.stats.starttime
    # #     elif not isinstance(starttime, UTCDateTime):
    # #         raise TypeError('starttime must be UTCDateTime or None')
            
    # #     old_dt = self.stats.endtime - starttime + old_delta
    # #     max_npts = np.floor(old_dt*sampling_rate)
    # #     if npts is None:
    # #         npts = max_npts
    # #     elif npts > max_npts:
    # #         raise ValueError(f'specified npts {npts} would exceed the endtime of this FoldTrace (max: {max_npts})')
        
    # #     new_endtime = starttime + npts/sampling_rate
        
    # #     new_dt = new_endtime - starttime
    # #     old_times = np.arange(0, old_dt + old_delta, old_delta)
    # #     new_times = np.arange(0, new_dt + new_delta, new_delta)
    # #     breakpoint()

    # #     int_fold = np.interp(new_times, old_times, self.fold)
    # #     return int_fold


    # def interpolate(self,
    #                 sampling_rate,
    #                 method='weighted_average_slopes',
    #                 starttime=None,
    #                 npts=None,
    #                 time_shift=0.0,
    #                 *args, **kwargs):
    #     """Use the ObsPy Trace :meth:`~obspy.core.trace.Trace.interpolate` method on this
    #     FoldTrace object and resample the **fold** using linear interpolation version of 
    #     the same method. 

    #     :param sampling_rate: new sampling rate
    #     :type sampling_rate: float-like
    #     :param method: interpolation method, defaults to 'weighted_average_slopes'
    #         Supported methods: see :meth:`~obspy.core.trace.Trace.interpolate` for full descriptino
    #         - "lanczos" - (Sinc interpolation) - highest quality, but computationally costly
    #         - "weighted_average_slopes" - SAC standard
    #         - "slinear" - 1st order spline
    #         - "quadratic" - 2nd order spline
    #         - "cubic" - 3rd order spline
    #         - "linear" - linear interpolation (always used to interpolate **fold**)
    #         - "nearest" - nearest neighbor
    #         - "zero" - last encountered value
    #     :type method: str, optional
    #     :param starttime: Start time for the new interpolated FoldTrace, defaults to None
    #     :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
    #     :param npts: new number of samples, defaults to None
    #     :type npts: int, optional
    #     :param time_shift: Shift the trace by a specified number of seconds, defaults to 0.0
    #         see :meth:`~obspy.core.trace.Trace.interpolate` for more information
    #     :type time_shift: float, optional
    #     :param fold_density: Should up-sampling result in a scalar reduction of **fold**? Defaults to False
    #         scalar = sampling_rate / original sampling_rate, if 0 < scalar < 1, else scalar = 1
    #     :type fold_density: bool, optional
    #     """        
    #     Trace.interpolate(self,
    #                       sampling_rate,
    #                       method=method,
    #                       starttime=starttime,
    #                       npts=npts,
    #                       time_shift=time_shift,
    #                       *args, **kwargs)
    #     # Interpolate fold using updated stats from **data** processing
    #     self.fold = self.interpolate_fold(sampling_rate,
    #                                       starttime= self.stats.starttime,
    #                                       npts = self.stats.npts)

    #     self.verify()
    #     return self

    # def resample(self,
    #     sampling_rate,
    #     window='hann',
    #     no_filter=True,
    #     strict_length=False,
    #     **kwargs):
    #     """Apply the ObsPy Trace :meth:`~obspy.core.trace.Trace.resample` method to this FoldTrace
    #     object. Fold is resampled using linear interpolation.

    #     :param sampling_rate: _description_
    #     :type sampling_rate: _type_
    #     :param window: _description_, defaults to 'hann'
    #     :type window: str, optional
    #     :param no_filter: _description_, defaults to True
    #     :type no_filter: bool, optional
    #     :param strict_length: _description_, defaults to False
    #     :type strict_length: bool, optional
    #     :param fold_taper: _description_, defaults to 0.05
    #     :type fold_taper: float, optional
    #     :return: _description_
    #     :rtype: _type_
    #     """
    #     ts0 = self.stats.starttime
    #     te0 = self.stats.endtime
    #     npts0 = self.stats.npts
    #     sr0 = self.stats.sampling_rate
    #     dt = te0 - ts0 + (1./sr0)
    #     npts1 = int(np.floor(dt*sampling_rate))
    #     # Resample fold first
    #     self.fold = self.interpolate_fold(sampling_rate, starttime=ts0, npts=npts1)
    #     # Resample data
    #     Trace.resample(
    #         self,
    #         sampling_rate,
    #         window=window,
    #         no_filter=no_filter,
    #         strict_length=strict_length,
    #         **kwargs)
    #     return self

    # POLYMORPHIC METHODS - DATA ONLY
    # def detrend(self, type='simple', **options):
    #     options.update({'type': type})
    #     super().detrend(self, **options)
    #     self.fold = self._enforce_fold_masking_rules(self.fold)
    #     self.verify()
    #     return self


    # def 


    ##################################
    # DYNAMIC CALL SUPPORTER METHODS #
    ##################################

    def apply_method(self, method, apply_to_fold=False, **options):
        method = getattr(super(), method)
        if apply_to_fold:
            fold_ftr = self._get_fold_view_trace()
            getattr(fold_ftr, method)(**options)
        getattr(self, method)(**options)
        if apply_to_fold:
            self.fold = fold_ftr.data
        # Add a quick note on if fold processing was applied
        self.stats.processing[-1] += f' PULSE: apply_method(apply_to_fold={apply_to_fold})'
        
    def apply_to_gappy(self, method, **options):
        if isinstance(self.data, np.ma.MaskedArray):
            fill_value = self.data.fill_value
            st = self.split()
            for _e, _ftr in enumerate(st):
                _ftr.apply_method(method, **options)
                if _e == 0:
                    self = _ftr
                else:
                    self.__add__(_ftr, method=0, fill_value=fill_value)
            # Add a note that this used apply_to_gappy        
            self.stats.processing[-1] += f' PULSE: apply_to_gappy()'
        else:
            self.apply_method(method, **options)
    
    def __iadd__(self, other, **options):
        """Provide rich implementation of += operator

        :param other: _description_
        :type other: _type_
        """        
        added = self.__add__(other, **options)
        self.stats = added.stats
        self.data = added.data
        self.fold = added.fold
        return self

    def __repr__(self, id_length=None):
        """Provide a human readable string describing the contents of this
          :class:`~PULSE.data.foldtrace.FoldTrace` object

        :param id_length: maximum ID length passed to the inherited 
            :meth:`~obspy.core.trace.Trace.__str__` method, defaults to None
        :type id_length: None or int, optional
        :return rstr: representative string
        :rtype: str
        """        
        rstr = super().__str__(id_length=id_length)
        if self.stats.npts > 0:
            rstr += f' | Fold:'
            for _i in range(int(self.fold.max()) + 1):
                ff = sum(self.fold == _i)/self.stats.npts
                if ff > 0:
                    rstr += f' [{_i}] {ff:.2f}'
        return rstr

    ######################
    # VIEW-BASED METHODS #
    ######################

    def get_view(self, starttime=None, endtime=None):
        """Create a new FoldTrace object that houses subset views
        of this FoldTrace's **data** and **fold** attributes and
        a deepcopy of this FoldTrace's **stats** with updated 
        timing information.

        **Warning: Alterations to the contents of views alter the original data**
        
        This can be memory-adventageous because it does not
        produce a copy of the **data** and **fold** vectors in
        memory, rather it provides a *view* of these data vectors.

        In combination with the :meth:`~obspy.core.trace.Trace.copy()`
        method, this provides improved efficiency for creating subset copies
        of the contents of a FoldTrace.

        
        :param starttime: starttime of view, defaults to None
        :type starttime: None or :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: endtime of view, defaults to None
        :type endtime: None or :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :return:
         - **ftr** (*PULSE.data.foldtrace.FoldTrace**) -- FoldTrace containing the subset
            **data** and **fold** views with an updated copy of metadata in **stats**
        """        
        # Get starting and ending indices
        if starttime is None:
            ii = 0
        else:
            ii = self.stats.utc2nearest_index(starttime,ref='starttime')
        if ii < 0:
            ii = 0
        if endtime is None:
            ff = self.stats.npts - 1
        else:
            ff = self.stats.utc2nearest_index(endtime,ref='endtime')
        if ff > self.stats.npts:
            ff = self.stats.npts - 1
        # Create deep copy of header
        header = self.stats.copy()
        # Get starttime of view
        header.starttime = self.stats.starttime + ii*self.stats.sampling_rate
        # # Get number of samples of view (updates endtime)
        header.npts = ff - ii + 1
        # Generate new trace with views and copied metadata
        ftr = FoldTrace(data=self.data[ii:ff],
                        fold=self.fold[ii:ff],
                        header=header)
        return ftr
    
    #################################
    # FOLD-SPECIFIC PRIVATE METHODS #
    #################################

    def _enforce_fold_masking_rules(self, value):
        """Enforce rules for fold related to masked arrays
        for a candidate fold value

        Rule 1) masked data are automatically 0 fold, but not vice versa
        Rule 2) If fold has masked values, fill them with 0's
        Rule 3) fold.dtype matches FoldTrace.dtype
        """
        # Enforce masked data values = 0 fold
        if isinstance(self.data, np.ma.MaskedArray):
            value[self.data.mask] = 0
        # Enforce masked fold values -> 0 fold
        if isinstance(value, np.ma.MaskedArray):
            value.fill_value = 0
            value = value.filled()
        if value.dtype != self.dtype:
            value = value.astype(self.dtype)
        return value

    def _data_sanity_checks(self, value):
        """Adapted version of obspy's _data_sanity_checks
        to be consistent with error raise types
        """
        if not isinstance(value, np.ndarray):
            raise TypeError('FoldTrace.data must be type np.ndarray')
        if value.ndim != 1:
            msg = f'NumPy array for FoldTrace.data has bad shape ({value.shape}). '
            msg += 'Only 1-d arrays are allowed.'
            raise ValueError(msg)

    def _fold_sanity_checks(self, value):
        """Complement to _data_sanity_checks from obspy.core.trace

        Check fold rules:
        0) fold must be numpy.ndarray
        1) fold shape must match data shape
        """        
        if not isinstance(value, np.ndarray):
            raise TypeError('FoldTrace.fold must be a numpy.ndarray')   
        if self.data.shape != value.shape:
            raise ValueError('FoldTrace.fold shape must match FoldTrace.data shape')

    def _get_fold_trace(self):
        """Create a new FoldTrace object with data that houses
        a view of this FoldTrace's fold values and copied metadata

        :return:
         - **ftr** (*PULSE.data.foldtrace.FoldTrace*) - FoldTrace object containing 
            a view of the **fold** of the source FoldTrace object as its **data** 
            attribute and a deep copy of the source FoldTrace's **stats** attribute
        """        
        ftr = FoldTrace(data=self.fold, header=self.stats.copy())
        return ftr

    def _ltrim(self, starttime, pad=False, nearest_sample=True, fill_value=None):
        npts_old = self.stats.npts
        # Trim Data
        Trace._ltrim(self, starttime, pad=pad, nearest_sample=nearest_sample, fill_value=fill_value)
        npts_new = self.stats.npts
        # For shortened vectors
        if npts_old > npts_new:
            dn = npts_old - npts_new
            self.fold = self.fold[dn:]
        # For lengthened vectors - all padding samples get fold = 0
        elif npts_old < npts_new:
            dn = npts_new - npts_old
            new_fold = np.zeros(self.data.shape, dtype=self.dtype)
            new_fold[dn:] += self.fold
            self.fold = new_fold
        else:
            pass
        self.verify()
        return self
    
    def _rtrim(self, endtime, pad=False, nearest_sample=True, fill_value=None):
        npts_old = self.stats.npts
        Trace._rtrim(self, endtime, pad=pad, nearest_sample=nearest_sample, fill_value=fill_value)
        npts_new = self.stats.npts
        if npts_old > npts_new:
            self.fold = self.fold[:npts_new]
        elif npts_old < npts_new:
            new_fold = np.zeros(self.data.shape, dtype=self.dtype)
            new_fold[:npts_old] += self.fold
            self.fold = new_fold
        else:
            pass
        self.verify()
        return self
    
    def trim(self, starttime=None, endtime=None, pad=False, nearest_sample=True, fill_value=None):
        super().trim(starttime=starttime,
                     endtime=endtime,
                     pad=pad,
                     nearest_sample=nearest_sample,
                     fill_value=fill_value)
        self.verify()
        return self
    
        


    def verify(self):
        super().verify()
        if isinstance(self.fold, np.ndarray) and\
            self.fold.dtype.byteorder not in ['=', '|']:
            raise Exception('FoldTrace fold should be stored as np.ndarray in the system specific byte order')
        if self.fold.shape != self.data.shape:
            raise Exception('FoldTrace.data shape and FoldTrace.fold shape mismatch %s != %s'%\
                            (str(self.data.shape), str(self.fold.shape)))
        if self.data.dtype != self.fold.dtype:
            raise Exception('FoldTrace.data dtype and FoldTrace.fold dtype mismatch %s != %s'%\
                            (str(self.data.dtype), str(self.fold.dtype)))
        return True

    # PROPERTY METHODS AND ASSIGNMENTS #

    def get_ml_id(self):
        rstr = self.get_id()
        rstr += f'.{self.stats.model}.{self.stats.weight}'
        return rstr
    
    mlid = property(get_ml_id)

    
    # def resample(self, sampling_rate, window='hann', no_filter=True, strict_length=False):
    #     # Resample fold first
    #     ftr = self._get_fold_view_trace()
    #     Trace.resample(
    #             ftr,
    #             sampling_rate,
    #             window=window,
    #             no_filter=no_filter,
    #             strict_length=strict_length)
    #     breakpoint()
    #     # Resample data next
    #     Trace.resample(
    #         sampling_rate,
    #         window=window,
    #         no_filter=no_filter,
    #         strict_length=strict_length)
    #     # Re-assign fold-trace to fold
    #     self.fold = ftr.data
    #     # Round fold to suppresss Gibbs phenomena
    #     self.fold = self.fold.round()
    #     self.stats.processing[-1] += f' PULSE: fold = resample + round'


    # def apply_inplace_method_to_masked(self, method, *args, process_fold=True, **kwargs):
    #     """Conduct a dynamic call of a method from :class:`~obspy.core.trace.Trace`
    #     with internal handling of gappy (i.e., masked) **data** not supported by
    #     most ObsPy Trace class methods. 

    #     Provides the option to also process the **fold** vector with the same method.
        
    #     Updates to FoldTrace.stats.processing are only propagated from the application of the
    #     method to **data**. 

    #     NOTE: This method assumes that the method applies changes to this object IN-PLACE
    #     and does not capture or return method outputs.

    #     :param method: name of the method to use
    #     :type method: str
    #     :param *args: collector for addtional positional arguments of **method**
    #     :param process_fold: should the **fold** attribute also be processed? Defaults to True.
    #     :type process_fold: bool, optional
    #     :param **kwargs: collector for additional key-word arguments for **method**
    #     """        
    #     # If the data vector is masked
    #     if isinstance(self.data, np.ma.MaskedArray):
    #         # Presserve it's fill_value
    #         fill_value = self.data.fill_value
    #         # Split into multiple, self-contiguous foldtraces
    #         st = self.split()
    #         # Iterate corss foldtraces to apply method individually
    #         for _e, _ftr in enumerate(st):
    #             _ftr._apply_inplace_method(method, *args, process_fold=process_fold, **kwargs)
    #             # If first iteration, redefine this foldtrace as the first foldtrace
    #             if _e == 0:
    #                 self = _ftr
    #             # all subsequent iterations, use __add__ to extend the first foldtrace
    #             else:
    #                 self.__add__(_ftr, method=0, fill_value=fill_value)
    #     # Otherwise use the method as normal
    #     else:
    #         self._apply_inplace_method(method, *args, process_fold=process_fold, **kwargs)

    
    

    # ###################################
    # # POLYMORPHIC "DATA ONLY" METHODS #
    # ###################################

    # def filter(self, type, **options):
    #     """Applys the :meth:`~obspy.core.trace.Trace.filter` method to the 
    #     **data** and **stats** attributes of this FoldTrace, allowing for
    #     masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

    #     :param type: type of filter to apply
    #     :type type: str
    #     :param **options: key-word argument collector for the ObsPy method.
    #         see :meth:`~obspy.core.trace.Trace.filter`
    #     """
    #     self.apply_inplace_method_to_masked('filter', type, **options)
        
    # def detrend(self, **options):
    #     """Applys the :meth:`~obspy.core.trace.Trace.detrend` method to the 
    #     **data** and **stats** attributes of this FoldTrace, allowing for
    #     masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

    #     :param **options: key-word argument collector for the ObsPy method.
    #         see :meth:`~obspy.core.trace.Trace.detrend`
    #     """
    #     self.apply_inplace_method_to_masked('detrend', process_fold=False, **options)
    
    # def differentiate(self, **options):
    #     """Applys the :meth:`~obspy.core.trace.Trace.differentiate` method to the 
    #     **data** and **stats** attributes of this FoldTrace, allowing for
    #     masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

    #     :param **options: key-word argument collector for the ObsPy method.
    #         see :meth:`~obspy.core.trace.Trace.differentiate`
    #     """
    #     self.apply_inplace_method_to_masked('differentiate', process_fold=False, **options)
    
    # def integrate(self, **options):
    #     """Applys the :meth:`~obspy.core.trace.Trace.integrate` method to the 
    #     **data** and **stats** attributes of this FoldTrace, allowing for
    #     masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

    #     :param **options: key-word argument collector for the ObsPy method.
    #         see :meth:`~obspy.core.trace.Trace.integrate`
    #     """
    #     self.apply_inplace_method_to_masked('integrate', process_fold=False, **options)

    # def remove_response(self, **options):
    #     self.apply_inplace_method_to_masked('remove_response', process_fold=False, **options)

    # def remove_sensitivity(self, **options):
    #     """Applys the :meth:`~obspy.core.trace.Trace.remove_sensitivity` method to the 
    #     **data** and **stats** attributes of this FoldTrace, allowing for
    #     masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

    #     :param **options: key-word argument collector for the ObsPy method.
    #         see :meth:`~obspy.core.trace.Trace.remove_sensitivity`
    #     """
    #     self.apply_inplace_method_to_masked('remove_sensitivity', process_fold=False, **options)

    # def taper(self, max_percentage, **options):
    #     """Applys the :meth:`~obspy.core.trace.Trace.taper` method to the 
    #     **data** and **stats** attributes of this FoldTrace, allowing for
    #     masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

    #     :param **options: key-word argument collector for the ObsPy method.
    #         see :meth:`~obspy.core.trace.Trace.taper`
    #     """
    #     self.apply_inplace_method_to_masked('taper', max_percentage, process_fold=False, **options)

    # def trigger(self, type, **options):
    #     """Applys the :meth:`~obspy.core.trace.Trace.trigger` method to the 
    #     **data** and **stats** attributes of this FoldTrace, allowing for
    #     masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`
        
    #     :param type: trigger type to use
    #     :type type: str
    #     :param **options: key-word argument collector for the ObsPy method.
    #         see :meth:`~obspy.core.trace.Trace.trigger`
    #     """
    #     self.apply_inplace_method_to_masked('trigger', type, process_fold=False, **options)

    # #######################################
    # # POLYMORPHIC "DATA AND FOLD" METHODS #
    # #######################################

    # def decimate(self, factor, **options):
    #     """Applys the :meth:`~obspy.core.trace.Trace.decimate` method to the 
    #     **data**, **fold**, and **stats** attributes of this FoldTrace, allowing for
    #     masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`
    #     :param factor: factor by which sampling rate is lowered by decimation
    #     :type factor: int
    #     :param **options: key-word argument collector for the ObsPy method.
    #         see :meth:`~obspy.core.trace.Trace.decimate`
    #     """
    #     self.apply_inplace_method_to_masked('decimate', factor, process_fold=True, **options)

    # def interpolate(self, sampling_rate, **options):
    #     """Applys the :meth:`~obspy.core.trace.Trace.interpolate` method to the 
    #     **data**, **fold**, and **stats** attributes of this FoldTrace, allowing for
    #     masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

    #     :param sampling_rate: new sampling rate in Hz 
    #     :type sampling_rate: float
    #     :param **options: key-word argument collector for the ObsPy method.
    #         see :meth:`~obspy.core.trace.Trace.interpolate`
    #     """
    #     self.apply_inplace_method_to_masked('interpolate', sampling_rate, process_fold=True, **options)

    # def resample(self, sampling_rate, **options):
    #     """Applys the :meth:`~obspy.core.trace.Trace.resample` method to the 
    #     **data**, **fold**, and **stats** attributes of this FoldTrace, allowing for
    #     masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`
        
    #     :param sampling_rate: new sampling rate in Hz 
    #     :type sampling_rate: float
    #     :param **options: key-word argument collector for the ObsPy method.
    #         see :meth:`~obspy.core.trace.Trace.resample`
    #     """
    #     self.apply_inplace_method_to_masked('resample', sampling_rate, process_fold=True, **options)

    # def split(self):
    #     """Split a masked FoldTrace into a :class:`~obspy.core.stream.Stream` object
    #     containing non-masked, self-contiguous FoldTrace objects

    #     :raises NotImplementedError: _description_
    #     :return:
    #      - **output** (*obspy.core.stream.Stream*) -- stream of FoldTrace segments
    #         copied from this FoldTrace
    #     """        
    #     output = Stream()
    #     # Data is not a masked array
    #     if not isinstance(self.data, np.ma.MaskedArray):
    #         output += self.copy()
    #     # Data is a masked array
    #     else:
    #         # No data are masked - convert back to unmasked array
    #         if not np.ma.is_masked(self.data):
    #             self.data = self.data.data
    #             output += self.copy()
    #         else:
    #             # Generate slices
    #             slices = flat_not_masked_contiguous(self.data)
    #             output = Stream()
    #             for slice in slices:
    #                 if slice.step:
    #                     raise NotImplementedError("step not supported")
    #                 stats = self.stats.copy()
    #                 ftr = FoldTrace(header=stats)
    #                 ftr.stats.starttime += (stats.delta*slice.start)
    #                 ftr.data = self.data.data[slice.start:slice.stop]
    #                 ftr.fold = self.fold[slice.start:slice.stop]
    #                 output += ftr
    #     return output

    # def _ltrim(self, starttime, **options):
    #     self._apply_inplace_method('_ltrim', starttime, process_fold=True, **options)

    # def _rtrim(self, endtime, **options):
    #     self._apply_inplace_method('_rtrim', endtime, process_fold=True, **options)


    # #########################
    # # ADDED PRIVATE METHODS #
    # #########################

    # def _apply_inplace_method(self, method, *args, process_fold=True, **kwargs):
    #     """Conduct a dynamic call of a method from :class:`~obspy.core.trace.Trace` with
    #     the option to also process the **fold** vector with the same method. Updates
    #     to FoldTrace.stats.processing are only propagated from the application of the
    #     method to **data**

    #     NOTE: This method assumes that the method applies changes to this object IN-PLACE
    #     and does not capture or return method outputs.

    #     :param method: name of the method to use
    #     :type method: str
    #     :param *args: collector for addtional positional arguments of **method**
    #     :param process_fold: should the **fold** attribute also be processed? Defaults to True.
    #     :type process_fold: bool, optional
    #     :param **kwargs: collector for additional key-word arguments for **method**
    #     """        
    #     # If processing fold too, create fold_view_trace
    #     if process_fold:
    #         ftr = self._get_fold_view_trace()
    #     # Process for data & metadata
    #     getattr(self, method)(*args, **kwargs)
    #     if process_fold:
    #         getattr(ftr, method)(*args, **kwargs)
    


                         

