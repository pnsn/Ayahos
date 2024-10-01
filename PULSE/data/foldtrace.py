
import numpy as np
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.trace import Trace, Stats, _data_sanity_checks
from PULSE.data.header import MLStats
from obspy.core.util.misc import flat_not_masked_contiguous

class FoldTrace(Trace):

    def __init__(self, data=np.array([]), fold=None, header=None, dtype=None):
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
        :param dtype: default data type to use for **data** and **fold**, defaults to np.float32
        :type dtype: type, optional
        """        
        # Handle case where a Trace is directly passed as an arg
        if isinstance(data, Trace):
            trace = data
            data = trace.data
            header = trace.stats
        # Initialize as trace with no header
        super().__init__(data=data,header=None)
        if dtype is not None:
            self.data = self.data.astype(dtype)
        # Upgrade Stats to MLStats
        if header is None:
            header = {}
        self.stats = MLStats(header=header)
        # Populate fold
        self.fold = np.ones(len(self))
        if fold is not None:
            self.__setattr__('fold', fold)



    def __setattr__(self, key, value):
        """__setattr__ method of FoldTrace object

        :param key: _description_
        :type key: _type_
        :param value: _description_
        :type value: _type_
        """
        # Any change in FoldTrace.data will dynamically set Trace.stats.npts
        if key == 'data':
            _data_sanity_checks(value)
            if self._always_contiguous:
                value = np.require(value, requirements=['C_CONTIGUOUS'])
            self.stats.npts = len(value)
            # Ensure that changes in data dtype is propagated to fold dtype
            self._enforce_fold_rules()
        # Any change in FoldTrace.fold cannot violate fold rules
        elif key == 'fold':
            # Run fold sanity checks
            value = self._enforce_fold_rules(value)
    
        return super(FoldTrace, self).__setattr__(key,value)




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


    ###################
    # SPECIAL METHODS #
    ###################
    def extend(self, other, fill_value=None):
        self.__add__(other, method=0, fill_value=fill_value)

    def max_stack(self, other, fill_value=None):
        self.__add__(other, method=2, fill_value=fill_value)
    
    def avg_stack(self, other, fill_value=None):
        self.__add__(other, method=3, fill_value=fill_value)


    def __add__(self, other, method=0, fill_value=None):
        """Add another Trace-like object to this FoldTrace
        Overwrites the functionalities of :meth:`~obspy.core.trace.Trace.__add__`
            
        :param other: FoldTrace object
        :type other: :class:`~PULSE.data.foldtrace.FoldTrace`
        :param method: method to apply, see above, defaults to 0
        :type method: int, optional
        :param fill_value: fill value passed to numpy.ma.MaskedArray if there are
            masked values, defaults to None
        :type fill_value: scalar, optional

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
        
        # Get relative temporal positions of traces
        # Self leads other by starttime
        if self.stats.starttime < other.stats.starttime:
            lt = self
            rt = other
        elif self.stats.startime > other.stats.starttime:
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
                if self.data == other.data:
                    self.fold = np.sum(np.c_[self.fold, other.fold],axis=1)
                    return
                # If data are not identical, proceed with merge
                else:
                    lt = self
                    rt = other
        # Create output holder
        output = self.__class__(header=lt.stats.copy())
        # Get relative indices
        i1 = lt.stats.npts
        i2 = lt.stats.utc2nearest_index(rt.stats.starttime,ref='starttime')
        i3 = lt.stats.utc2nearest_index(rt.stats.endtime,ref='starttime')
        # Create (2, new_npts) data array with default value of NaN
        data = np.full(shape=(2,i3+1), fill_value=np.nan, dtype=self.data.dtype)
        add_data = np.full(shape=i3+1, fill_value=np.nan, dtype=self.data.dtype)
        # Insert data from both FoldTraces
        data[0,:i1] = lt.data
        data[1,i2:] = rt.data
        # Create (2 by new_npts) fold array with default value of 0
        fold = np.full(shape=data.shape, fill_value=0, dtype=self.data.dtype)
        add_fold = np.full(shape=add_data.shape, fill_value=0, dtype=self.data.dtype)
        # Insert fold from both FoldTraces
        fold[0,:i1] = lt.fold
        fold[1,i2:] = rt.fold
        # Find gaps in data
        gaps = ~np.isfinite(data).any(axis=0)
        # Find overlaps in data
        overlaps = np.isfinite(data).all(axis=0)
        
        ## PROCESS FOLD IDENTICALLY FOR (almost) ALL METHODS
        # Find where there are gaps and set fold to 0, sum overlaps
        add_fold = np.sum(fold, axis=0)

        ## PROCESS DATA - DIFFERENTLY DEPENDING ON METHOD
        # Method 0) Behavior of Trace.__add__(method=0)
        if method == 0:
            # Merge the finite values where there aren't overlaps or gaps
            add_data[~gaps & ~overlaps] = np.nansum(data, axis=0)
        # Method 1) Behavior of Trace.__add__(method=1)
        if method == 1:
            raise NotImplementedError('ObsPy simple overlap interpolation not used for PULSE')

        # Method 2) Max stacking
        if method == 2:
            # Find where therere are gaps and set them to np.nan
            add_data[~gaps] = np.nansum(data, axis=0)

        # Method 3) Fold-weighted stacking
        if method == 3:
            # Set standard values where there are no gaps or overlaps
            add_data[~gaps & ~overlaps] = np.nansum(data, axis=0)
            # Find where there are overlaps and use fold to get the weighted average
            add_data[overlaps] = np.sum(data*fold, axis=0)/add_fold[overlaps]
        
        # CLEANUP
        # Enforce masked array if gaps exist
        if ~np.isfinite(add_data).all():
            add_data = np.ma.MaskedArray(data=add_data,
                                         mask=~np.isfinite(add_data),
                                         fill_value=fill_value)
        # Assign data & fold
        output.data = add_data
        output.fold = add_fold
        # Enforce fold rules
        output._enforce_fold_rules()

        return output
    
    def __iadd__(self, other, **options):
        """_summary_

        :param other: _description_
        :type other: _type_
        """        
        self = self.__add__(other, **options)

    def __eq__(self, other):
        if not np.array_equal(self.fold, other.fold):
            return False
        super().__eq__(self, other)

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

    def _enforce_fold_rules(self, value):
        """Enforce defining rules for the fold attribute

        Rule 2) fold data-type must match data data-type
        Rule 3) masked data are automatically 0 fold, but not vice versa
        Rule 4) If fold has masked values, fill them with 0's
        """
        self._fold_sanity_checks(value)
        # Enforce dtype match
        if self.data.dtype != value.dtype:
            value.dtype = value.dtype.astype(self.data.dtype)
        # Enforce masked data values = 0 fold
        if isinstance(self.data, np.ma.MaskedArray):
            value[self.data.mask] = 0
        # Enforce masked fold values -> 0 fold
        if isinstance(value, np.ma.MaskedArray):
            value.fill_value = 0
            value = value.filled()
        return value

    def _fold_sanity_checks(self, value):
        """Complement to _data_sanity_checks from obspy.core.trace

        Check fold rules:
        0) fold must be numpy.ndarray
        1) fold shape must match data shape

        :param value: _description_
        :type value: _type_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises TypeError: _description_
        """        
        if not isinstance(value, np.ndarray):
            raise TypeError('value must be a numpy.ndarray')   
        if self.data.shape != value.shape:
            raise ValueError('value shape must match FoldTrace.data shape')

    def _get_fold_view_trace(self):
        """Create a new FoldTrace object with data that houses
        a view of this FoldTrace's fold values and copied metadata

        :return:
         - **ftr** (*PULSE.data.foldtrace.FoldTrace*) - FoldTrace object containing 
            a view of the **fold** of the source FoldTrace object as its **data** 
            attribute and a deep copy of the source FoldTrace's **stats** attribute
        """        
        ftr = FoldTrace(data=self.fold, header=self.stats.copy())
        return ftr


    # PROPERTY METHODS AND ASSIGNMENTS #

    def get_ext_id(self):
        rstr = self.get_id()
        rstr += f'.{self.stats.model}.{self.stats.weight}'
        return rstr
    
    id = property(get_ext_id)

    
    def resample(self, sampling_rate, window='hann', no_filter=True, strict_length=False):
        # Resample fold first
        ftr = self._get_fold_view_trace()
        Trace.resample(
                ftr,
                sampling_rate,
                window=window,
                no_filter=no_filter,
                strict_length=strict_length)
        breakpoint()
        # Resample data next
        Trace.resample(
            sampling_rate,
            window=window,
            no_filter=no_filter,
            strict_length=strict_length)
        # Re-assign fold-trace to fold
        self.fold = ftr.data
        # Round fold to suppresss Gibbs phenomena
        self.fold = self.fold.round()
        self.stats.processing[-1] += f' PULSE: fold = resample + round'


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
    


                         

