import decorator
import numpy as np
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.trace import Trace, Stats
from PULSE.data.header import MLStats

class FoldTrace(Trace):

    def __init__(self, data=np.array([]), fold=None, header=None, dtype=None):
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
        self.stats = MLStats(header=header)
        # Populate fold
        if fold is None:
            self.fold = np.ones(len(self), dtype=self.data.dtype)
        elif isinstance(fold, np.ndarray):
            self.fold = fold
        # Enforce rules for fold
        self._enforce_fold_rules()

    #################
    # ADDED METHODS #
    #################

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
        :type starttime: None or UTCDateTime, optional
        :param endtime: endtime of view, defaults to None
        :type endtime: _type_, optional
        :return: _description_
        :rtype: _type_
        """        
        # Get starting and ending indices
        ii = self.stats.utc2nearest_index(starttime,ref='starttime')
        if ii < 0:
            ii = 0
        ff = self.stats.utc2nearest_index(endtime,ref='endtime')
        if ff > self.stats.npts:
            ff = self.stats.npts - 1
        # Create deep copy of header
        header = self.stats.copy()
        # Get starttime of view
        header.starttime = self.stats.starttime + ii*self.stats.sampling_rate
        # Get number of samples of view (updates endtime)
        header.npts = ff - ii + 1
        # Generate new trace with views and copied metadata
        ftr = FoldTrace(data=self.data[ii:ff],
                        fold=self.fold[ii:ff],
                        header=self.header)
        return ftr

    def view_trim(self, starttime=None, endtime=None, **options):
        """Extends the ObsPy :meth:`~obspy.core.trace.Trace.trim` method using the 
        PULSE :meth:`~PULSE.data.foldtrace.Foldtrace.get_view` method to get an initial
        view of the source FoldTrace's data and fold attributes, then copies all of this
        to a new FoldTrace

        :param starttime: specify the starttime, defaults to None
            None defaults to the starttime of the source FoldTrace
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime` or None, optional
        :param endtime: endtime to trim/pad data, defaults to None
            None defaults to the endtime of the source FoldTrace
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime` or None, optional
        :param pad: gives the option to add padded values, defaults to False
        :type pad: bool, optional
        :param fill_value: _description_, defaults to None
        :type fill_value: _type_, optional
        """        
        # Create a deep-copy of a subset view
        ftr = self.get_view(starttime=starttime, endtime=endtime).copy()
        # Apply trim to data and fold
        options.update({'starttime': starttime, 'endtime': endtime})
        ftr._apply_to_data_and_fold('trim', **options)
        return ftr
    
    def extend(self, other, **options):
        """Extend this FoldTrace with a Trace-like object 
        If **other** is an ObsPy trace, it is first converted into a FoldTrace and subsequently added.

        :param other: Trace-like object
        :type other: :class:`~obspy.core.trace.Trace`, or child-class object
        """
        if not isinstance(other, FoldTrace):
            other = FoldTrace(other)
        self.apply_inplace_method_to_masked('__add__', other, process_fold=True, **options)

    def apply_inplace_method_to_masked(self, method, *args, process_fold=True, **kwargs):
        """Conduct a dynamic call of a method from :class:`~obspy.core.trace.Trace`
        with internal handling of gappy (i.e., masked) **data** not supported by
        most ObsPy Trace class methods. 

        Provides the option to also process the **fold** vector with the same method.
        
        Updates to FoldTrace.stats.processing are only propagated from the application of the
        method to **data**. 

        NOTE: This method assumes that the method applies changes to this object IN-PLACE
        and does not capture or return method outputs.

        :param method: name of the method to use
        :type method: str
        :param *args: collector for addtional positional arguments of **method**
        :param process_fold: should the **fold** attribute also be processed? Defaults to True.
        :type process_fold: bool, optional
        :param **kwargs: collector for additional key-word arguments for **method**
        """        
        # If the data vector is masked
        if isinstance(self.data, np.ma.MaskedArray):
            # Presserve it's fill_value
            fill_value = self.data.fill_value
            # Split into multiple, self-contiguous foldtraces
            st = self.split()
            # Iterate corss foldtraces to apply method individually
            for _e, _ftr in enumerate(st):
                _ftr._apply_inplace_method(method, *args, process_fold=process_fold, **kwargs)
                # If first iteration, redefine this foldtrace as the first foldtrace
                if _e == 0:
                    self = _ftr
                # all subsequent iterations, use __add__ to extend the first foldtrace
                else:
                    self.__add__(_ftr, method=0, fill_value=fill_value)
        # Otherwise use the method as normal
        else:
            self._apply_inplace_method(method, *args, process_fold=process_fold, **kwargs)

    ###################################
    # POLYMORPHIC "DATA ONLY" METHODS #
    ###################################

    def filter(self, type, **options):
        """Applys the :meth:`~obspy.core.trace.Trace.filter` method to the 
        **data** and **stats** attributes of this FoldTrace, allowing for
        masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

        :param type: type of filter to apply
        :type type: str
        :param **options: key-word argument collector for the ObsPy method.
            see :meth:`~obspy.core.trace.Trace.filter`
        """
        self.apply_inplace_method_to_masked('filter', type, **options)
        
    def detrend(self, **options):
        """Applys the :meth:`~obspy.core.trace.Trace.detrend` method to the 
        **data** and **stats** attributes of this FoldTrace, allowing for
        masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

        :param **options: key-word argument collector for the ObsPy method.
            see :meth:`~obspy.core.trace.Trace.detrend`
        """
        self.apply_inplace_method_to_masked('detrend', process_fold=False, **options)
    
    def differentiate(self, **options):
        """Applys the :meth:`~obspy.core.trace.Trace.differentiate` method to the 
        **data** and **stats** attributes of this FoldTrace, allowing for
        masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

        :param **options: key-word argument collector for the ObsPy method.
            see :meth:`~obspy.core.trace.Trace.differentiate`
        """
        self.apply_inplace_method_to_masked('differentiate', process_fold=False, **options)
    
    def integrate(self, **options):
        """Applys the :meth:`~obspy.core.trace.Trace.integrate` method to the 
        **data** and **stats** attributes of this FoldTrace, allowing for
        masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

        :param **options: key-word argument collector for the ObsPy method.
            see :meth:`~obspy.core.trace.Trace.integrate`
        """
        self.apply_inplace_method_to_masked('integrate', process_fold=False, **options)

    def remove_response(self, **options):
        self.apply_inplace_method_to_masked('remove_response', process_fold=False, **options)

    def remove_sensitivity(self, **options):
        """Applys the :meth:`~obspy.core.trace.Trace.remove_sensitivity` method to the 
        **data** and **stats** attributes of this FoldTrace, allowing for
        masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

        :param **options: key-word argument collector for the ObsPy method.
            see :meth:`~obspy.core.trace.Trace.remove_sensitivity`
        """
        self.apply_inplace_method_to_masked('remove_sensitivity', process_fold=False, **options)

    def taper(self, max_percentage, **options):
        """Applys the :meth:`~obspy.core.trace.Trace.taper` method to the 
        **data** and **stats** attributes of this FoldTrace, allowing for
        masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

        :param **options: key-word argument collector for the ObsPy method.
            see :meth:`~obspy.core.trace.Trace.taper`
        """
        self.apply_inplace_method_to_masked('taper', max_percentage, process_fold=False, **options)

    def trigger(self, type, **options):
        """Applys the :meth:`~obspy.core.trace.Trace.trigger` method to the 
        **data** and **stats** attributes of this FoldTrace, allowing for
        masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`
        
        :param type: trigger type to use
        :type type: str
        :param **options: key-word argument collector for the ObsPy method.
            see :meth:`~obspy.core.trace.Trace.trigger`
        """
        self.apply_inplace_method_to_masked('trigger', type, process_fold=False, **options)

    #######################################
    # POLYMORPHIC "DATA AND FOLD" METHODS #
    #######################################

    def decimate(self, factor, **options):
        """Applys the :meth:`~obspy.core.trace.Trace.decimate` method to the 
        **data**, **fold**, and **stats** attributes of this FoldTrace, allowing for
        masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`
        :param factor: factor by which sampling rate is lowered by decimation
        :type factor: int
        :param **options: key-word argument collector for the ObsPy method.
            see :meth:`~obspy.core.trace.Trace.decimate`
        """
        self.apply_inplace_method_to_masked('decimate', factor, process_fold=True, **options)

    def interpolate(self, sampling_rate, **options):
        """Applys the :meth:`~obspy.core.trace.Trace.interpolate` method to the 
        **data**, **fold**, and **stats** attributes of this FoldTrace, allowing for
        masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`

        :param sampling_rate: new sampling rate in Hz 
        :type sampling_rate: float
        :param **options: key-word argument collector for the ObsPy method.
            see :meth:`~obspy.core.trace.Trace.interpolate`
        """
        self.apply_inplace_method_to_masked('interpolate', sampling_rate, process_fold=True, **options)

    def resample(self, sampling_rate, **options):
        """Applys the :meth:`~obspy.core.trace.Trace.resample` method to the 
        **data**, **fold**, and **stats** attributes of this FoldTrace, allowing for
        masked data via :meth:`~PULSE.data.foldtrace.FoldTrace.apply_inplace_method_to_masked`
        
        :param sampling_rate: new sampling rate in Hz 
        :type sampling_rate: float
        :param **options: key-word argument collector for the ObsPy method.
            see :meth:`~obspy.core.trace.Trace.resample`
        """
        self.apply_inplace_method_to_masked('resample', sampling_rate, process_fold=True, **options)
    
        

    #########################
    # ADDED PRIVATE METHODS #
    #########################

    def _apply_inplace_method(self, method, *args, process_fold=True, **kwargs):
        """Conduct a dynamic call of a method from :class:`~obspy.core.trace.Trace` with
        the option to also process the **fold** vector with the same method. Updates
        to FoldTrace.stats.processing are only propagated from the application of the
        method to **data**

        NOTE: This method assumes that the method applies changes to this object IN-PLACE
        and does not capture or return method outputs.

        :param method: name of the method to use
        :type method: str
        :param *args: collector for addtional positional arguments of **method**
        :param process_fold: should the **fold** attribute also be processed? Defaults to True.
        :type process_fold: bool, optional
        :param **kwargs: collector for additional key-word arguments for **method**
        """        
        # If processing fold too, create fold_view_trace
        if process_fold:
            ftr = self._get_fold_view_trace()
        # Process for data & metadata
        getattr(self, method, *args, **kwargs)
        if process_fold:
            getattr(ftr, method, *args, **kwargs)
    
    def _apply_to_data_only(self, method, *args, **kwargs):
        """Apply a specified :class:`~obspy.core.trace.Trace` class
        method to only the **data** attributes of this FoldTrace and
        include handling of "gappy" (i.e., masked) **data** using the
        :meth:`~PULSE
        and follow up with :meth:`~PULSE.data.foldtrace.FoldTrace._enforce_fold_rules`
        
         
        :param method: name of the :class:`~obspy.core.trace.Trace` method
        :type method: _type_
        """        
        self.apply_inplace_method_to_masked(method, *args, **kwargs)
        # Enforce fold rules
        self._enforce_fold_rules()

    ########################
    # PRIVATE FOLD METHODS #
    ########################
        
    def _enforce_fold_rules(self):
        """Enforce defining rules for the fold attribute

        Rule 0) fold must be a numpy.ndarray
        Rule 1) fold shape must match data shape
        Rule 2) fold data-type must match data data-type
        Rule 3) masked data are automatically 0 fold, but not vice versa
        """
        if not isinstance(self.fold, np.ndarray):
            raise TypeError('Fold must be a numpy.ndarray')   
        if self.data.shape != self.fold.shape:
            raise ValueError('Fold shape must match FoldTrace.data shape')
        # Enforce fold.dtype == data.dtype
        if self.data.dtype != self.fold.dtype:
            self.fold.dtype = self.fold.dtype.astype(self.data.dtype)
        # Enforce masked values = 0 fold
        if isinstance(self.data, np.ma.MaskedArray):
            self.fold[self.data.mask] = 0

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

        

                         
