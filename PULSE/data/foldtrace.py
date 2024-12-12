"""
:module: PULSE.data.foldtrace
:auth: Nathan T Stevens
:org: Pacific Northwest Seismic Network
:email: ntsteven (at) uw.edu
:license: AGPL-3
:purpose: This module contains the class definition for :class:`~.FoldTrace`, which extends
    the functionalities of the ObsPy :class:`~obspy.core.trace.Trace` class.
"""
import copy, warnings
import numpy as np
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.trace import Trace, Stats
from obspy.core.stream import Stream
from PULSE.util.header import MLStats

# FIXME: Track down processing metadata bleed - has to do with copy/deepcopy of MLStats

class FoldTrace(Trace):
    """A :class:`~obspy.core.trace.Trace` child class that adds:
     1) a **fold** attribute recording the relative importance of each sample in **data**
     2) an updated metadata object for it's **stats** attribute (:class:`~PULSE.data.header.MLStats`)
     3) a **dtype** attribute that helps maintain data-type consistency between **data** and **fold**

    :param data: data vector or Trace-like object, defaults to None
        None input results in an empty :class:`~numpy.ndarray`
    :type data: numpy.ndarray, obspy.core.trace.Trace, or NoneType, optional
    :param fold: fold vector, defaults to None
        None input results in a :meth:`~numpy.ones` vector matching the shape
    :type fold: numpy.ndarray or NoneType, optional
    :param header: non-default metadata inputs for the **stats** attribute, defaults to None
    :type header: dict, NoneType, :class:`~obspy.core.trace.Stats`, optional
        also see :class:`~PULSE.data.header.MLTrace`
    :param dtype: specified data-type for this FoldTrace's **data** and **fold** attributes, defaults to None
    :type dtype: :class:`~numpy.dtype` or NoneType
        see dtype heirarchy note

    :var data: :class:`~numpy.ndarray` -- vector containing discrete time series data
    :var fold: :class:`~numpy.ndarray` -- vector containing relative importance of values in **data**
    :var stats: :class:`~PULSE.data.header.MLStats` -- Attribute dictionary containing metadata for this FoldTrace
    :var dtype: :class:`~numpy.dtype` -- dtype of **data** and **fold**

    .. Note:: 
        The **fold** attribute reflects the relative importance (e.g., number of observations)
        a given sample in **data** results from and is used in the :meth:`~.FoldTrace.__add__` method
        for weighted average stacking. **fold** values adhere to the following rules
            1) all masked **data** values have a **fold** of 0, but not vice versa
            2) **fold** has the same dtype as **data**
            3) if **fold** becomes masked, masked values are automatically filled with 0 values
    
            
    .. Note:: 
        When initializing a :class:`~.FoldTrace` object, setting **dtype** supercedes
        the native data type of inputs for **data** and **fold**, with the following ranked
        importance:
            1) **dtype** = :class:`~numpy.dtype`
            2) **data.dtype**
            3) **fold.dtype**
            4) numpy.float64

        Such that, **FoldTrace.dtype** is set by the highest ranking non-NoneType input.

        Subsequent modifications of **dtype** are dictated by changes in **FoldTrace.data.dtype**
        to conform with dtype modifications enacted by inherited :class:`~obspy.core.trace.Trace` methods.

        
    """    
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

        # Compatability checks on header
        # If none input, make sure processing is not populated
        if header is None:
            # Ensure that processing is explicitly defined
            header = {'processing': []}
        # If dict or Stats, pass to MLStats
        elif isinstance(header, (dict, Stats)):
            # If processing is not supplied, explicitly define
            if 'processing' not in header.keys():
                header.update({'processing': []})
        # Otherwise raise TypeError
        else:
            raise TypeError('header must be type dict, ObsPy Stats, or NoneType')
        
        # Compatability checks on fold
        if fold is None:
            fold = np.ones(data.shape, dtype=self.dtype)
        elif isinstance(fold, np.ndarray):
            fold = fold.astype(self.dtype)
        else:
            raise TypeError('fold must be type NumPy ndarray or NoneType')
        

        # Initialize as empty Trace
        super().__init__(header={})
        
        # Upgrade stats class & set values
        self.stats = MLStats(header)
        # Set data as a view of input (needed for efficiency, but requires some additional care)
        self.data = data
        # Check fold against self.data and self.dtype
        self._fold_sanity_checks(fold)
        # Set fold
        self.fold = self._enforce_fold_masking_rules(fold)
            
    def _internal_add_processing_info(self, info):
        proc = self.stats.processing
        if len(proc) == self._max_processing_info-1:
            msg = ('List of procesing information in FoldTrace.stats.processing '
                   'reached maximum length of {} entries.')
            warnings.warn(msg.format(self._max_processing_info))
        if len(proc) < self._max_processing_info:
            proc.append(info)
    
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
    
    def _get_id_keys(self):
        """Get commonly used ID elements from this FoldTrace's **stats**

        .. rubric:: elementary keys  
            N - network
            S - station
            L - location (default is '--')
            C - channel
            M - model (ML model architecture name)
            W - weight (ML model pretrained weight name)

        .. rubric:: compound keys  
            id - N.S.L.C(.M.W)
            nslc - N.S.L.C (id for ObsPy Trace)
            sncl - S.N.C.L (ID order for Earthworm)
            site - N.S
            inst - N.S.L.C[:-1]
            mod - M.W
        """
        return self.stats.get_id_keys()

    id_keys = property(_get_id_keys)

    def verify(self):
        """Conduct sanity checks on this FoldTrace to make sure it's **data**
        and **fold** attributes have a specified byteorder, and identical
        dtype, and shape
        """        
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
        updating the raised error types and adding a check
        for array protocol type (numpy.dtype.kind) to allow only 
        (un)signed integer, float, and complex dtypes:
        unsigned integer -- dtype.kind == 'u'
        signed integer -- dtype.kind == 'i'
        float -- dtype.kind == 'f'
        complex -- dtype.kind == 'c'
        """
        if not isinstance(value, np.ndarray):
            raise TypeError('FoldTrace.data must be type np.ndarray')
        if value.ndim != 1:
            msg = f'NumPy array for FoldTrace.data has bad shape ({value.shape}). '
            msg += 'Only 1-d arrays are allowed.'
            raise ValueError(msg)
        if value.dtype.kind not in ['u','i','f','c']:
            msg = f'NumPy dtype "{value.dtype}" with array-protocol kind "{value.dtype.kind}" not supported.'
            msg += f'Supported protocol types: "u", "i", "f", "c". See :class:`~numpy.dtype'
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

    def validate_other(self, other):
        #TODO: Make a unit test for me!
        # Run sanity checks
        if not isinstance(other, FoldTrace):
            raise ValueError('other is not type FoldTrace')
        if self.dtype != other.dtype:
            raise ValueError('other has mismatched datatype')
        if self.get_id() != other.get_id():
            raise ValueError('other has mismatched ID')
        if self.stats.sampling_rate != other.stats.sampling_rate:
            raise ValueError('other has mismatched sampling_rate')
        if self.stats.calib != other.stats.calib:
            raise ValueError('other has mismatched calibration factor')

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
        try:
            self.validate_other(other)
        except ValueError as msg:
            raise ValueError(msg)
        
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
        i2 = lt.stats.utc2nearest_index(rt.stats.starttime)
        # Add one to conform to numpy slicing syntax
        i3 = lt.stats.utc2nearest_index(rt.stats.endtime) + 1
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
            intfold = np.ceil(self.fold.copy())
            unique, counts = np.unique(intfold, return_counts=True)
            for _u, _c in zip(unique, counts):
                rstr += f' [{int(_u)}] {_c}'
        return rstr

    ## NEW PUBLIC METHODS ##
    
    def astype(self, dtype=None):
        """Conduct an in-place change of this FoldTrace's **dtype**
        for both **data** and **fold**

        :param dtype: New data type, defaults to None
        :type dtype: numpy.dtype or NoneType, optional
        """        
        if dtype is None:
            pass
        else:
            try:
                dtype = np.dtype(dtype)
            except TypeError:
                raise TypeError(f'dtype "{dtype}" not understood.')
            if hasattr(self, 'data'):
                self.data = self.data.astype(dtype)
        return self
    
    def apply_to_gappy(self, method, **options):
        """Apply a specified :class:`~.FoldTrace` method to this FoldTrace even if it
        has masked **data** values. In the case of masked (gappy) data, the FoldTrace
        is:
         1) split into contiguous, non-masked elements (via :meth:`~.FoldTrace.split`)
         2) has the **method** applied to each element
         3) elements are recombined into a single FoldTrace (:meth:`~.FoldTrace.__add__`)

        :param method: _description_
        :type method: _type_
        """        
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


    def view(self, starttime=None, endtime=None):
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
        # if no specified starttime, use initial starttime
        if starttime is None:
            ii = None
        elif not isinstance(starttime, UTCDateTime):
            raise TypeError('starttime must be type obspy.UTCDateTime or NoneType')
        # if starttime is specified before the start of this FoldTrace,
        # start from starttime of this FoldTrace
        elif starttime <= self.stats.starttime:
            ii = None
        # otherwise, calculate the position of the first sample to view
        else:
            ii = self.stats.utc2nearest_index(starttime)
        # if no specified endtime, use initial endtime
        if endtime is None:
            ff = None
        elif not isinstance(endtime, UTCDateTime):
            raise TypeError('endtime must be type obspy.UTCDateTime or NoneType')
        # if endtime is specified after the end of this FoldTrace,
        # end at the endtime of this FoldTrace
        elif endtime >= self.stats.endtime:
            ff = None
        # otherwise, calculate the position of the last sample to view
        else:
            ff = self.stats.utc2nearest_index(endtime) + 1
        # ii = self.stats.utc2nearest_index(starttime, ref='starttime')
        # ff = self.stats.utc2nearest_index(endtime, ref='endtime') + 1
        header = self.stats.copy()
        if ii is not None:
            header.starttime += ii/self.stats.sampling_rate
        ftr = FoldTrace(header=header)
        ftr.data = self.data[ii:ff]
        ftr.fold = self.fold[ii:ff]

        return ftr
    
    def get_valid_fraction(self, starttime=None, endtime=None, threshold=0):
        if starttime is None:
            starttime = self.stats.starttime
        if endtime is None:
            endtime = self.stats.endtime
        target_npts = (endtime - starttime)*self.stats.sampling_rate + 1
        view =  self.view(starttime=starttime, endtime=endtime)
        valid_npts = len(view.data[view.fold > threshold])
        return valid_npts/target_npts
    
    vf = property(get_valid_fraction)

    ################################
    ## UPDATED, INHERITED METHODS ##
    ################################

    # PRIVATE SUPPORTING METHODS #

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
        # Get old and new stats alised for brevity
        old = trf.stats
        new = self.stats
        # Apply left padding if needed
        if old.starttime > new.starttime:
            trf._ltrim(new.starttime - 1./old_sampling_rate,
                       pad=True, fill_value=0,
                       nearest_sample=False)
        # Apply right padding if needed
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
    
    def _enforce_time_domain(self, old_stats, nearest_sample=False, **options):
        """Enforce the time domain [starttime, endtime]
        for this FoldTrace from a different :class:`~obspy.core.trace.Stats`-like
        object with matching **id** values.

        Primarily used as a sub-routine for :meth:`~.resample` to eliminate
        estimated values that fall outside the time bounds of **old_stats**

        :param old_stats: Stats object 
        :type old_stats: _type_
        :return: _description_
        :rtype: _type_
        """      
        if not isinstance(old_stats, Stats):
            raise TypeError('old_stats must be type obspy.core.trace.Stats')
        if self.stats.id != old_stats.id:
            raise ValueError('Stats "id" attributes do not match. Incompatable.')
        tsn = self.stats.starttime
        tso = old_stats.starttime
        ten = self.stats.endtime
        teo = old_stats.endtime
        if tsn < tso:
            self._ltrim(tso, nearest_sample=nearest_sample, **options)
        if ten > teo:
            self._rtrim(teo, nearest_sample=nearest_sample, **options)
        return self
    
    def _ltrim(self, starttime, pad=False, nearest_sample=True, fill_value=None, apply_fill=True):
        """Updated :meth:`~obspy.core.trace._ltrim` method that also trims/pads the
        left end of the **fold** attribute for this FoldTrace object. Padding values
        for **fold** are always set to 0.

        :param starttime: new starting time for this FoldTrace
        :type starttime: obspy.core.utcdatetime.UTCDateTime
        :param pad: should padding be enabled? Defaults to False
        :type pad: bool, optional
        :param nearest_sample: should trimming/padding go to the nearest sample?
            Defaults to True
        :type nearest_sample: bool, optional
        :param fill_value: what fill value should be applied to padding **data**
            samples, defaults to None
        :type fill_value: scalar, optional
        :param apply_fill: should the fill_value be applied (i.e., fill masked values)?
            Defaults to `True`. This default value emulates its parent class method's behavior
        :type apply_fill: bool, optional

        """        
        npts_old = self.stats.npts

        # Trim Data with masking for padding samples
        Trace._ltrim(self, starttime, pad=pad,
                     nearest_sample=nearest_sample,
                     fill_value=None)
                     
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

        # Bugfix for uniform fill_value application on gappy data
        if np.ma.is_masked(self.data):
            if fill_value is None:
                pass
            elif fill_value != self.data.fill_value:
                self.data.fill_value = fill_value
        
        # Add option for if fill should be applied
        if apply_fill:
            if isinstance(self.data, np.ma.MaskedArray):
                self.data = self.data.filled()

        self.verify()
        return self
    
    def _rtrim(self, endtime, pad=False, nearest_sample=True, fill_value=None, apply_fill=True):
        """Updated :meth:`~obspy.core.trace._rtrim` method that also trims/pads the
        right end of the **fold** attribute for this FoldTrace object. Padding values
        for **fold** are always set to 0.

        :param endtime: new ending time for this FoldTrace
        :type endtime: obspy.core.utcdatetime.UTCDateTime
        :param pad: should padding be enabled? Defaults to False
        :type pad: bool, optional
        :param nearest_sample: should trimming/padding go to the nearest sample?
            Defaults to True
        :type nearest_sample: bool, optional
        :param fill_value: what fill value should be applied to padding **data**
            samples, defaults to None
        :type fill_value: scalar, optional
        :param apply_fill: should the fill_value be applied (i.e., fill masked values)?
            Defaults to `True`. This default value emulates its parent class method's behavior
        :type apply_fill: bool, optional

        """  
        npts_old = self.stats.npts

        # Trim with masking for padding samples
        Trace._rtrim(self, endtime, pad=pad,
                     nearest_sample=nearest_sample,
                     fill_value=None)
        
        npts_new = self.stats.npts
        if npts_old > npts_new:
            self.fold = self.fold[:npts_new]
        elif npts_old < npts_new:
            new_fold = np.zeros(self.data.shape, dtype=self.dtype)
            new_fold[:npts_old] += self.fold
            self.fold = new_fold
        else:
            pass

        # Bugfix for uniform fill_value application on gappy data
        if np.ma.is_masked(self.data):
            if fill_value is None:
                pass
            elif fill_value != self.data.fill_value:
                self.data.fill_value = fill_value
        
        # Add option for if fill should be applied
        if apply_fill:
            if isinstance(self.data, np.ma.MaskedArray):
                self.data = self.data.filled()

        self.verify()
        return self
    
    # PUBLIC UPDATED METHODS #

    def trim(self, starttime=None, endtime=None, pad=False, nearest_sample=True, fill_value=None, apply_fill=True):
        """Trim/pad this FoldTrace to the specified starting and/or ending times
        with the option to pad and fill **data** values. Wraps updated methods:
        :meth:`~obspy.core.trace.Trace._ltrim` -> :meth:`~PULSE.data.foldtrace.FoldTrace._ltrim`
        :meth:`~obspy.core.trace.Trace._rtrim` -> :meth:`~PULSE.data.foldtrace.FoldTrace._rtrim`

        :param starttime: new starting time, defaults to None
            None uses current starttime of this FoldTrace
        :type starttime: obspy.core.utcdatetime.UTCDateTime, optional
        :param endtime: new ending time, defaults to None
        :type endtime: obspy.core.utcdatetime.UTCDateTime, optional
        :param pad: should padding be enabled? Defaults to False
        :type pad: bool, optional
        :param nearest_sample: should nearest sample inclusion be used? Defaults to True
            also see :meth:`~obspy.core.trace.Trace.trim` documentation
        :type nearest_sample: bool, optional
        :param fill_value:  fill value to assign to masked/padding values, defaults to None
        :type fill_value: scalar, optional
        :param apply_fill: should the fill_value be applied (i.e., fill masked values)?
            Defaults to `True`. This behavior matches behavoirs of its parent class method
        :type apply_fill: bool, optional
        """
        if isinstance(starttime, UTCDateTime):
            if isinstance(endtime, UTCDateTime):
                if starttime > endtime:
                    raise ValueError('starttime is larger than endtime')
        if starttime:
            self._ltrim(starttime, pad=pad, nearest_sample=nearest_sample,
                        fill_value=fill_value, apply_fill=apply_fill)
        if endtime:
            self._rtrim(endtime, pad=pad, nearest_sample=nearest_sample,
                        fill_value=fill_value, apply_fill=apply_fill)

        self.verify()
        return self
        


        # # Trim with masking for gaps/padding
        # super().trim(starttime=starttime,
        #              endtime=endtime,
        #              pad=pad,
        #              nearest_sample=nearest_sample,
        #              fill_value=None)
        # # Bugfix for uniform fill_value application on gappy data
        # if np.ma.is_masked(self.data):
        #     if fill_value is None:
        #         pass
        #     elif fill_value != self.data.fill_value:
        #         self.data.fill_value = fill_value
        
        # # Add option for if fill should be applied
        # if apply_fill:
        #     if isinstance(self.data, np.ma.MaskedArray):
        #         self.data = self.data.filled()

    
    def split(self, ascopy=True):
        """Split this FoldTrace into an :class:`~obspy.core.stream.Stream` object
        containing one or more non-masked :class:`~PULSE.data.foldtrace.FoldTrace`
        objects.

        :param ascopy: should FoldTraces in **st** be deep copies of the contents
            of this FoldTrace? Defaults to True.
                True  - FoldTrace objects in **st** are deepcopy'd segments of 
                        this FoldTrace
                False - FoldTrace objects in **st** have **data** and **fold** that
                        are views of **data** and **fold** of this FoldTrace. This
                        means that any modifications made to the **data** or **fold**
                        values. 
        :type ascopy: bool, optional
        :return:
            - **st** (*obspy.core.stream.Stream*) -- Stream containing one or more
                    :class:`~PULSE.data.foldtrace.FoldTrace` objects
        """        
        # If data are not masked, return a stream containing a view of self (NOTE: ObsPy returns a copy of self)
        views = []
        if not isinstance(self.data, np.ma.masked_array):
            views.append(self)
        # TODO: enforce transition of MaskedArray but not masked data to filled: in __setattr__?
        # if data are masked and 
        elif isinstance(self.data, np.ma.masked_array) and np.ma.is_masked(self.data):
            # Get contiguous "runs" of False entries in mask
            # Solution from: https://stackoverflow.com/questions/68514880/finding-contiguous-regions-in-a-1d-boolean-array
            # User UniCycle958 - Posted Dec. 3 2023
            um_runs = np.argwhere(np.diff(self.data.mask, prepend=True, append=True))
            um_runs = um_runs.reshape(len(um_runs)//2, 2)
            um_runs = [tuple(r) for r in um_runs]
            views = []
            for irun in um_runs:
                ts = self.stats.starttime + irun[0]*self.stats.delta
                te = self.stats.starttime + (irun[1] - 1)*self.stats.delta
                view = self.view(starttime=ts, endtime=te)
                views.append(view)
        st = Stream(views)
        if ascopy:
            st = copy.deepcopy(st)
        return st


    def taper(self, max_percentage, type='hann', max_length=None, side='both', taper_fold=True, **kwargs):
        """Taper the contents of **data** using :meth:`~obspy.core.trace.Trace.taper` with an option
        to also taper the **fold** to indicate a reduction in data importance/density at tapered
        samples.

        The taper is identically applied to **data** and **fold** to reflect that data

        :param max_percentage: Decimal percentage of taper at one end (ranging from 0. to 0.5)
        :type max_percentage: float
        :param type: type of taper function, defaults to 'hann'
            see documentation for :meth:`~obspy.core.trace.Trace.taper` for supported methods
        :type type: str, optional
        :param max_length: maximum taper length at one end in seconds, defaults to None
        :type max_length: float, optional
        :param side: Specify if both sides or one side ('left' / 'right') should be tapered, defaults to 'both'
        :type side: str, optional
        :param taper_fold: should the tapering also be applied to **fold**? Defaults to True
        :type taper_fold: bool, optional
        """   
        pcount = len(self.stats.processing)
        if taper_fold:
            ftf = FoldTrace(data=self.fold, header=self.stats.copy())
            Trace.taper(ftf, max_percentage, type=type, max_length=max_length, side=side, **kwargs)
            self.fold = ftf.data
        Trace.taper(self, max_percentage, type=type, max_length=max_length, side=side, **kwargs)
        if len(self.stats.processing) - pcount > 1:
            breakpoint()
        return self


    def align_starttime(self, starttime, sampling_rate, subsample_tolerance=0.01, **options):
        """Align the starttime of a component trace to the sampling
        index described by target values in **stats** using either a petty
        time shift for small subsample adjustments, or interpolation for
        larger subsample adjustments.
        
        Behaviors
        ---------
        Modifications are made in-place on the specified component.

        Modification behaviors are dictated by how large of a subsample
        adjustment is necessary to place the component's starttime into
        the discretized time sampling index specified by inputs **starttime**
        and **sampling_rate**

        Misalignment <= **subsample_tolerance** of a sample - 
            only starttime is updated, a "petty adjustment".

        Misailgnment > **subsample_tolerance** of a sample - uses 
           :meth:`~.FoldTrace.interpolate` referenced to the first
           aligned timestamp after the original starttime of
           the component being adjusted.

        Parameters
        ----------
        :param starttime: reference starttime for the new sampling index
        :type starttime: obspy.UTCDateTime
        :param sampling_rate: sampling rate for the new sampling index
        :type sampling_rate: float
        :param subsample_tolerance: subsample tolerance fraction determining
            if a "petty adjustment" is applied, defaults to 0.01
            Must be a value :math:`\in` [0., 0.5]
        :type subsample_tolerance: float, optional
        """
        # Argument compatability checks
        if isinstance(subsample_tolerance, int):
            subsample_tolerance = float(subsample_tolerance)
        if not isinstance(subsample_tolerance, float):
            raise TypeError(f'subsample_tolerance must be type float. Not type {type(subsample_tolerance)}')
        if not 0 <= subsample_tolerance <= 0.5:
            raise ValueError('subsample_tolerance must be in [0, 0.5]')
        if not isinstance(starttime, UTCDateTime):
            raise TypeError
        if not isinstance(sampling_rate, (int, float)):
            raise TypeError
        elif isinstance(sampling_rate, int):
            sampling_rate = float(sampling_rate)
        # If exact match, do nothing
        if starttime == self.stats.starttime:
            return
        
        # Calculate nearest starttime
        dt = starttime - self.stats.starttime
        npts = dt*sampling_rate        
        nearest_starttime = starttime - np.round(npts)/sampling_rate
        # Get sample fraction of misalignment
        dnpts = np.abs(nearest_starttime - self.stats.starttime)*sampling_rate
        # If misalignment is small, use "petty adjustment"
        if dnpts <= subsample_tolerance:
            self.stats.processing.append(f'PULSE 0.0.0: align_starttime(starttime={starttime}, sampling_rate={sampling_rate}, subsample_tolerance={subsample_tolerance})')
            self.stats.starttime = nearest_starttime
        # Otherwise, use interpolation starting from the nearest starttime within the current trace
        else:
            nearest_starttime = starttime - np.floor(npts)/sampling_rate
            self.interpolate(sampling_rate=self.stats.sampling_rate,
                             starttime=nearest_starttime,
                             **options)

    def interpolate(self, sampling_rate, method='weighted_average_slopes',
                    starttime=None, npts=None, time_shift=0.0,
                    *args, **kwargs):
        """Resample the **data** of this FoldTrace using the :meth:`~obspy.core.trace.Trace.interpolate` method
        and resample **fold** using linear interpolation via :meth:`~._interp_fold`

        :param sampling_rate: new sampling rate
        :type sampling_rate: float-like
        :param method: interpolation method, defaults to 'weighted_average_slopes'
            Supported methods: see :meth:`~obspy.core.trace.Trace.interpolate` for full descriptino
            - "lanczos" - (Sinc interpolation) - highest quality, but computationally costly
            - "weighted_average_slopes" - SAC standard
            - "slinear" - 1st order spline
            - "quadratic" - 2nd order spline
            - "cubic" - 3rd order spline
            - "linear" - linear interpolation (always used to interpolate **fold**)
            - "nearest" - nearest neighbor
            - "zero" - last encountered value
        :type method: str, optional
        :param starttime: Start time for the new interpolated FoldTrace, defaults to None
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param npts: new number of samples, defaults to None
        :type npts: int, optional
        :param time_shift: Shift the trace by a specified number of seconds, defaults to 0.0
            see :meth:`~obspy.core.trace.Trace.interpolate` for more information
        :type time_shift: float, optional
        :param *args: positional argument collector passed to internal interpolation method
        :param **kwargs: key-word argument collector passed to internal interpolation method
        """            
        # Grab initial stats info
        if np.ma.is_masked(self.data):
            raise Exception('masked data - try using FoldTrace.split to get contiguous FoldTraces')
        old_stats = self.stats.copy()
        Trace.interpolate(self,
                          sampling_rate,
                          starttime=starttime,
                          method=method,
                          npts=npts,
                          time_shift=time_shift,
                          *args,
                          **kwargs)
        # if enforce_time_domain:
        #     self._enforce_time_domain(old_stats)
        self._interp_fold(old_stats.starttime, old_stats.sampling_rate)
        return self

    def resample(self, sampling_rate, window='hann', no_filter=True, strict_length=False, enforce_time_domain=True):
        """Resample this FoldTrace using a Fourier method via :meth:`~obspy.core.trace.Trace.resample` for **data**
        and linear interpolation of **fold** via :meth:`~._interp_fold`. **stats** is updated with **data**.

        :param sampling_rate: new sampling rate or the resampled signal
        :type sampling_rate: float-like
        :param window: Name of the window applied to the signal in the Fourier domain, defaults to 'hann'
        :type window: str, optional
        :param no_filter: Should automatic filtering be skipped? Defaults to True
        :type no_filter: bool, optional
        :param strict_length: Should the FoldTrace be left unchanged if resampling changes this FoldTrace's endtime? Defaults to False
        :type strict_length: bool, optional
        :param enforce_time_domain: Should samples generated by resampling that 
            fall outside the original start and end times of this FoldTrace be
            discarded? Defaults to True.
        :type enforce_time_domain: bool, optional
        
        """        
        if np.ma.is_masked(self.data):
            raise Exception('masked data - try using FoldTrace.split to get contiguous FoldTraces')
        old_stats = self.stats.copy()
        Trace.resample(self,
                       sampling_rate,
                       window=window,
                       no_filter=no_filter,
                       strict_length=strict_length)

        self._interp_fold(old_stats.starttime, old_stats.sampling_rate)
        if enforce_time_domain:
            self._enforce_time_domain(old_stats)
        return self
    
    def decimate(self, factor, no_filter=False, strict_length=False):
        """Downsample this FoldTrace's **data** by an integer factor
        using :meth:`~obspy.core.trace.Trace.decimate` and match
        new sampling for **fold** using :meth:`~._interp_fold`

        :param factor: Factor by which the sampling rate is lowered by decimation.
        :type factor: int
        :param no_filter: Should automatic filtering be skipped? Defaults to True
        :type no_filter: bool, optional
        :param strict_length: Should the FoldTrace be left unchanged if resampling changes this FoldTrace's endtime? Defaults to False
        :type strict_length: bool, optional

        """
        # Handle non
        if isinstance(factor, (int, float)):
            if factor == int(factor):
                factor = int(factor)
            else:
                raise TypeError('decimation factor must be int-like')
        else:
            raise TypeError('decimation factor must be int-like')
        if factor < 1:
            raise ValueError('decimation factor must be a positive integer')
        
        if np.ma.is_masked(self.data):
            raise Exception('masked data - try using FoldTrace.split to get contiguous FoldTraces')
        old_stats = self.stats.copy()
        Trace.decimate(self, factor, no_filter=no_filter, strict_length=strict_length)
        self._interp_fold(old_stats.starttime, old_stats.sampling_rate)
        return self

    def max(self):
        # Add in check to ignore masked data
        if np.ma.is_masked(self.data):
            data = self.data.data[~self.data.mask]
        else:
            data = self.data
        value = max(abs(np.nanmax(data)), abs(np.nanmin(data)))
        return value
    


    def normalize(self, norm=None):
        """Normalize this FoldTrace's **data** values using
        normalization norm calculated from non-masked values
        of **data**. 
        
        Wraps :meth:`~obspy.core.trace.Trace.normalize`

        **fold** values are not modified by this method.

        :param norm: name of normalization type or numeric norm, defaults to 'max'
            Supported Values:
                'max' - maximum amplitude
                    accepted aliases include: 'minmax' and 'peak'
                'std' - standard deviation
                    accepted aliases include: 'standard'
                int/float - passed as the norm
        :type type: str, optional
        """        
        if norm in [None, 'max','minmax','peak']:
            norm = self.max()
        elif norm in ['std','standard','sigma']:
            norm = self.std()
        elif isinstance(norm, float):
            pass
        elif isinstance(norm, int):
            pass
        else:
            raise ValueError(f'type {type} not supported.')
        Trace.normalize(self, norm=norm)

        return self
        
    