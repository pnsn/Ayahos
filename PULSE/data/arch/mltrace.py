"""
:module: PULSE.data.mltrace
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This contains class definitions for :class:`~PULSE.data.mltrace.MLTrace` data objects
    and :class:`~PULSE.data.mltrace.MLTraceStats` metadata objects that extend functionalities 
    of their ObsPy :class:`~obspy.core.trace.Trace` and :class:`~obspy.core.trace.Stats` forebearers.
    
    Modifications are made to capture additional metadata associated with continuous waveform processing
    using machine learning models based on the SeisBench :class:`~seisbench.models.WaveformModel` class.

"""
import time ,os, copy, obspy, logging, warnings
import numpy as np
import pandas as pd
from decorator import decorator
from obspy import Stream, read, UTCDateTime
from obspy.core.trace import Trace, Stats
from obspy.core.util.misc import flat_not_masked_contiguous
from PULSE.util.header import MLStats
from PULSE.util.seisbench import pretrained_dict
from PULSE.util.stats import estimate_quantiles, estimate_moments
from PULSE.util.pyew import is_wave_msg

Logger = logging.getLogger(__name__)

###################################################################################
# Machine Learning Trace Class Definition #########################################
###################################################################################

class MLTrace(Trace):
    """Extends the :class:`~obspy.core.trace.Trace` class to handle additional data and metadata associated
    with machine learning based workflows such as models built on the SeisBench :class:`~seisbench.models.WaveformModel`
    template class.

    :param data: data vector to write to this MLTrace, defaults to empty numpy array
    :type data: numpy.ndarray
    :param fold: vector conveying the number of observations associated with each
            data point in `data` array. Defaults to None, resulting in a fold vector
            that equals numpy.ones(shape=data.shape)
    :type fold: NoneType or numpy.ndarray
    :param header: initial values for the MLStats object
    :type header: dict or NoneType, optional

    :var data: vector that holds waveform data
    :var fold: vector that holds information on the number of *valid* observations for each data sample.
        fold values of 0 are assigned for masked data values and can also be assigned for data points that should be ignored (i.e., blinded).
        fold values can be non-negative integers or float, with float values arising from resampling
            also see    
             - :meth:`~PULSE.data.mltrace.MLTrace.resample`
             - :meth:`~PULSE.data.mltrace.MLTrace.interpolate`
             - :meth:`~PULSE.data.mltrace.MLTrace.decimate`

    .. rubric:: Updated Trimming Methods
    The following methods are updated from their ObsPy Trace forebearers to also truncate the contents of **mltrace.fold**
        * :meth:`~PULSE.data.mltrace.MLTrace.view_trim` - creates a trimmed copy of the data using an intermediate view to reduce memory overhead
        * :meth:`~PULSE.data.mltrace.MLTrace._ltrim` - left trim data, fold, and starttime metadata
        * :meth:`~PULSE.data.mltrace.MLTrace._rtrim` - right trim data, fold, and starttime metadata
        * :meth:`~PULSE.data.mltrace.MLTrace.slice` - left trim data, fold, and starttime metadata
        * :meth:`~PULSE.data.mltrace.MLTrace.slide` - Generator yielding equal length sliding windows of MLTrace


    .. rubric:: Updated Resampling Methods
    The following methods are updated to resample the **mltrace.fold** attribute in addition to ObsPy Trace behaviors for the same methods.
    * :meth:`~PULSE.data.mltrace.MLTrace.resample`
    * :meth:`~PULSE.data.mltrace.MLTrace.interpolate`
    * :meth:`~PULSE.data.mltrace.MLTrace.decimate`

    .. rubric:: Expanded Merging Methods
    The following methods add new treatments for handling overlapping samples and fold when merging MLTrace-type objects
    * :meth:`~PULSE.data.mltrace.MLTrace.__add__`
    * :meth:`~PULSE.data.mltrace.MLTrace.merge`

    """
    _max_processing_info = 100

    def __init__(self, data=np.array([]), fold=None, header=None, dtype=np.float32):
        """
        Initialize an MLTrace object
        
        :param data: data vector to write to this MLTrace, defaults to empty numpy array
        :type data: numpy.ndarray
        :param fold: vector conveying the number of observations associated with each data 
            point in `data` array. Defaults to None, resulting in a fold vector that equals
            numpy.ones(shape=data.shape).
        :type fold: NoneType or numpy.ndarray
        :param header: initial values for the MLStats object, defaults to None
        :type header: dict or NoneType, optional
        :param dtype: numpy datatype to use for :var:`data` and :var:`fold`

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

        # TODO: Make real fix to this bandage
        # Ignore the warnings arising from MLTrace.__add__'s use of np.where
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

    ####################
    # HELPER METHODS ###
    def _apply_as_super(self, method, *args, **kwargs):
        """Apply an inherited method from :class:`~obspy.core.trace.Trace` 
        as written for Trace (rather than MLTrace)

        :param method: _description_
        :type method: _type_
        """        
        super_method = getattr(super(), method)
        super_method(*args, **kwargs)

    def _apply_to_masked(self, method, *args, **kwargs):
        """Apply a method to an MLTrace that may have masked data,
        splitting the MLTrace into contiguous, unmasked parts if
        masked values exist, apply the method to the parts, and then
        merge the mltraces back together.

        :param method: _description_
        :type method: _type_
        :param super: _description_, defaults to True
        :type super: bool, optional
        """        
        if np.ma.is_masked(self.data):
            fill_value = self.data.fill_value
            st = self.split()
            for _e, _mlt in enumerate(st):
                # if super:
                _mlt._apply_as_super(method, *args, **kwargs)
                # else:
                #     getattr(_mlt, method)(*args, **kwargs)
                if _e == 0:
                    self = _mlt
                else:
                    self.__add__(_mlt, method=0, fill_value=fill_value)
        else:
            # if super:
            self._apply_as_super(method, *args, **kwargs)
            # else:
            #     getattr(self, method)(*args, **kwargs)

    def _apply_to_data_and_fold(self, method, *args, **kwargs):
        """Apply a specified, inherited method to the data, fold, and stats 
        attributes of this MLTrace that for it's parent :class:`~obspy.core.trace.Trace`
        only modifies the data and stats attributes.

        :param method: _description_
        :type method: _type_
        """        
        # generate fold trace
        fold_trace = self.copy()
        fold_trace.data = fold_trace.fold
        # apply method to data
        self._apply_as_super(method, *args, **kwargs)
        # apply method to fold
        fold_trace._apply_as_super(method, *args, **kwargs)
        self.fold = fold_trace.data.copy()
        del fold_trace

    def _apply_to_masked_data_and_fold(self, method, *args, **kwargs):
        """Apply a specified, inherited method to the data, fold, and stats 
        attributes of this MLTrace that for it's parent :class:`~obspy.core.trace.Trace`
        only modifies the data and stats attributes.

        :param method: _description_
        :type method: _type_
        """        
        # generate fold trace
        fold_trace = self.copy()
        fold_trace.data = fold_trace.fold
        # apply method to data
        self._apply_to_masked(method, *args, **kwargs)
        # apply method to fold
        fold_trace._apply_to_masked(method, *args, **kwargs)
        self.fold = fold_trace.data.copy()
        del fold_trace

    # POLYMORPHIC METHODS ONLY AFFECTING DATA
            
    def filter(self, type, **options):
        self._apply_to_masked('filter', type, **options)
        
    def detrend(self, type='simple', **options):
        options.update({'type': type})
        self._apply_to_masked('detrend', **options)
    
    def differentiate(self, **options):
        self._apply_to_masked('differentiate', **options)
    
    def integrate(self, **options):
        self._apply_to_masked('integrate', **options)

    def remove_response(self, **options):
        self._apply_to_masked('remove_response', **options)

    def remove_sensitivity(self, **options):
        self._apply_to_masked('remove_sensitivity', **options)

    def taper(self, max_percentage, **options):
        self._apply_to_masked('taper', max_percentage, **options)

    def trigger(self, type, **options):
        self._apply_to_masked('trigger', type, **options)


    # POLYMORPHIC METHODS AFFECTING DATA AND FOLD SENSITVE TO MASKED DATA
        
    def decimate(self, factor, **options):
        self._apply_to_masked_data_and_fold('decimate', factor, **options)

    def interpolate(self, sampling_rate, **options):
        self._apply_to_masked_data_and_fold('interpolate', sampling_rate, **options)

    def resample(self, sampling_rate, **options):
        self._apply_to_masked_data_and_fold('resample', sampling_rate, **options)
    

    # POLYMORPHIC PRIVATE METHODS AFFECTING DATA AND FOLD
    # Also updates: trim, slice, slide, split
        
    def _ltrim(self, starttime, **options):
        self._apply_to_data_and_fold('_ltrim', starttime, **options)
    
    def _rtrim(self, endtime, **options):
        self._apply_to_data_and_fold('_rtrim', endtime, **options)

    # POLYMORPHIC SPECIAL METHODS
    def __add__(self, other, method=0, **options):
        if isinstance(other, Trace):
            if not isinstance(other, MLTrace):
                other = MLTrace(other)
        if method in [0, 1]:
            self._apply_to_data_and_fold('__add__', other, method=method, **options)
        elif method in [2,3]:
            # add0 = self.copy().__add__(method=0, fill_value=None)
            # add1 = self.copy().__add__(method=1, fill_value=None)
            raise NotImplementedError('Working on re-developing stacking methods')
            # apply method 0 to a copy
            # get gap 
            


    # def __add__(self, other, method=0, **options):
    #     options.update({'method': method})
    #     if isinstance(other, Trace):
    #         if not isinstance(other, MLTrace):
    #             other = MLTrace(other)
    #     if method in [0, 1]:
    #         self._apply_to_data_and_fold('__add__', other, method=method, **options)
    #     elif method == 2:



        # 0) blank out overlapping methods

        


#     ########################
#     # VIEW-BASED METHODS ###
#     ######################## 
#     def get_view(self, starttime=None, endtime=None):
#         """Fetch a view of the contents of this MLTrace's data and fold attributes

#         Notes:
#           - Producing views of numpy arrays is more computationally efficient and less memory-intensive compared
#             to the :meth:`~obspy.core.trace.Trace.trim` approach that modifies the data arrays contained in a trace-like object
#             but also expose the source arrays to modification if the contents of the views are modified.

#           - This approach is useful for repeat/sliding queries of a MLTrace object's data 

#           - None values for either reference time defaults to the relevant start/end time of this MLTrace

#         :param starttime: reference starttime, defaults to None
#         :type starttime: None or obspy.core.utcdatetime.UTCDateTime, optional
#         :param endtime: reference endtime, defaults to None
#         :type endtime: None or obspy.cre.utcdatetime.UTCDateTime, optional
#         :return: 
#             - **data_view** (*numpy.ndarray*) -- view of the data attribute
#             - **fold_view** (*numpy.ndarray*) -- view of the fold attribute

#         """        
#         # Get indicies of initial sample
#         if starttime is None:
#             ii = 0
#         elif isinstance(starttime, UTCDateTime):
#             ii = self.utcdatetime_to_nearest_index(starttime)
#             if ii < 0:
#                 ii = 0
#         else:
#             raise TypeError
        
#         # Get indices of final sample
#         if endtime is None:
#             ff = self.stats.npts
#         elif isinstance(endtime, UTCDateTime):
#             ff = self.utcdatetime_to_nearest_index(endtime)
#             if ff < 0:
#                 ff = 0
#         else:
#             raise TypeError
        
#         data_view = self.data[ii:ff]
#         fold_view = self.fold[ii:ff]
#         return data_view, fold_view

#     def view_copy(self, starttime=None, endtime=None, pad=False, fill_value=None):
#         """
#         Create a trimmed copy of this MLTrace where the subset data are fetched as a view and then copied
#         into a new :class:`~PULSE.data.mltrace.MLTrace` object.

#         This is beneficial for reducing memory overhead when sampling small
#         segments of long traces (i.e., window generation from MLTraceBuffer objects)
#         where the source trace data cannot be altered.

#         :param starttime: reference starttime, defaults to None
#             None uses the starttime for this MLTrace
#         :type starttime: NoneType or obspy.core.utcdatetime.UTCDateTime
#         :param endtime: reference endtime, defaults to None
#             None uses the endtime for this MLTrace
#         :type endtime: NoneType or obspy.core.utcdatetime.UTCDateTime
#         :param pad: Should the output trace be padded (i.e. filled gaps/padded edges)?, defaults to False
#         :type pad; bool
#         :param fill_value: fill value to use with padding, defaults to None
#         :type fill_value: NoneType, int, float
#             also see :meth:`~obspy.core.trace.Trace.trim`
#                      :meth:`~PULSE.data.mltrace.MLTrace.trim`
#         :return: 
#             - **mlt** (*PULSE.data.mltrace.MLTrace*) -- new MLTrace conataining copied data (and fold) information
        
#         """
#         data_view, fold_view = self.get_view(starttime=starttime, endtime=endtime)
#         ii = self.utcdatetime_to_nearest_index(starttime)
#         if starttime is None:
#             new_starttime = self.stats.starttime
#         else:
#             new_starttime = self.stats.starttime + ii*self.stats.delta
        
#         header = {_k: self.stats[_k] for _k in ['network',
#                                                 'station',
#                                                 'location',
#                                                 'channel',
#                                                 'model',
#                                                 'weight',
#                                                 'sampling_rate',
#                                                 'calib']}
#         mlt = MLTrace(data=data_view.copy(), fold=fold_view.copy(), header=copy.deepcopy(header))
#         # print(f'$$$$$$ {mlt.stats.processing}')
#         # Update copied view starttime if specified starttime 
#         # is after the source MLTrace starttime
#         if ii > 0:
#             mlt.stats.starttime = new_starttime
#         else:
#             mlt.stats.starttime = self.stats.starttime

#         if pad:
#             # breakpoint()
#             mlt = mlt.trim(starttime=new_starttime, endtime=endtime, pad=pad, fill_value=fill_value)
#             # # QnD treatment for run-away duplication of trim info...
#             # mlt.stats.processing.pop()
#             # NTS (18 JUN 2024) Debugged with :meth:`~PULSE.data.mltrace.MLTrace._internal_add_processing_info`
#         return mlt

#     def _internal_add_processing_info(self, info):
#         """
#         Disable processing update via decorator inherited from :class:`~obspy.core.trace.Trace`
#         """
#         pass

#     #####################################################################
#     # UPDATED SPECIAL METHODS ###########################################
#     #####################################################################
    
#     def __repr__(self, id_length=None):
#         """Provide a human readable string describing the contents of this :class:`~PULSE.data.mltrace.MLTrace` object

#         :param id_length: maximum ID length passed to the inherited :meth:`~obspy.core.trace.Trace.__str__` method, defaults to None
#         :type id_length: None or int, optional
#         :return rstr: representative string
#         :rtype: str
#         """        
#         rstr = super().__str__(id_length=id_length)
#         if self.stats.npts > 0:
#             rstr += f' | Fold:'
#             for _i in range(int(self.fold.max()) + 1):
#                 ff = sum(self.fold == _i)/self.stats.npts
#                 if ff > 0:
#                     rstr += f' [{_i}] {ff:.2f}'
#         return rstr
    
#     def __eq__(self, other):
#         """
#         Check the match between this MLTrace object and another
#         potential MLTrace object

#         This implementation checks the following fields in each
#         MLTrace's header:
#             via self.id: network, station, location, channel, model, weight
#             via self.stats: starttime, sampling_rate, npts, calib
#         And assesses matches between self.data and self.fold

#         :param other: other object to check agains this MLTrace
#         :type other: any, seeking PULSE.data.mltrace.MLTrace
#         :return: are the objects equivalent?
#         :rtype: bool

#         """
#         if not isinstance(other, MLTrace):
#             return False

#         if self.id != other.id:
#             return False
#         if self.stats.starttime != other.stats.starttime:
#             return False
#         if self.stats.npts != other.stats.npts:
#             return False
#         if self.stats.sampling_rate != other.stats.sampling_rate:
#             return False
#         if self.stats.calib != other.stats.calib:
#             return False
#         if not np.array_equal(self.data, other.data):
#             return False
#         if not np.array_equal(self.fold, other.fold):
#             return False
#         return True

#     # @_add_processing_info
#     def __add__(self, trace, method=1, fill_value=None, sanity_checks=True, reset_processing=True, dtype=np.float32):
#         """
#         Join a (compatable) Trace-type object to this MLTrace-type object using a specified method

#         method 0 - conforms to obspy.core.trace.Trace.__add__(method=0) behavior
#                     where overlapping samples are rejected from both traces and
#                     gaps are filled according to behaviors tied to fill_value
#                     Aliases: method = 'dis', 'discard'
#         method 1 - conforms to obspy.core.trace.Trace.__add__(method=1, interpolation_samples=-1)
#                     behviors where overlapping samples are replaced by a linear interpolation
#                     between overlap-bounding values
#                     Aliases: method = 'int', 'interpolate'
#         method 2 - stacking using the 'max' rule wherein the maximum value at a given
#                     overlapping sample is assigned to that sample point
#         method 3 - stacking using the 'avg' rule wherein the value of an overlapping point
#                     is calculated as the fold-weighted mean of input values

#         TODO: add a method that does a quadriture based interpolation approach (discussion with JRHartog)

#         In all cases, overlapping samples resulting in a new sample are assigned a fold equal
#         to the sum of the contributing samples' fold, and eliminated samples are assigned
#         a fold of 0.

#         :param trace: Trace-like object to add to this MLTrace
#         :type trace: obspy.core.trace.Trace or child class thereof
#         :param method: method to use for merging traces see above, defaults to 1
#         :type method: str or int, optional
#         :param fill_value: value to fill gaps with, defaults to None
#                         also see :meth: `~obspy.core.trace.Trace.__add__`
#                                  :meth: `~obspy.core.trace.Trace.trim`
#                                  :meth: `~PULSE.data.mltrace.MLTrace.view_copy`
#         :type fill_value: int, float, or NoneType
#         :param sanity_checks: run checks on metadata before attempting to merge data?, defaults to True
#         :type sanity_checks: bool, optional
#         :param dtype: data type to use as the reference data type, defaults to numpy.float32
#         :type: type, optional

#         TODO: Split different methods into private sub-methods to boost readability?

#         """
#         if dtype is None:
#             # Ensure fold matches data dtype
#             if self.fold.dtype != self.data.dtype:
#                 self.fold = self.fold.astype(self.data.dtype)
#             if trace.fold.dtype != trace.data.dtype:
#                 trace.fold = trace.fold.astype(trace.data.dtype)
#         else:
#             original_dtypes={'self': self.data.dtype, 'other': trace.data.dtype}
#             self.data = self.data.astype(dtype)
#             self.fold = self.fold.astype(dtype)
#             trace.data = trace.data.astype(dtype)
#             trace.fold = trace.fold.astype(dtype)

#         if sanity_checks:
#             if isinstance(trace, Trace):
#                 if not isinstance(trace, MLTrace):
#                     trace = MLTrace(data=trace.data, header=trace.stats)
#                 else:
#                     pass
#             # NOTE: If model/weight are specified in self.stats, 
#             # the update above will raise the error below
#             if self.id != trace.id:
#                 raise TypeError
#             if self.stats.sampling_rate != trace.stats.sampling_rate:
#                 raise TypeError
#             if self.stats.calib != trace.stats.calib:
#                 raise TypeError
#             if self.data.dtype != trace.data.dtype:
#                 breakpoint()
#                 raise TypeError

#         # Get data and fold vectors
#         sdata = self.data
#         sfold = self.fold
#         tdata = trace.data
#         tfold = trace.fold
#         # Apply NaN masks to data for subsequent operations
#         if isinstance(sdata, np.ma.masked_array):
#             if np.ma.is_masked(sdata):
#                 sdata.fill_value = np.nan
#                 sfold[sdata.mask] = 0
#                 sdata = sdata.filled()
#             else:
#                 sdata = sdata.data
#         if isinstance(tdata, np.ma.masked_array):
#             if np.ma.is_masked(tdata):
#                 tdata.fill_value = np.nan
#                 tfold[tdata.mask] = 0
#                 tdata = tdata.filled()
#             else:
#                 tdata = tdata.data
        
#         # Get relative indexing
#         idx = self._relative_indexing(trace)

#         # Create temporary data array
#         tmp_data_array = np.full(shape=(2, max(idx)),
#                                  fill_value=np.nan,
#                                  dtype=self.data.dtype)
#         # Create temporary fold array for 
#         tmp_fold_array = np.zeros(shape=(2, max(idx)),
#                                  dtype = self.fold.dtype)
#         # Place self data into joined indexing positions (data being appended to)
#         tmp_data_array[0, idx[0]:idx[1]] = self.data
#         tmp_fold_array[0, idx[0]:idx[1]] = self.fold
#         # Place trace data into joined indexing positions (data to be appended)
#         tmp_data_array[1, idx[2]:idx[3]] = trace.data
#         tmp_fold_array[1, idx[2]:idx[3]] = trace.fold

#         # Get positions where gaps and overlaps exist
#         gap_mask = ~np.isfinite(tmp_data_array).any(axis=0)
#         overlap_mask = np.isfinite(tmp_data_array).all(axis=0)
#         # Discard Method (effectively obspy.core.trace.Trace.__add__(method=0))
#         if method in ['discard', 'dis', 0]:
#             tmp_data, tmp_fold = self._drop_add(
#                 gap_mask,
#                 overlap_mask,
#                 tmp_data_array,
#                 tmp_fold_array
#             )
#         # (1) Simple Interpolation method (effectively the same as obspy.core.trace.Trace.__add__(method=1))
#         elif method in ['int', 'interpolate', 1]:
#             tmp_data, tmp_fold = self._interp_add(
#                 gap_mask,
#                 overlap_mask,
#                 idx,
#                 tmp_data_array,
#                 tmp_fold_array
#             )
#         # (2) Use maximum value of overlapping elements(max stacking)
#         elif method in ['max', 'maximum', 2]:
#             tmp_data, tmp_fold = self._max_add(gap_mask, tmp_data_array, tmp_fold_array)
#         # (3) Use fold-weighted average of overlapping elements (average stacking)
#         elif method in ['avg', 'average', 'mean', 3]:
#             tmp_data, tmp_fold = self._avg_add(gap_mask, tmp_data_array, tmp_fold_array)
#         # If not all data are finite (i.e., gaps exist)
#         if not np.isfinite(tmp_data).all():
#             # Convert tmp_data into a masked array
#             mask = ~np.isfinite(tmp_data)
#             if fill_value is None:
#                 tmp_data = np.ma.masked_array(data=tmp_data,
#                                             mask=mask,
#                                             fill_value=fill_value)
#             else:
#                 tmp_data[mask] = fill_value
#             tmp_fold[mask] = 0

#         # CLEANUP SECTION
#         # Update starttime
#         new_t0 = self.stats.starttime + idx[0]/self.stats.sampling_rate
#         # Overwrite starttime
#         self.stats.starttime = new_t0
#         # Overwrite data
#         self.data = tmp_data
#         # Overwrite fold
#         if tmp_fold.dtype != self.data.dtype:
#             tmp_fold = tmp_fold.astype(self.data.dtype)
#         self.fold = tmp_fold
#         self.enforce_zero_mask()
#         return self
    
#     def _relative_indexing(self, other):
#         """
#         Helper method for __add__() - calculates the integer index positions
#         of the first and last samples of self and other on a uniformly
#         sampled time index vector

#         :param other: Trace-like object being added
#         :type other: obspy.core.trace.Trace-like
#         :return index: list with 4 elements
#                         index[0] = relative position of self.data[0]
#                         index[1] = relative position of self.data[1]
#                         index[2] = relative position of other.data[0]
#                         index[3] = relative position of other.data[1]
#         :rtype index: list of int

#         """
#         if not isinstance(other, Trace):
#             raise TypeError
#         if self.stats.sampling_rate != other.stats.sampling_rate:
#             raise AttributeError('sampling_rate mismatches between this trace and "other"')
#         self_t0 = self.stats.starttime
#         self_t1 = self.stats.endtime
#         other_t0 = other.stats.starttime
#         other_t1 = other.stats.endtime
#         sr = self.stats.sampling_rate
#         if self_t0 <= other_t0:
#             self_n0 = 0
#             self_n1 = self.stats.npts
#             other_n0 = int((other_t0 - self_t0)*sr) + 1
#             other_n1 = other_n0 + other.stats.npts
#         else:
#             other_n0 = 0
#             other_n1 = other.stats.npts
#             self_n0 = int((self_t0 - other_t0)*sr) + 1
#             self_n1 = self_n0 + self.stats.npts
        
#         index = [self_n0, self_n1, other_n0, other_n1]
#         return index


#     def _drop_add(self, gap_mask, overlap_mask, tmp_data_array, tmp_fold_array):
#         """Subroutine for :meth:`~PULSE.data.mltrace.MLTrace.__add__` that replicates the behaviors of
#         :meth:`~obspy.core.trace.Trace.__add__` for method = 0

#         :param gap_mask: boolean vector for where gaps exists between the self and other data arrays
#         :type gap_mask: numpy.ndarray
#         :param overlap_mask: boolean array for where overlaps exist between the self and other data arrays
#         :type overlap_mask: numpy.ndarray
#         :param tmp_data_array: temporary data array 
#         :type tmp_data_array: numpy.ndarray
#         :param tmp_fold_array: temporary fold array
#         :type tmp_fold_array: numpy.ndarray
#         :returns:
#          - **tmp_data** (*numpy.ndarray*) -- merged data vector
#          - **tmp_fold** (*numpy.ndarray*) -- merged fold vector
#         """
#         # Take the "sum" (NaN -> 0 + value) at all non gap/overlap spots
#         tmp_data = np.where(gap_mask | overlap_mask,
#                             np.nan, np.nansum(tmp_data_array, axis=0))
#         # 0-value fold in gaps and overlaps (that are now gaps)
#         tmp_fold = np.where(gap_mask | overlap_mask,
#                             0, np.nansum(tmp_fold_array, axis=0))
#         return tmp_data, tmp_fold
    
#     def _interp_add(self, gap_mask, overlap_mask, idx, tmp_data_array, tmp_fold_array):
#         """Subroutine for :meth:`~PULSE.data.mltrace.MLTrace.__add__` that replicates the behaviors of
#         :meth:`~obspy.core.trace.Trace.__add__` for method = 1

#         :param gap_mask: boolean vector for where gaps exists between the self and other data arrays
#         :type gap_mask: numpy.ndarray
#         :param overlap_mask: boolean array for where overlaps exist between the self and other data arrays
#         :type overlap_mask: numpy.ndarray
#         :param idx: indices of terminal samples of the self and other data/fold vectors
#         :type idx: list of int
#             generated by :meth:`~PULSE.data.mltrace.MLTrace._relative_indexing`
#         :param tmp_data_array: temporary data array 
#         :type tmp_data_array: numpy.ndarray
#         :param tmp_fold_array: temporary fold array
#         :type tmp_fold_array: numpy.ndarray
#         :returns:
#          - **tmp_data** (*numpy.ndarray*) -- merged data vector
#          - **tmp_fold** (*numpy.ndarray*) -- merged fold vector
#         """
#         # If any overlaps
#         if overlap_mask.any():
#             # self leading trace
#             if idx[0] < idx[2] < idx[1] < idx[3]:
#                 ls = tmp_data_array[0, idx[2]]
#                 rs = tmp_data_array[1, idx[1]]
#                 ds = idx[1] - idx[2]
#                 tmp_data_array[:,idx[2]:idx[1]] = np.linspace(ls, rs, ds)

#             # trace leading self
#             elif idx[2] < idx[0] < idx[3] < idx[1]:
#                 ls = tmp_data_array[1, idx[1]]
#                 rs = tmp_data_array[0, idx[3]]
#                 ds = idx[3] - idx[0]
#                 tmp_data_array[:, idx[0]:idx[3]] = np.linspace(ls, rs, ds)

#             # trace in self - reset other contributions to fill values
#             else:
#                 tmp_data_array[1, :] = np.nan
#                 tmp_fold_array[1, :] = 0

#         # For gaps, pad as np.nan, otherwise take nanmax for data
#         tmp_data = np.where(~gap_mask, np.nanmax(tmp_data_array, axis=0), np.nan)
#         # breakpoint()
#         # ...and nansum for fold
#         tmp_fold = np.where(~gap_mask,np.nansum(tmp_fold_array, axis=0), 0)
#         # For gaps or contiguous, use nansum to get 
#         return tmp_data, tmp_fold
    
#     def _max_add(self, gap_mask, tmp_data_array, tmp_fold_array):
#         """Subroutine for :meth:`~PULSE.data.mltrace.MLTrace.__add__` that returns the maximum value of overlapping values for two data vectors being added.

#         :param gap_mask: boolean vector for where gaps exists between the self and other data arrays
#         :type gap_mask: numpy.ndarray
#         :param overlap_mask: boolean array for where overlaps exist between the self and other data arrays
#         :type overlap_mask: numpy.ndarray
#         :param idx: indices of terminal samples of the self and other data/fold vectors
#         :type idx: list of int
#             generated by :meth:`~PULSE.data.mltrace.MLTrace._relative_indexing`
#         :param tmp_data_array: temporary data array 
#         :type tmp_data_array: numpy.ndarray
#         :param tmp_fold_array: temporary fold array
#         :type tmp_fold_array: numpy.ndarray
#         :returns:
#          - **tmp_data** (*numpy.ndarray*) -- merged data vector
#          - **tmp_fold** (*numpy.ndarray*) -- merged fold vector
#         """
#         # Where there are gaps, fill with nan, otherwise, take nanmax
#         tmp_data = np.where(gap_mask, np.nan, np.nanmax(tmp_data_array, axis=0));
#         # ..for fold, fill gaps with 0, otherwise take nansum
#         tmp_fold = np.where(gap_mask, 0, np.nansum(tmp_fold_array, axis=0))
#         return tmp_data, tmp_fold
            
#     def _avg_add(self ,gap_mask, tmp_data_array, tmp_fold_array):
#         """Subroutine for :meth:`~PULSE.data.mltrace.MLTrace.__add__` that calculates the fold-weighted average of overlapping data for two trace data being added

#         :param gap_mask: boolean vector for where gaps exists between the self and other data arrays
#         :type gap_mask: numpy.ndarray
#         :param overlap_mask: boolean array for where overlaps exist between the self and other data arrays
#         :type overlap_mask: numpy.ndarray
#         :param idx: indices of terminal samples of the self and other data/fold vectors
#         :type idx: list of int
#             generated by :meth:`~PULSE.data.mltrace.MLTrace._relative_indexing`
#         :param tmp_data_array: temporary data array 
#         :type tmp_data_array: numpy.ndarray
#         :param tmp_fold_array: temporary fold array
#         :type tmp_fold_array: numpy.ndarray
#         :returns:
#          - **tmp_data** (*numpy.ndarray*) -- merged data vector
#          - **tmp_fold** (*numpy.ndarray*) -- merged fold vector
#         """
#         # Get new sum array
#         tmp_fold = np.where(gap_mask, 0, np.nansum(tmp_fold_array, axis=0))
#         # Where there there are some data, get weighted average, otherwise set to np.nan
#         # Fix 18 JUN 2024: Debug for "RuntimeWarning: invalid value encountered in divide..."
#         tmp_idx = tmp_fold > 0
#         tmp_data = np.full(shape=tmp_fold.shape, fill_value=np.nan)
#         try:
#             tmp_data[tmp_idx] = np.nansum(tmp_data_array[:,tmp_idx]*tmp_fold_array[:,tmp_idx], axis=0)/tmp_fold[tmp_idx]
#         except:
#             breakpoint()
#         # tmp_data = np.where(tmp_fold > 0, np.nansum(tmp_data_array*tmp_fold_array, axis=0)/tmp_fold, np.nan)
        
#     #############################
#     # TIME<->INDEX METHODS ######
#     #############################
#     ## TODO: Perhaps move these to a utility methods section (also crops up in PULSE.data.window.Window.sync_to_target_window)

#     def utcdatetime_to_nearest_index(self, utcdatetime):
#         """get the index of the nearest sample in `data` to a provided :class:`~obspy.core.utcdatetime.UTCDateTime` reference time

#         index = (utcdatetime - starttime)*sampling_rate

#         :param utcdatetime: reference datetime
#         :type utcdatetime: obspy.core.utcdatetime.UTCDateTime
#         :return: index number
#         :rtype: int
        
#         """        
#         return round((utcdatetime - self.stats.starttime)*self.stats.sampling_rate)


#     def is_utcdatetime_in_sampling(self, utcdatetime):
#         """check if a given utcdatetime timestamp aligns with the sampling mesh of this MLTrace

#         :param utcdatetime: reference datetime
#         :type utcdatetime: obspy.core.utcdatetime.UTCDateTime
#         :return: truth of "is this `utcdatetime` in the sampling mesh?"
#         :rtype: bool

#         """        
#         npts = (utcdatetime - self.stats.starttime)*self.stats.sampling_rate
#         return npts == int(npts)
    
#     def get_fvalid_subset(self, starttime=None, endtime=None, threshold=1):
#         """Get the fraction of valid (non-masked & fold >= threshold) data contained in this trace (or a subset view thereof)

#         :param starttime: optional alternative starttime for creating a view, defaults to None
#         :type starttime: obspy.core.utcdatetime.UTCDateTime, optional
#         :param endtime: optional alternative endtime for creating a view, defaults to None
#         :type endtime: obspy.core.utcdatetime.UTCDateTime, optional
#             also see :meth: `~PULSE.data.mltrace.MLTrace.get_subset_view`
#         :param threshold: minimum fold value to consider a datapoint "valid", defaults to 1
#         :type threshold: int, optional
#         :return fvalid: fraction of data that are valid
#         :rtype: float

#         """        
#         _, fold_view = self.get_subset_view(starttime=starttime, endtime=endtime)
#         num = sum(fold_view >= threshold)
#         if starttime is None:
#             ii = 0
#         else:
#             ii = self.utcdatetime_to_nearest_index(starttime)
        
#         if endtime is None:
#             ff = self.stats.npts
#         else:
#             ff = self.utcdatetime_to_nearest_index(endtime)
        
#         den = ff - ii
#         if den > 0:
#             fvalid = num/den
#         else:
#             fvalid = 0
#         return fvalid

#     ###############################################################################
#     # FOLD HANDLING METHODS #######################################################
#     ###############################################################################        
#     def apply_to_data_and_fold(self, method, *args, **kwargs):
#         """Apply a :class:`~obspy.core.trace.Trace` methods that modify

#         :param method: _description_
#         :type method: _type_
#         """        
#         # save a copy of the data
#         original_data = self.data.copy()
#         original_stats = self.stats.copy()
#         # assign fold to data
#         self.data = self.fold
#         # apply method to "data"
#         getattr(self, method)(*args, **kwargs)
#         # reassign fold to fold
#         self.fold = self.data.copy()
#         # reassign data to data
#         self.data = original_data

#     def get_fold_trace(self):
#         """
#         Return an :class:`~obspy.core.trace.Trace` object with data=fold vector for this MLTrace, replacing the componet character with an 'f' and discarding additional properties in the trace stats

#         i.e., the mltrace.stats.model and mltrace.stats.weight are discarded

#         :return ft: fold trace
#         :rtype ft: obspy.core.trace.Trace

#         """
#         header = Stats()
#         for _k in header.defaults.keys():
#             header.update({_k: copy.deepcopy(self.stats[_k])})
#         ft = Trace(data=self.fold, header=header)
#         ft.stats.channel += 'f'        
#         return ft

#     # @_add_processing_info
#     def apply_blinding(self, blinding=(500,500)):
#         """
#         Apply blinding to this MLTrace, which is defined by setting `blinding` samples on either end of the fold array to 0

#         :param blinding: 2-tuple: positive int-like number of samples to
#                                 blind on the left (blinding[0]) and 
#                                 right (blinding[1]) ends of fold
#                          int: positive number of samples to blind
#                                 on either end of fold
#         :type blinding: [2-tuple] of int values, or int

#         """
#         if isinstance(blinding, (list, tuple)):
#             if len(blinding) != 2:
#                 raise SyntaxError
#             elif any(int(_b) != _b for _b in blinding):
#                 raise ValueError
#             elif any(_b < 0 for _b in blinding):
#                 raise ValueError
#             elif any(_b > self.stats.npts for _b in blinding):
#                 raise ValueError
            
#             if blinding[0] > 0:
#                 self.fold[:blinding[0]] = 0
#             if blinding[1] > 0:
#                 self.fold[-blinding[1]:] = 0

#         elif isinstance(blinding, (int, float)):
#             if int(blinding) != blinding:
#                 raise ValueError
#             elif blinding < 0:
#                 raise ValueError
#             elif blinding > self.stats.npts//2:
#                 raise ValueError
#             else:
#                 self.fold[:blinding] = 0
#                 self.fold[-blinding:] = 0
#         return self
    
#     # @_add_processing_info
#     # TODO: Should this be migrated to a WindowStream method?
#     def to_zero(self, method='both'):
#         """
#         Convert this MLTrace 0-valued vector for self.data and/or self.fold with the same shape and dtype.

#         This is a key subroutine for applying channel fill rules in :meth:`~PULSE.data.windowstream.WindowStream.apply_fill_rule`

#         :param method: method for zeroing out trace
#             Supported 'both' - convert mlt.fold and mlt.data to 0-vectors
#                         Generally used for 0-fill traces passed to ML prediction
#                       'data' - convert mlt.data only to a 0-vector
#                         Rarely used, here for completeness
#                       'fold' - convert mlt.fold only to a 0-vector
#                         Generally used for cloned traces passed to ML prediction

#         """
#         if method not in ['both','data','fold']:
#             raise ValueError(f'method {method} not supported. Supported: "both", "data", "fold"')

#         shape = self.data.shape
#         dtype = self.data.dtype
#         v0 = np.zeros(shape=shape, dtype=dtype)
#         if method in ['both','data']:
#             self.data = v0
#         if method in ['both','fold']:
#             self.fold = v0
#         return self

#     def enforce_zero_mask(self):
#         """
#         Enforce equivalence of masking in data to 0-values in fold but not the reciprocal
#         used as the standard syntax for MLTraces. Enforced in place

#         In short, 
#             all masked values are 0-fold
#             not all 0-fold values are masked

#         ..rubric: Toy Example for a :class: `~PULSE.data.mltrace.MLTrace` object 

#         data ->   1, 3, 4, 5, 2, 6, 7      --, 3, 4,--, 2, 6, 7
#         mask ->   F, F, F, T, F, F, F  ==>  F, F, F, T, F, F, F
#         fold ->   0, 1, 1, 1, 1, 1, 1       0, 1, 1, 0, 1, 1, 1
#         notice:   *        *                *        *
#                                         no change   change

#         """
#         # Update any places with masked values to 0 fold
#         if isinstance(self.data, np.ma.MaskedArray):
#             if np.ma.is_masked(self.data):
#                 self.fold[self.data.mask] = 0
#         return self
    
 
#     ###############################################################################
#     # UPDATED TRIMMING METHODS ####################################################
#     ###############################################################################

#     def _ltrim(self, starttime, pad=False, nearest_sample=True, fill_value=None):
#         """
#         Run :meth:`~obspy.core.trace.Trace._ltrim` on this MLTrace object's data and fold arrays.
        
#         Padding values in fold entries are always 0 to indicate no valid observations are present, regardless of the fill_value selected

#         :param starttime: starttime for left trim
#         :type starttime: obspy.core.utcdatetime.UTCDateTime
#         :param pad: should extra samples be padded?, defaults to False
#         :type pad: bool, optional
#             also see :meth:`~obspy.core.trace.Trace._ltrim`
#         :param nearest_sample: passed to nearest_sample in inherited Trace._ltrim
#         :type nearest_sample: bool, optional
#             also see :meth:`~obspy.core.trace.Trace._ltrim`
#         :param fill_value: fill value to apply to pad entries in data
#         :type fill_value: NoneType, int, float
#             also see :meth:`~obspy.core.trace.Trace._ltrim`
#         """
#         old_fold = self.fold
#         old_npts = self.stats.npts
#         super()._ltrim(starttime, pad=pad, nearest_sample=nearest_sample, fill_value=fill_value)
#         if old_npts == 0:
#             self.fold = np.zeros(shape=self.data.shape, dtype=self.data.dtype)
#         elif old_npts < self.stats.npts:
#             self.fold = np.full(shape=self.data.shape, fill_value=0, dtype=self.data.dtype)
#             self.fold[-old_npts:] = old_fold
            
#         elif old_npts > self.stats.npts:
#             self.fold = old_fold[:self.stats.npts]
#         else:
#             self.fold = self.fold.astype(self.data.dtype)
#         self.enforce_zero_mask()
#         return self

#     def _rtrim(self, endtime, pad=False, nearest_sample=True, fill_value=None):
#         """
#         Run :meth:`~obspy.core.trace.Trace._rtrim` on this MLTrace object's data and fold arrays.
        
#         Padding values in fold entries are always 0 to indicate no valid observations are present, regardless of the fill_value selected

#         :param endtime: endtime for right trim
#         :type endtime: obspy.core.utcdatetime.UTCDateTime
#         :param pad: should extra samples be padded?, defaults to False
#         :type pad: bool, optional
#             also see :meth:`~obspy.core.trace.Trace._rtrim`
#         :param nearest_sample: passed to nearest_sample in inherited Trace._rtrim
#         :type nearest_sample: bool, optional
#             also see :meth:`~obspy.core.trace.Trace._rtrim`
#         :param fill_value: fill value to apply to pad entries in data
#         :type fill_value: NoneType, int, float
#             also see :meth:`~obspy.core.trace.Trace._rtrim`
#         """
#         old_fold = self.fold
#         old_npts = self.stats.npts
#         super()._rtrim(endtime, pad=pad, nearest_sample=nearest_sample, fill_value=fill_value)
#         if old_npts == 0:
#             self.fold = np.zeros(shape=self.data.shape, dtype=self.data.dtype)
#         elif old_npts < self.stats.npts:
#             self.fold = np.full(shape=self.data.shape, fill_value=0, dtype=self.data.dtype)
#             self.fold[:old_npts] = old_fold
#         elif old_npts > self.stats.npts:
#             self.fold = old_fold[:self.stats.npts]
#         else:
#             self.fold = self.fold.astype(self.data.dtype)
#         self.enforce_zero_mask()
#         return self

#     def split(self):
#         """
#         Slight modification to :meth:`~obspy.core.trace.Trace.split` that also applys the split operation to the MLTrace.fold attribute
#         also see :meth:`~obspy.core.trace.Trace.split`

#         :returns:
#             - 
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
#             tr.fold = self.fold[slice.start:slice.stop]
#             trace_list.append(tr)
#         return Stream(trace_list)
    
#     def apply_to_masked(self,method , super=False, **options):
#         """Applys a specified class method of :class:`~PULSE.data.mltrace.MLTrace`
#         to non-masked elements of its **data** and **fold** attributes by splitting
#         the mltrace into a :class:`~obspy.core.stream.Stream` of individually contiguous
#         mltrace objects, applys the method, and then merges the modified mltrace objects
#         using :meth:`~PULSE.data.mltrace.MLTrace.__add__`.
        
#         apply_to_masked assumes the specified method conducts in-place modifications
#         to a :class:`~PULSE.data.mltrace.MLTrace` object. It does not return outputs
#         of **method** and it does permanently alter the data of this mltrace.

#         :param method: name of the class method to apply
#         :type method: str
#         :param options: key word argument collector passed to the specified class method
#             Note: positional arguments can also be passed as kwargs!
        
#         """
#         if np.ma.is_masked(self.data):
#             fill_value = self.data.fill_value
#             # FIXME: Currently this makes a copy of the data, work out how to use get_view
#             # to make this a fully in-place operation.
#             st = self.split()
#             for _e, _mlt in enumerate(st):
#                 if super:
#                     getattr(super(), method)(**options)
#                 else:
#                     getattr(_mlt, method)(**options)
#                 if _e == 0:
#                     self = st[0]
#                 else:
#                     self.__add__(_mlt, method=0, fill_value=fill_value)
#         else:
#             if super:
#                 getattr(super(), method)(**options)
#             else:
#                 getattr(self, method)(**options)








    

#     # def apply_super_to_masked(self, method, **options):
#     #     if np.ma.is_masked(self.data):
#     #         fill_value = self.data.fill_value
#     #         st = self.split()
#     #         for _e, _mlt in enumerate(st)
#     ###############################################################################
#     # UPDATED RESAMPLING METHODS ##################################################
#     ###############################################################################
#     # def resample(self, sampling_rate, window='hann', no_filter=True, strict_length=False):



#     def resample(self, sampling_rate, window='hann', no_filter=True, strict_length=False):
#         """
#         Run the :meth:`~obspy.core.trace.Trace.resample` method on this MLTrace, interpolating
#         the self.fold attribute with numpy.interp using relative times to the original
#         MLTrace.stats.starttime value

#         see indepth description in :meth:`~obpsy.core.mltrace.MLTrace.resample`

#         :param sampling_rate: new sampling rate in samples per second
#         :type sampling_rate: float
#         :param window: window type to use, defaults to 'hann'
#         :type window: str, optional
#         :param no_filter: do not apply prefiltering, defaults to True
#         :type no_filter: bool, optional
#         :param strict_length: leave traces unchanged for which the endtime of the MLTrace
#             would change, defaults to False
#         :type strict_length: bool, optional
#         :return: view of this MLTrace
#         :rtype: PULSE.data.mltrace.MLTrace
#         """

#         tmp_fold = self.fold
#         tmp_times = self.times()
#         super().resample(sampling_rate, window=window, no_filter=no_filter, strict_length=strict_length)
#         tmp_fold = np.interp(self.times(), tmp_times, tmp_fold)
#         # Match data dtype
#         if self.data.dtype != tmp_fold.dtype:
#             tmp_fold = tmp_fold.astype(self.data.dtype)
#         self.fold = tmp_fold
#         return self

#     def interpolate(
#             self,
#             sampling_rate,
#             method='weighted_average_slopes',
#             starttime=None,
#             npts=None,
#             time_shift=0.0,
#             *args,
#             **kwargs
#     ):
#         """
#         Run the :meth:`~obspy.core.trace.Trace.interpolate` method on this MLTrace, interpolating
#         the self.fold attribute using relative times if `starttime` is None or POSIX times
#         if `starttime` is specified using numpy.interp

#         see :meth`~obspy.core.trace.Trace.interpolate` for detailed descriptions of inputs

#         :: INPUTS ::
#         :param sampling_rate: new sampling rate in samples per second
#         :type sampling_rate: float
#         :param method: interpolation method - default is 'weighted_average_slopes'
#         :type method: str
#         :param starttime: alternative start time for interpolation that is within the
#                 bounds of this MLTrace's starttime and endtime
#         :type starttime: NoneType or obspy.core.utcdatetime.UTCDateTime
#                           alternative starttime for interpolation (must be in the current
#                           bounds of the original MLTrace)
#         :param npts: new number of samples to interpolate to
#         :type npts: int or NoneType
#         :param time_shift: seconds to shift the starttime metadata by prior to interpolation
#         :type time_shift: float
#         :param *args: dditional positional arguments to pass to the interpolator in :meth `~obspy.core.trace.Trace.interpolate`
#         :type *args: list-like
#         :param **kwargs: additional key word arguments to pass to the interpolator in :meth: `~obspy.core.trace.Trace.interpolate`
#         :type **kwargs: kwargs

#         """
#         tmp_fold = self.fold
#         if starttime is None:
#             tmp_times = self.times()
#         else:
#             tmp_times = self.times(type='timestamp')
#         super().interpolate(
#             sampling_rate,
#             method=method,
#             starttime=starttime,
#             npts=npts,
#             time_shift=time_shift,
#             *args,
#             **kwargs)
#         if starttime is None:
#             tmp_fold = np.interp(self.times(), tmp_times, tmp_fold)
#         else:
#             tmp_fold = np.interp(self.times(type='timestamp'), tmp_times, tmp_fold)
#         # Length check
#         if self.stats.npts < len(tmp_fold):
#             # Trim off end values in the odd case where np.interp doesn't correctly scale tmp_fold
#             tmp_fold = tmp_fold[:self.stats.npts]
#         # dtype check
#         if self.data.dtype != tmp_fold.dtype:
#             tmp_fold = tmp_fold.astype(self.data.dtype)
#         # Write update to self
#         self.fold = tmp_fold
#         return self

#     def decimate(self, factor, no_filter=False, strict_length=False):
#         """
#         Run the :meth:`~obspy.core.trace.Trace.decimate` method on this MLTrace, interpolating
#         its `fold` attribute with :meth:`~numpy.interp` using relative times to the original
#         `stats.starttime` value

#         see :meth:`~obpsy.core.mltrace.MLTrace.decimate` for details on parameters and specific behaviors.

#         :param factor: decimation factor
#         :type factor: float
#         :param no_filter: do not apply contextual prefilter? Defaults to False
#         :type no_filter: bool, optional
#         :param strict_length: maintain strict length for output trace? Defaults to False
#         :type strict_length: bool, optional
        
#         """
#         tmp_fold = self.fold
#         tmp_times = self.times()
#         super().decimate(factor, no_filter=no_filter, strict_length=strict_length)
#         tmp_fold = np.interp(self.times(), tmp_times, tmp_fold)
#         # Match data dtype
#         if self.data.dtype != tmp_fold.dtype:
#             tmp_fold = tmp_fold.astype(self.data.dtype)
#         self.fold = tmp_fold
#         return self
          
#     ######################################
#     # ML SIGNAL PRE-PROCESSING METHODS ###
#     ######################################

        
#     def detrend(self, type='simple', **options):
#         options.update({'type': type})
#         self.apply_to_masked('detrend',**options)

#     def filter(self, type, **options):
#         options.update({'type': type})
#         self.apply_to_masked('filter', **options)


#     def treat_gaps(self,
#                    filterkw=False,
#                    detrendkw={'type': 'linear'},
#                    resample_method=False,
#                    resamplekw={},
#                    taperkw={'max_percentage':None, 'max_length': 0.06},
#                    mergekw={'method': 1, 'fill_value':0},
#                    trimkw=False
#                    ):
#         """
#         Conduct a self-contained processing pipeline to treat gappy data with the
#         following structure comprising required (req.) and optioanl (opt.) steps:

#         split -> filter -> detrend -> resample -> taper -> merge -> trim/pad
#         (req.)   (opt.)    (opt.)     (opt.)      (opt.)   (req.)   (opt.)

#         The minimum processing could be accomplished using a method that wraps the
#         :meth:`~numpy.ma.MaskedArray.filled()` method (e.g., MLTrace.trim()), however
#         the optional steps between split() and merge() represent typical steps
#         for data pre-processing prior to forming tensor for ML prediction.

#         :param filterkw: keyword arguments to pass to :meth:`~PULSE.data.mltrace.MLTrace.filter`
#             also see :meth:`~obspy.core.trace.Trace.filter
#         :type filterkw: dict or bool
#         :param detrendkw: keyword arguments to pass to :meth:`~PULSE.data.mltrace.MLTrace.detrend`,
#         :type detrendkw: dict or bool
#         :param resample_method: name of the resampling method to use. Supported values
#                                     - 'resample' -- :meth:`~obspy.core.trace.Trace.resample`
#                                     - 'interpolate' -- :meth:`~obspy.core.trace.Trace.interpolate`
#                                     - 'decimate' -- :meth:`~obspy.core.trace.Trace.decimate`
#         :type resample_method: str or bool
#         :param resamplekw: keyword arguments to pass to getattr(MLTrace, resample_method)(**resamplekw)
#         :type resamplekw: dict or bool
#         :param taperkw: keyword arguments to pass to :meth:`~PULSE.data.mltrace.MLTrace.taper`
#                     also see :meth:`~obspy.core.trace.Trace.filter
#         :type taperkw: dict or bool
#         :param mergekw: keyword arguments to pass to :meth:`~PULSE.data.mltrace.MLTrace.merge`
#                     also see :meth:`~obspy.core.trace.Trace.filter
#         :type mergekw: dict or bool
#         :param trimkw: keyword arguments to pass to :meth: `~PULSE.data.mltrace.MLTrace.trim`
#                     also see :meth:`~obspy.core.trace.Trace.filter
#         :type trimkw: dict or bool
        
        
#         TODO: Need to clean up the split/merge handling at the end

#         """
#         # See if splitting is needed
#         if isinstance(self.data, np.ma.MaskedArray):
#             st = self.split()
#         else:
#             st = obspy.Stream([self])
#         # Apply trace(segment) processing
#         for tr in st:
#             if isinstance(filterkw, dict):
#                 if filterkw['type'] == 'bandpass':
#                     if tr.stats.sampling_rate/2 < filterkw['freqmax']:
#                         tr = tr.filter('highpass', freq=filterkw['freqmin'])
#                     else:
#                         tr = tr.filter(**filterkw)
#                 else:
#                     tr = tr.filter(**filterkw)
#             elif filterkw is not None:
#                 raise TypeError('filterkw must be type dict or NoneType')
#             if isinstance(detrendkw, dict):
#                 tr = tr.detrend(**detrendkw)
#             elif detrendkw is not None:
#                 raise TypeError('detrendk kw myst be type dict or NoneType')
#             if isinstance(resample_method, str):
#                 if resample_method in [func for func in dir(tr) if callable(getattr(tr, func))]:
#                     if self.stats.sampling_rate != resamplekw['sampling_rate']:
#                         tr = getattr(tr,resample_method)(**resamplekw)
#                 else:
#                     raise TypeError(f'resample_method "{resample_method}" is not supported by {self.__class__}')
#             elif resample_method is not None:
#                 raise TypeError('resample_method must be type str or NoneType')
#             # TODO: Need to properly handle taper longer than trace UserWarning
#             if isinstance(taperkw, dict):
#                 if tr.stats.npts * tr.stats.delta < taperkw['max_length']:
#                     taperkw.update({'max_length': tr.stats.npts * tr.stats.delta * 0.5})
#                 tr = tr.taper(**taperkw)
#             elif taperkw is not None:
#                 raise TypeError('taperkw must be type dict or NoneType')
            
#         # If there is more than one trace, sequentailly re-stitch into a single trace
#         if len(st) > 1:
#             for _i, tr in enumerate(st):
#                 if _i == 0:
#                     first_tr = tr
#                 else:
#                     first_tr.__add__(tr, **mergekw)
#         # If we have one trace, just pull it out f the stream
#         elif len(st) == 1:
#             first_tr = st[0]
#         # If input is a fully masked MLTrace, such that split() produced an empty Stream
#         else:
#             # Update data and fold to be 0-traces
#             self.to_zero(method='both')
#             # And resample to target S/R
#             getattr(self, resample_method)(**resamplekw)
#             # raise NotImplementedError('handling for all-masked trace is under review')
#             # # Resample 
#             # getattr(self, resample_method)(**resamplekw)
#             # self.to_zero(method='both')
#         # If we have a valid output stream  
#         if len(st) > 0:
#             self.stats = first_tr.stats
#             self.data = first_tr.data
#             self.fold = first_tr.fold

#         if trimkw:
#             self.trim(**trimkw)

#         return self

#     def align_sampling(self, starttime=None, sampling_rate=None, npts=None, fill_value=None, window='hann', **kwargs):
#         """Given a discrete, time-domain index defined by a starttime, sampling_rate,
#         and number of samples, apply trimming, padding, interpolation, and/or resampling
#         to estimate data and fold values at sample points on this new time index.

#         This method wraps
#           - :meth:`~PULSE.data.mltrace.MLTrace.trim` to pad or trim data and fold vectors
#           - :meth:`~PULSE.data.mltrace.MLTrace.resample` to change sampling rates
#           - :meth:`~PULSE.data.mltrace.MLTrace.interpolate` to estimate data values on the
#           specified sampling index if the source sampling index is mis-aligned at a sample-by-sample level.

#         :param starttime: starttime for the new time index, defaults to None.
#             None value uses the starttime of this MLTrace as the reference starttime
#         :type starttime: UTCDateTime or None, optional
#         :param sampling_rate: sampling_rate for the new time index, defaults to None
#             None value uses the sampling_rate of this MLTrace as the reference sampling_rate
#         :type sampling_rate: float-like or None, optional
#         :param npts: the number of samples in the new time index, defaults to None
#             None value uses the npts of this MLTrace as the reference npts
#         :type npts: int-like or None, optional
#         :param fill_value: Default padding value with which to fill missing samples,
#                 passed to :meth:`~PULSE.data.mltrace.MLTrace.trim`, defaults to None
#             None value will be superceded by the leading data sample in this MLTrace
#                 in a particular case where interpolation is required and the specified starttime
#                 is before the starttime of this MLTrace, in which case padding samples are added
#                 to encompass the new starttime and subsequently trimmed off.
#         :type fill_value: int, float or None, optional
#         :param window: name of the window method to use for resampling, defaults to 'hann'.
#             also see :meth:`~obspy.core.trace.Trace.resample`, and references therein
#         :type window: str, optional
#         :param kwargs: key word argument collector for passing additional arguments to
#             :meth:`~PULSE.data.mltrace.MLTrace.interpolate`.
#         """        
#         # Test 0: Handle None inputs for required values
#         if not isinstance(starttime, UTCDateTime):
#             starttime = self.stats.starttime
#         if not isinstance(sampling_rate, (int, float)):
#             sampling_rate = self.stats.sampling_rate
#         elif sampling_rate < 0:
#             raise ValueError('sampling_rate must be non-negative')
#         if not isinstance(npts, int):
#             npts = self.stats.npts
#         elif npts < 0:
#             raise ValueError('npts must be non-negative')
#         # Explicitly calculate target endtime
#         endtime = starttime + npts/sampling_rate

#         # Test 1: If starttime does not line up with target starttime
#         if self.stats.starttime != starttime:
#             # Check 1.1: Interpolation not required, just trimming to start trace at starttime
#             if self.is_utcdatetime_in_sampling(starttime):
#                 self.trim(starttime=starttime,pad=True,fill_value=fill_value)
#             # Check 1.2: Interpolation needed to re-align sampling indices
#             else:
#                 # Safety catch on NoneType fill_value
#                 if not isinstance(fill_value, (int, float)):
#                     leading_fill_value = self.data[0]
#                 else:
#                     leading_fill_value = fill_value
#                 # Pad by 5 extra samples beyond the target starttime with the leading data value
#                 self.trim(starttime=starttime - self.stats.delta*5,
#                           pad=True,
#                           fill_value=leading_fill_value)
#                 self.interpolate(sampling_rate=self.stats.sampling_rate,
#                                     starttime=starttime,
#                                     **kwargs)
                
#         # Test 2: If sampling_rate does not line up with target sampling_rate
#         if self.stats.sampling_rate != sampling_rate:
#             # Check 2.1: is this downsampling? If so (no_filter=False) apply auto-lowpass (chebychev-2)
#             no_filter = self.stats.sampling_rate < sampling_rate
#             self.resample(sampling_rate,
#                           no_filter=no_filter,
#                           window=window)

#         # Test 3: If npts does not equate, pad
#         if self.stats.npts != npts:
#             self.trim(starttime=starttime,
#                       endtime=endtime,
#                       pad=True,
#                       fill_value=fill_value,
#                       nearest_sample=True)
        

#     def sync_to_window(self, starttime=None, endtime=None, fill_value=None, pad_after=True, **kwargs):
#         """
#         Syncyronize the time sampling index of this trace to a specified
#         starttime and/or endtime using a combination of the (ML)Trace.trim()
#         and (ML)Trace.interpolate() methods.

#         Interpolation is triggered only if the specified starttime / endtime
#         is not aligned with the temporal sampling of this trace.

#         :: INPUTS ::
#         :param starttime:   [None] - no reference starttime, fits to nearest
#                              starttime that is consistent with MLTrace data
#                              and specified endtime
#                             [UTCDateTime] - reference starttime for trim and interpolate
#                             defaults to None
#         :type starttime: None or obspy.core.utcdatetime.UTCDateTime
#         :param endtime:     [None] - no reference endtime
#                             [UTCDateTime] - reference endtime
#                             defaults to None
#         :type endtime: None or obspy.core.utcdatetime.UTCDateTime
#         :param fill_value:  [None], [int], [float] - value passed to (ML)Trace.trim(), defaults to 0
#         :type fill_value: None, int, or float
#         :param pad_after: should padding out to inputs `starttime` and `endtime` be enforced after
#                         running sampling synchronization?, defaults to True
#         :type pad_after: bool
#         :param **kwargs:    [kwargs] key word argument collector passed
#                             to (ML)Trace.interpolate()
#         :type **kwargs: key-word arguments
#         :return: view of this object
#         :rtype: PULSE.data.mltrace.mltrace.MLTrace

#         """                 
#         if starttime is not None:
#             try:
#                 starttime = UTCDateTime(starttime)
#             except TypeError:
#                 raise TypeError
            
#         if endtime is not None:
#             try:
#                 endtime = UTCDateTime(endtime)
#             except TypeError:
#                 raise TypeError
#         # If no bounds are stated, return self
#         if endtime is None and starttime is None:
#             return self
#         # If only an endtime is stated
#         elif starttime is None:
#             # Nearest sample aligned with endtime that falls after starttime
#             dn = (endtime - self.stats.starttime)//self.delta
#             # So do not apply as (dn + 1) * delta
#             starttime = endtime - dn*self.stats.delta
#         else:
#             dn = (self.stats.starttime - starttime)*self.stats.sampling_rate
#         # Sampling points align, use MLTrace.trim() to pad to requisite size if pad_after is True
#         if int(dn) == dn:
#             if self.stats.starttime != starttime or self.stats.endtime != endtime:
#                 self.trim(starttime=starttime,endtime=endtime, pad=pad_after, nearest_sample=True, fill_value=fill_value)
#         # If samplign points don't align use MLTRace.interpolate() and .trim() to re-index and trim to size
#         else:
#             # If starttime is inside this MLTrace
#             if self.stats.starttime < starttime < self.stats.endtime:
#                 self.interpolate(sampling_rate=self.stats.sampling_rate,
#                                  starttime=starttime,**kwargs)
#             # If starttime is before the start of the MLTrace
#             elif starttime < self.stats.starttime:
#                 # Get time of the first aligned sample inside MLTrace
#                 dn = 1 + (self.stats.starttime - starttime)//self.stats.delta
#                 istart = starttime + dn*self.stats.delta
#                 # Interpolate on that sample
#                 self.interpolate(sampling_rate=self.stats.sampling_rate,
#                                  starttime=istart,**kwargs)
#             # If after padding is enabled
#             if pad_after:
#                 if self.stats.starttime != starttime or self.stats.endtime != endtime:
#                     self.trim(starttime=starttime, endtime=endtime, pad=pad_after, nearest_sample=True, fill_value=fill_value)
                        
#         return self

#     def normalize(self, norm_type='max'):
#         """Normalize data in this trace by specified method
#         Extension of the obspy.core.trace.Trace.normalize method
#         where norm_type dictates what scalar is calculated for:
#         [ML](Trace)
#             super().normalize(norm=scalar)

#         :param norm_type: normalization method name, defaults to 'max'
#         :type norm_type: str, optional
#             Supported:
#                 'max': maximum absolute value of trace data
#                     aliases: 'minmax','peak'
#                 'std': standard deviation of trace data
#                     aliases: 'standard'
        
#         :return: view of this MLTrace object
#         :rtype: PULSE.data.mltrace.MLTrace

#         """
#         # Safety catch for all-0 or all-masked arrays
#         if np.all(self.data == 0):
#             return self
#         elif np.ma.is_masked(self.data):
#             if all(self.data.mask):
#                 Logger.warning('encountered all masked array in normalize')
#                 return self
#             else:
#                 pass
#         else:
#             pass
#         if norm_type.lower() in ['max','minmax','peak']:
#             scalar = np.nanmax(np.abs(self.data))
#             super().normalize(norm=scalar)
#         elif norm_type.lower() in ['std','standard']:
#             scalar = np.nanstd(self.data)
#             super().normalize(norm=scalar)
#         return self

#     # TRIGGER METHOD ##############################################################
#     # TODO: Migrate this to a triggering POP submodule
#     def prediction_trigger_report(self, thresh,
#                            blinding=(None, None),
#                            min_len=5, max_len=9e99,
#                            extra_quantiles=[0.159, 0.25, 0.75, 0.841],
#                            stats_pad=20,
#                            decimals=6,
#                            include_processing_info=False):
#         """Conduct triggering on what is assumed to be the output of a continuous time-series
#         output from a ML classifier (e.g., PhaseNet). This method estimatesstatistical measures
#         of a trigger returned by :meth: `~obspy.signal.trigger.trigger_onset`
#         assuming the samples of the trigger represent a data distribution y(x).


#         :param thresh: triggering threshold, typically \in (0, 1)
#         :type thresh: float
#         :param blinding: left and right blinding samples, defaults to (None, None)
#         :type blinding: tuple of int, optional
#         :param min_len: minimum trigger length in samples, defaults to 5
#         :type min_len: int-like, optional
#         :param max_len: maximum trigger length in samples, defaults to 9e99
#         :type max_len: int-like, optional
#         :param extra_quantiles: additional quantiles to estimate, defaults to [0.159, 0.25, 0.75, 0.841]
#             in addition to Q = 0.5 (the median)
#         :type extra_quantiles: list, optional
#         :param stats_pad: number of samples to pad outward from the trigger onset and offset samples to
#             use for calculating statistical measures of a trigger, defaults to 20
#         :type stats_pad: int, optional

#         :TODO: Obsolite these two arguments and underlying subroutines
#         :param decimals: precision of values, defaults to 6
#         :type decimals: int, optional
#         :param include_processing_info: Should processing information from the MLTrace , defaults to False
#         :type include_processing_info: bool, optional

#         :return: _description_
#         :rtype: _type_

#         report fields:
#         network     - network code of this MLTrace
#         station     - station code
#         location    - location code
#         channel     - channel code (component replaced with label)
#         model       - model name
#         weight      - weight name
#         label       - label code
#         t0          - reference starttime (in epoch)
#         SR          - sampling rate (Hz)
#         npts        - number of samples in this MLTrace
#         lblind      - number of samples blinded on the left end of trace
#         rblind      - number of samples blinded on the right end of trace
#         stats_pad   - number of samples added to either side of trigger(s) for estimating statistics
#         thrON       - threshold ON value = thresh
#         iON         - sample index of trigger onset
#         thrOFF      - threshold OFF value = thresh
#         iOFF        - sample index of trigger conclusion
#         pMAX        - maximum value of trigger
#         iMAX        - sample index of the maximum value in trigger
#         imean       - sample index of the mean of the trigger
#         istd        - estimated standard deviation in number of samples 
#         iskew       - estimated skew in number of samples
#         ikurt       - estimated kurtosis in number of samples
#         pQ?         - trigger value at the ? quantile
#         iQ?         - position of the ? quantile value

#         """        
#         if self.stats.model == self.stats.defaults['model']:
#             raise ValueError('model is unassigned - not working on a prediction trace')
#         elif self.stats.weight == self.stats.defaults['weight']:
#             raise ValueError('weight is unassigned - not working on a prediction trace')
#         else:
#             pass
#         output = []
#         cols = ['network','station','location','channel','model','weight','label',
#                 't0','SR','npts','lblind','rblind','stats_pad',
#                 'thrON','iON','thrOFF','iOFF','pMAX','iMAX',
#                 'imean','istd','iskew','ikurt']
#         quantiles = [0.5]
#         if isinstance(extra_quantiles, list):
#             if all([0<=_q<=1 for _q in extra_quantiles]):
#                 quantiles += extra_quantiles
#             else:
#                 raise ValueError
#         elif extra_quantiles is None:
#             pass
#         else:
#             raise TypeError
        
#         for _q in quantiles:
#             cols += [f'pQ{_q:.3f}', f'iQ{_q:.3f}']

#         triggers = obspy.signal.trigger.trigger_onset(self.data[blinding[0]:-blinding[1]], thresh, thresh)
#         if blinding[0] is not None:
#             offset = blinding[0]
#         else:
#             offset = 0
#         for trigger in triggers:
#             # Correct for blinding offset
#             trigger[0] += offset - stats_pad
#             trigger[1] += offset + stats_pad
#             if min_len <= trigger[1] - trigger[0] <= max_len:
#                 trig_data = self.data[trigger[0]:trigger[1]]
#                 trig_index = np.arange(trigger[0], trigger[1])
#                 imax = trigger[0] + np.nanargmax(trig_data)
#                 pmax = self.data[imax]
#                 iq, pq = estimate_quantiles(trig_index, trig_data, q=quantiles)
#                 line = [self.stats.network,
#                         self.stats.station,
#                         self.stats.location,
#                         self.stats.channel,
#                         self.stats.model,
#                         self.stats.weight,
#                         self.stats.component,
#                         np.round(self.stats.starttime.timestamp, decimals=decimals),
#                         self.stats.sampling_rate,
#                         self.stats.npts,
#                         blinding[0], blinding[1],stats_pad,
#                         thresh,trigger[0] + stats_pad,
#                         thresh,trigger[1] - stats_pad,
#                         pmax,imax]
#                 norm_stats = list(estimate_moments(trig_index, trig_data))
#                 # breakpoint()
#                 line += norm_stats
#                 for _i, _p in zip(iq, pq):
#                     line += [_p, _i]
#                 output.append(line)
#         if len(output) > 0:
#             df_out = pd.DataFrame(output, columns=cols)
#             if include_processing_info:
#                 holder =[]
#                 for line in self.stats.processing:
#                     iline = line.copy()
#                     iline += [self.id, self.stats.starttime, self.stats.endtime]
#                     holder.append(iline)
#                 holder.append(['MLTrace','prediction_trigger_report','output',time.time()])
#                 df_proc = pd.DataFrame(holder, columns=['class','method','status','timestamp','id','starttime','endtime'])

#                 return df_out, df_proc
#             else:
#                 return df_out
#         else:
#             return None            


#     ###############################################################################
#     # I/O METHODS #################################################################
#     ###############################################################################
    
#     def to_trace(self, fold_threshold=0, attach_mod_to_loc=False):
#         """
#         Convert this MLTrace into a :class:`~obspy.core.trace.Trace` object,
#         masking values that have fold less than `fold_threshold`

#         :param fold_threshold: minimum fold value to consider as "valid" data, all
#             datapoints with fold lower than fold_threshold are converted to masked
#             values, defaults to 0
#         :type fold_threshold: int-like
#         :param output_mod: should the model and weight names be included in the output?, defaults to False
#         :type output_mod: bool, optional
#         :returns:
#             - **tr** (*obspy.core.trace.Trace*) -- trace object with data masked on samples with fold < fold_threshold
            
#         """
#         tr = Trace()
#         for _k in tr.stats.keys():
#             if _k not in tr.stats.readonly:
#                 tr.stats.update({_k: self.stats[_k]})
#         if attach_mod_to_loc:
#             tr.stats.location = f'{tr.stats.location}∂{self.stats.model}∂{self.stats.weight}'

#         data = self.data
#         mask = self.fold < fold_threshold
#         if any(mask):
#             data = np.ma.masked_array(data=data,
#                                       mask=mask)
#         tr.data = data
#         return tr
    
#     def read(file_name):
#         """
#         Read a MSEED file that was generated by :meth:`~PULSE.data.mltrace.MLTrace.write` to reconstitute a new MLTrace object from
#         a specificlly formatted MiniSEED file.

#         :param file_name: file to read
#         :type file_name: str
#         :return mlt: reconstituted MLTrace 
#         :rtype: PULSE.data.mltrace.MLTrace
#         """        
#         # Get dictionary of pretrained model-weight combinations
#         ptd = pretrained_dict()
#         # read input file and merge in case data saved were gappy
#         st = read(file_name, fmt='MSEED').merge()
#         # Confirm there are 2 traces (data & fold)
#         if len(st) == 2:
#             for tr in st:
#                 # Catch fold trace
#                 if tr.stats.network == 'FO' and tr.stats.location=='LD':
#                     fold_tr = tr
#                 # Catch data trace
#                 else:
#                     data_tr = tr
#         # Initially populate header from data trace
#         header = data_tr.stats
#         # Reconstitute model & weight labels from fold trace header
#         for _m, _v in ptd.items():
#             if fold_tr.stats.station != '':
#                 if fold_tr.stats.station in _m:
#                     header.update({'model': _m})
#                     if fold_tr.stats.channel != '':
#                         for _w in _v:
#                             if fold_tr.stats.channel in _w:
#                                 header.update({'weight': _w})
#                                 break
#                     break
#         # Complete mltrace reconstitution
#         mltr = MLTrace(data=data_tr.data, fold=fold_tr.data, header=header)
#         return mltr

#     def _prep_fold_for_mseed(self):
#         """
#         PRIVATE METHOD

#         conduct processing steps to convert a copy of the fold,
#         model-code, and weight-code attributes contained in this
#         MLTrace object to compose a MSEED compliant trace that
#         can be saved with the :meth: `~obspy.core.stream.Stream.write`
#         method

#         :return fold_trace: fold trace
#         :rtype fold_trace: obspy.core.trace.Trace
#         """        
#         fold_trace = self.copy()
#         fold_trace.data = self.fold
#         # Shoehorn Model and Weight info into NSLC strings
#         fold_trace.stats.update({'network': 'FO',
#                                  'location': 'LD',
#                                  'station': self.stats.model[:5],
#                                  'channel': self.stats.weight[:3]})
#         return fold_trace
    
#     def write(self, file_name, pad=False, fill_value=None, **options):
#         """
#         Write the contents of this MLTrace object to a miniSEED file comprising two
#         trace types to preserve added labeling attributes 'model' and 'weight' in the
#         MLStats object attached to this MLTrace and the fold attribute contents of this MLTrace
#         DATA:
#             Net.Sta.Loc.Chan trace - with self.data as the data
#         AUX:
#             FO.Model.LD.Wgt - with self.fold as the data
#         Where the Net 'FO' and location 'LD' are fixed strings used as a flag that
#         this is an auxillary trace providing. 

#         This method wraps the :meth: `~obspy.core.stream.Stream.write` method and
#         forces the format "MSEED"

#         :param file_name: file name to save this MLTrace to
#         :type file_name: str
#         :param pad: should padding be applied to treat gaps?, defaults to False
#         :type pad: bool, optional
#         :param fill_value: fill_value to pass to padding (if pad is True), defaults to None
#         :type fill_value: NoneType, int, or float, optional
#         :param **options: key word argument gatherer to pass to :meth: `~obspy.core.stream.Stream.write`
#             Note: will automatically remove `fmt` inputs
#         :type **options: kwargs
#         :return st: the formatted stream saved to disk
#         :rtype st: obspy.core.stream.Stream
#         """
#         st = Stream()
#         # Convert data vector from MLTrace into
#         if isinstance(self.data, np.ma.MaskedArray):
#             if np.ma.is_masked(self.data):
#                 # Padding approach to handling gappy/padded data
#                 if pad:
#                     padded_copy = self.copy()
#                     if fill_value is not None:
#                         padded_copy.data = padded_copy.data.filled(fill_value=fill_value)
#                     else:
#                         raise NotImplementedError
#                     data_trace = padded_copy.to_trace(attach_mod_to_loc=False)
#                     fold_trace = padded_copy._prep_fold_for_mseed()
#                     st += data_trace
#                     st += fold_trace
#                 # Splitting approach to handling gappy/padded data
#                 else:
#                     split_copy = self.copy().split()
#                     for mlt in split_copy:
#                         st += mlt.to_trace(attach_mod_to_loc=False)
#                         st += mlt._prep_fold_for_mseed()
#         # Safety catch against fmt being specified in options
#         if 'fmt' in options:
#             options.pop('fmt')   
#         # Write to disk       
#         st.write(file_name, fmt='MSEED', **options)
#         return st

#     ###############################################################################
#     # ID PROPERTY ASSIGNMENT METHODS ##############################################
#     ###############################################################################

#     def get_site(self):
#         """return the site code (Network.Station) of this MLTrace
#         :return site: site code
#         :rtype site: str
#         """        
#         hdr = self.stats
#         site = f'{hdr.network}.{hdr.station}'
#         return site
    
#     site = property(get_site)

#     def get_inst(self):
#         """return the instrument code (Band Character + Instrument Character)
#         from this MLTrace's channel code

#         :return inst: instrument code
#         :rtype: str
#         """        
#         hdr = self.stats
#         if len(hdr.channel) > 0:
#             inst = f'{hdr.location}.{hdr.channel[:-1]}'
#         else:
#             inst = f'{hdr.location}.{hdr.channel}'
#         return inst
    
#     inst = property(get_inst)

#     def get_comp(self):
#         """get the component character for this MLTrace object

#         NOTE:
#         EWFlow uses the component code to denote ML model prediction
#         labels (e.g. Detection in EQTransformer becomes comp = 'D')

#         :return comp: component/label character
#         :rtype comp: str
#         """        
#         hdr = self.stats
#         if len(hdr.channel) > 0:
#             comp = hdr.channel[-1]
#         else:
#             comp = ''
#         return comp
    
#     def set_comp(self, other):
#         """convenience method for changing the component code of this MLTrace

#         :param other: new character to assign to the component character
#         :type other: str
#         """        
#         char = str(other).upper()[0]
#         self.stats.channel = self.stats.channel[:-1] + char            
    
#     comp = property(get_comp)

#     def get_mod(self):
#         """Get the MOD code (Model.Weight) for this MLTrace

#         :return mod: MOD code
#         :rtype mod: str
#         """        
#         hdr = self.stats
#         mod = f'{hdr.model}.{hdr.weight}'
#         return mod
    
#     def set_mod(self, model=None, weight=None):
#         """Set the MOD code (Model.Weight) for this MLTrace

#         :param model: new model code, defaults to None
#             None input does not modify the stats.model attribute
#         :type model: str or NoneType, optional
#         :param weight: new weight code, defaults to None
#             None input does not modify the stats.weight attribute
#         :type weight: str or NoneType, optional
#         """        
#         if model is not None:
#             self.stats.model = model
#         if weight is not None:
#             self.stats.weight = weight

#     mod = property(get_mod)

#     def get_id(self):
#         """Get the full id of this MLTrace Object
#         id = Network.Station.Location.Channel.Model.Weight

#         :return id: id string
#         :rtype id: str
#         """        
#         id = f'{self.site}.{self.inst}{self.comp}.{self.mod}'
#         return id

#     id = property(get_id)

#     def get_nslc(self):
#         """
#         Get the NSLC ID for this MLTrace object
#         """
#         id = f'{self.stats.network}.{self.stats.station}.{self.stats.location}.{self.stats.channel}'
#         return id
#     nslc = property(get_nslc)

#     def get_scnl(self):
#         """
#         Get the SCNL ID for this MLTrace object
#         """
#         id = f'{self.stats.station}.{self.stats.channel}.{self.stats.network}.{self.stats.location}'
#         return id
#     scnl = property(get_scnl)

#     def get_instrument_id(self):
#         """Get the instrument id (NSLC minus component code) of this MLTrace

#         :return id: instrument id code
#         :rtype id: str
#         """        
#         id = f'{self.site}.{self.inst}'
#         return id
    
#     instrument = property(get_instrument_id)

#     def get_valid_fraction(self, thresh=1):
#         """Convenience method for getting the valid fraction for this whole MLTrace

#         compare to :meth:`~PULSE.data.mltrace.MLTrace.get_fvalid_subset`

#         :param thresh: fold threshold, defaults to 1
#         :type thresh: int, optional
#         :return: fraction of data that are "valid"
#         :rtype: float
#         """        
#         npts = self.stats.npts
#         nv = np.sum(self.fold >= thresh)
#         return nv/npts
    
#     fvalid = property(get_valid_fraction)

#     def get_id_keys(self):
#         """Get a dictionary of commonly used trace naming strings

#         :return:
#          **id_keys** (*dict*) -- dictionary of attribute names and values 
#         """        
#         id_keys = {'id': self.id,
#                     'instrument': self.instrument,
#                     'site': self.site,
#                     'inst': self.inst,
#                     'component': self.comp,
#                     'mod': self.mod,
#                     'network': self.stats.network,
#                     'station': self.stats.station,
#                     'location': self.stats.location,
#                     'channel': self.stats.channel,
#                     'model': self.stats.model,
#                     'weight': self.stats.weight
#                     }
#         # If id is not in traces.keys() - use dict.update
#         return id_keys
    
#     id_keys = property(get_id_keys)

#     def why(self):
#         """Have fun storming the castle"""
#         print('True love is the greatest thing on Earth - except for an MLT...')

#     # TODO: Is this method necessary as a core method?
#     def to_dtype(self, dtype):
#         """Conduct an in-place change of the data and fold dtype to a new dtype

#         Supported types:
#          - np.int8
#          - np.int16
#          - np.int32
#          - np.float32
#          - np.float64

#         :param dtype: New dtype to assign to the data and fold numpy arrays
#         :type dtype: type - see aboce
#         """        
#         if isinstance(dtype, type):
#             if dtype in [np.int8, np.int16, np.int32, np.float32, np.float64]:
#                 self.data = self.data.astype(dtype)
#                 self.fold = self.fold.astype(dtype)
#             else:
#                 raise ValueError(f'dtype type {dtype} not supported')
#         else:
#             raise TypeError(f'dtype must be a `type` not type {type(dtype)}')




# # TODO: Consider migrating this to an IO submodule
# def wave2mltrace(wave):
#     """
#     Convert a PyEW wave dictionary message into a :class:`~PULSE.core.mltrace.MLTrace` object
#     """
#     status = is_wave_msg(wave)
#     if isinstance(status, str):
#         raise SyntaxError(status)
#     header = {}
#     for _k, _v in wave.items():
#         if _k in ['station','network','channel','location']:
#             header.update({_k:_v})
#         elif _k == 'samprate':
#             header.update({'sampling_rate': _v})
#         elif _k == 'data':
#             data = _v
#         elif _k == 'startt':
#             header.update({'starttime': UTCDateTime(_v)})
#         elif _k == 'datatype':
#             dtype = _v
#     try:
#         data = wave['data'].astype(dtype)
#     except TypeError:
#         data = wave['data']
#     mlt = MLTrace(data=data, header=header)
#     return mlt

# # TODO: Consider moving this to an IO submodule
# def read_mltrace(data_file, **obspy_read_kwargs):
#     """
#     Reconstitute a MLTrace object from output file(s) generated by :meth:`~PULSE.data.mltrace.MLTrace.write` that are associated as
#         {common/path}/{common_file_name}_DATA.{extension} <- use this as 'data_file' input
#         {common/path}/{common_file_name}_FOLD.{extension}
#         {common/path}/{common_file_name}_PROC.txt
    
#     If the method is successful in reading the `data_file` as indicated above,
#     it will use common path/file/extension naming elements to attempt to find
#     the _FOLD and _PROC files to populate the MLTrace.fold and MLTrace.stats.processing
#     attributes. 

#     If the _FOLD and/or _PROC files are not found, default values are assigned
#     for missing information consistent with the MLTrace.__init__ method.

#     :: INPUTS ::
#     :param data_file: [str] path/file name for a *_DATA.{ext} waveform 
#             data file written to disk by MLTrace.write()
#     :obspy_read_kwargs: [kwargs] collector for keyword arguments to pass to
#             the obspy.core.stream.read() method.

#     :: OUTPUT ::
#     :return mlt: [PULSE.data.mltrace.mltrace.MLTrace] mltrace object
        
#     """
#     tr = obspy.read(data_file, **obspy_read_kwargs)
#     if len(tr) == 1:
#         tr = tr[0]
#     else:
#         raise ValueError('data_file contains more than one trace!')
#     # Get metadata from filename
#     path, file_ext = os.path.split(data_file)
#     file, ext = os.path.splitext(file_ext)
#     # Get id from start of file name
#     parts = file.split('_')
#     id = parts[0]
#     common_name = '_'.join(parts[:-1])
#     # Break into components
#     n,s,l,c,m,w = id.split('.')
#     # initialize MLTrace with tr as information
#     mlt = MLTrace(data=tr)
#     # Update model
#     mlt.stats.model=m
#     # Update weight
#     mlt.stats.weight=w
#     # Try to find fold and processing information
#     fold_file = os.path.join(path, f'{common_name}_FOLD.{ext}')
#     proc_file = os.path.join(path, f'{common_name}_PROC.txt')
#     # If fold file is found
#     if os.path.exists(fold_file):
#         ftr = read(fold_file)[0]
#         mlt.fold = ftr.data
#     # If processing file is found
#     if os.path.isfile(proc_file):
#         with open(proc_file, 'r') as _p:
#             lines = _p.readlines()
#         for _l in lines:
#             # Strip newline character
#             _l = _l[:-1]
#             # Split comma separated into list
#             _l = _l.split(',')
#             # Append to processing
#             mlt.stats.processing.append(_l)
#         _p.close()
#     return mlt