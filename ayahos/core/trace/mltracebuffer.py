"""
:module: ayahos.core.trace.mltracebuffer
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose: This module provides a class definition for a waveform buffer that has the added
        attributes of the ayahos.data.mltrace.MLTrace class and works in an analogus way
        as the obspy.realtime.rttrace.RTTrace class is to the obspy.core.trace.Trace class.

        Additionally, this class provides some additional safety catch options to guard against
        appending data with spurious timing information and functionalities to use the machine learning
        "stack" style appends used on predicted values.
"""
import numpy as np
from obspy import Trace
from ayahos.core.trace.mltrace import MLTrace
from ayahos.util.input import bounded_floatlike
    
class MLTraceBuffer(MLTrace):

    def __init__(self,
        max_length=1,
        blinding=None,
        restrict_past_append=True,
        **add_kwargs):
        """
        Initialize an MLTraceBuffer object containing no data and default metadata

        :: INPUTS ::
        :param max_length: [float] positive-valued maximum record length in seconds
        :param blinding: [None], [positive int], or [2-tuple of positive int] 
                    bliding to apply to traces appended to this MLTraceBuffer
                    (including the initial trace) using the inherited
                    ayahos.data.mltrace.MLTrace.apply_blinding() class method
        :param restrict_past_append: [bool] enforce restrictions on appends that
                    would add chronologically older data to the buffer
                    ~also see MLTraceBuffer.append()
        :param **add_kwargs: [kwargs] key word argument collector that passes
                    alternative kwarg values to the MLTrace.__add__ method around
                    which the MLTraceBuffer.append primary method is wrapped.
                    NOTE: These can be superceded in subsequent calls of
                    MLTraceBuffer.append() using **kwarg inputs
                    ~also see MLTraceBuffer.append()
        """
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
        if blinding is None or not blinding:
            self.blinding = False
        elif isinstance(blinding, (list, tuple)):
            if len(blinding) == 2:
                if all(int(_b) >= 0 for _b in blinding):
                    self.blinding = (int(blinding[0]), int(blinding[1]))
                else:
                    raise ValueError
            elif len(blinding) == 1:
                if int(blinding[0]) >= 0:
                    self.blinding = (int(blinding[0]), int(blinding[0]))
        elif isinstance(blinding, (int, float)):
            if int(blinding) >= 0:
                self.blinding = (int(blinding), int(blinding))
            else:
                raise ValueError
        else:
            raise TypeError
         

        # Compatability check for restrict past appends
        if not isinstance(restrict_past_append, bool):
            raise TypeError
        else:
            self.RPA = restrict_past_append

        self.add_kwargs = add_kwargs

        # Initialize _has_data private flag attribute
        self._has_data = False

    # def __add__(self, other):
    #     """
    #     Wrapper around the ayahos.data.mltracebuffer.MLTraceBuffer.append method
    #     that uses the default key word arguments for MLTrace.__add__ or alternative
    #     kwargs specified when the MLTraceBuffer was initialized (see the append_kwargs attribute)

    #     Use:
    #     tr = Trace(data=np.ones(300), header={'sampling_rate': 10})
    #     mltb = MLTraceBuffer(max_length=30)
    #     mltb += tr
    #     """
    #     self.append(other, **self.add_kwargs)
    #     return self

    def append(self, other, **kwargs):
        """
        Primary method for adding data to this MLTraceBuffer object
        
        This method wraps the ayahos.data.mltrace.MLTrace.__add__ special method and
        provides some additional checks relevant to semi-sequential data streaming
        and buffering described in the following scenarios.

        Append Scenarios

        FIRST APPEND
        If no data have been appended to this MLTraceBuffer, uses the 
        _first_append() private method to append data and metadata.
        NOTE: Unlike obspy.realtime.rttrace.RTTrace, this initial append
        enforces a strict max_length for buffered data and references
        to the endtime of the data being appended (see MLTraceBuffer._first_append())

        ADDITIONAL APPENDS

        FUTURE APPEND) other extends beyond the endtime of this MLTraceBuffer
            NEAR FUTURE - some of the contents of the buffer and other
                    coexist within a `max_length` window. Data in the buffer
                    are shifted to meet the endtime of other and data before
                    the new starttime are truncated
            FAR FUTRUE - no contents of the buffer and other coexist within a
                    `max_length` window. Data in the buffer are discarded
                    and the .data and .fold arrays are re-initialized with the
                    contents of `other`
                    
            NOTE: Future appends uniformly favor preserving trailing samples
                    should `other` exceed the size of the buffer.
                        uses the _first_append() private method with a reference
                        to the `endtime`
        PAST APPEND)
            If self.RPA = True (Restrict Past Append)
                NEAR PAST - `other` is trimmed to fit inside available space 
                        in the buffer without omitting any current valid (unmasked)
                        data in the buffer
                FAR PAST - append of `other` is canceled
            If self.RPA = False (no Restrict Past Append)
                NEAR PAST - contents of the buffer are slid to accommodate the contents
                            of `other`
                FAR PAST - data and fold are re-initialized in th same manner as
                        FAR FUTURE appends.
        INTERNAL APPEND)
            Unrestricted append that conforms to the behaviors of MLTrace.__add__()
            NOTE: In an earlier version this only allowed merges where self.fold
                    was 0-valued. This has been relaxed.


        :: INPUTS ::
        :param other: [obspy.Trace] like object to append to this MLTraceBuffer


        :param kwargs: [kwargs] key-word arguments passed to MLTrace.__add__()
                    NOTE: any kwarg in kwargs that have a matching key to
                        self.add_kwargs will be superceded by inputs to **kwargs
                        here.

        :: OUTPUT ::
        :return self: [ayahos.data.mltracebuffer.MLTraceBuffer] enable cascading
        """
        if self._has_data:
            if self.id != other.id:
                raise ValueError(f'trace ID\'s do not match {self.id} vs {other.id}')

        # # Merge self.add_kwargs and user specified kwargs
        # for _k, _v in self.add_kwargs.items():
        #     breakpoint()
        #     if _k not in kwargs.keys():
        #         kwargs.update({_k, _v})

        if isinstance(other, Trace):
            # If other is a trace, but not an MLTrace, convert
            if not isinstance(other, MLTrace):
                other = MLTrace(other)
        else:
            raise TypeError('input other must be type obspy.core.trace.Trace or a child-class thereof')
        
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
                    # Conduct future append (always unrestricted)
                    self._slide_buffer(other.stats.endtime, reference_type='endtime')
                    self.__add__(self, other, **self.add_kwargs)
                    # self.enforce_max_length(reference='endtime')
                # If other starts later that self end + max_length - big gap
                else:
                    # Run as a first append if id matches
                    if self.id == other.id:
                        self._has_data = False
                        self._first_append(other)

            # (PAST APPEND) If other starts at or before self
            elif other.stats.starttime <= self.stats.starttime:
                # FAR PAST
                if self.stats.starttime - other.stats.endtime >= self.max_length:
                    # IF restriction in place
                    if self.RPA:
                        # Return self (cancel append)
                        pass
                    # IF restriction is not in place, run as first_append
                    else:
                        # Only if ID matches
                        if self.id == other.id:
                            self._has_data = False
                            self._first_append(other)
                # NEAR PAST
                else:
                    # If restricting past appends - trim other and append to buffer
                    if self.RPA:
                        # Trim other
                        other.trim(starttime=self.stats.starttime)
                        self.__add__(other, **self.add_kwargs)
                    # If not restricting past appends - slide buffer and append full other
                    else:
                        self._slide_buffer(other.stats.endtime, reference_type='endtime')
                        self.__add__(other, **self.add_kwargs)

            # (INNER APPEND)
            else:
                self.__add__(other, **self.add_kwargs)
                # # TODO: Make sure this is a copy
                # ftr = self.get_fold_trace().trim(starttime=other.stats.starttime, endtime=other.stats.endtime)
                # # If there are any 0-fold data in self that have information from other
                # if (ftr.data == 0 & other.fold >0).any():
                #     self.__add__(other, **kwargs)
                # else:
                #     pass
            
        return self
    
    # @_add_processing_info          
    def _first_append(self, other, overflow_ref='endtime'):
        """
        PRIVATE METHOD

        Conduct the initial append of some obspy.Trace-like object to this MLTraceBuffer
        object, scraping essential header data, and populating the MLTraceBuffer.data and .fold
        attributes to the max_length definied when initializing the the MLTraceBuffer object

        :: INPUTS ::
        :param other: [obspy.Trace] or [ayahos.MLTrace] like object
                        data and metadata to append to this initialized MLTraceBuffer object
        :param overflow_ref: [str] in the event that the appended trace has more data than
                        max_length allows, this specifies which endpoint of `other` is used
                        as a fixed referece (i.e., the end that is not truncated)
                        Supported arguments
                            'starttime' - use other.stats.starttime as the fixed reference
                            'endtime' - use other.stats.endtime as the fixed reference
        
        :: OUTPUT ::
        :return self: [ayahos.data.mltracebuffer.MLTraceBuffer] enable cascading
        """
        # Extra safety catch that this is a first append
        if not self._has_data:
            # Scrape SNCL, Model, Weight, sampling_rate, and starttime from `other`
            for _k in ['station','location','network','channel','model','weight','sampling_rate','calib']:
                if _k in other.stats.keys():
                    self.stats.update({_k:other.stats[_k]})
            # Inflate buffer to occupy memory allocation
            max_data = round(self.max_length*self.stats.sampling_rate)
            # Initialize as a masked data array...
            self.data = np.ma.MaskedArray(np.full(max_data, fill_value=np.nan),
                                        mask=np.full(max_data, fill_value=True))
            # ... and a 0-fold array
            self.fold = np.full(max_data, fill_value=0)

            # If appended data is smaller than the buffer size
            if other.stats.npts < max_data:
                # Assign starttime
                self.stats.starttime = other.stats.starttime
                # Bring in other's data and unmask those values
                if not isinstance(other.data, np.ma.MaskedArray):
                    self.data.data[:other.stats.npts] = other.data
                    self.data.mask[:other.stats.npts] = False
                else:
                    self.data.data[:other.stats.npts] = other.data.data
                    self.data.mask[:other.stats.npts] = other.data.mask
                # If other has a fold attribute, use it's fold
                if 'fold' in dir(other):
                    self.fold[:other.stats.npts] = other.fold
                # Otherwise populate as a 1-fold segment
                else:
                    self.fold[:other.stats.npts] = np.ones(shape=other.data.shape)

            # If data fit the buffer perfectly
            elif other.stats.npts == max_data:
                # Assign starttime
                self.stats.starttime = other.stats.starttime
                self.data = other.data
                if 'fold' in dir(other):
                    self.fold = other.fold
                else:
                    self.fold = np.ones(shape=other.data.shape, dtype=self.data.dtype)
                
            # if there is overflow
            else:
                # If referencing to the endtime
                if overflow_ref == 'endtime':
                    self.stats.starttime = other.stats.endtime - self.max_length
                    self.data = other.trim(starttime=self.stats.starttime).data
                    if 'fold' in dir(other):
                        self.fold = other.fold
                    else:
                        self.fold = np.ones(shape=self.data.shape, dtype=self.data.dtype)
                elif overflow_ref == 'starttime':
                    self.stats.starttime = other.stats.starttime
                    self.data = other.trim(endtime = self.stats.endtime).data
                    if 'fold' in dir(other):
                        self.fold = other.fold
                    else:
                        self.fold = np.ones(shape=self.data.shape, dtype=self.data.dtype)
                else:
                    raise ValueError(f'overflow_ref "{overflow_ref}" not supported. See documentation')
            self._has_data = True
        else:
           raise AttributeError('This MLTraceBuffer already contains data - canceling _first_append()')
        return self


    def _slide_buffer(self, reference_datetime, reference_type='endtime'):
        """
        PRIVATE METHOD

        Slide the contents of this MLTraceBuffer's .data and .fold attributes
        relative to a specified reference datetime and a specified current 
        endpoint of the MLTraceBuffer. Contents of .data and .fold that move
        out of the bounds of the new time window are discarded and emptied
        spaces (i.e. those shifted out of) are filled with masked values in
        self.data and 0-values in self.fold.

        :: INPUTS ::
        :param reference_datetime: [obspy.UTCDateTime] reference datetime object
                                    for shifting relative to the specified 
                                    `reference_type`
        :param reference_type: [str] initial time bound to use from this MLTraceBuffer
                                    for determining the shift
                                    Supported Values:
                                    'starttime'
                                    'endtime'
        
        :: OUTPUT ::
        :return self: [ayahos.data.mltracebuffer.MLTraceBuffer] enables cascading
        """

        if reference_type.lower() in ['end','endtime','t1']:
            dt = reference_datetime - self.stats.endtime

        elif reference_type.lower() in ['start','starttime','t0']:
            dt = reference_datetime  - self.stats.starttime
        else:
            raise ValueError(f'reference_type "{reference_type}" not supported.')
        # Calculate the floating point shift rightwards
        float_shift = -dt*self.stats.sampling_rate
        # Get the closest integer rightward shift in samples
        nshift = round(float_shift)
        # If some shift is being applied
        if nshift != 0:
            # Apply positive rightward shift to data and fold
            self.data = np.roll(self.data, nshift)
            self.fold = np.roll(self.fold, nshift)
            # Convert data to masked array if it isn't already a masked array
            if not isinstance(self.data, np.ma.MaskedArray):
                self.data = np.ma.MaskedArray(
                    data=self.data,
                    mask=np.full(self.data.shape, fill_value=False),
                    fill_value=None)
            # If the data are a masked array, but it has a bool for mask, expand mask to bool vector
            elif self.data.mask.shape != self.data.shape:
                self.data.mask = np.full(self.data.shape, fill_value=self.data.mask)

            # Mask data in emptied spots and set fold to 0 in those spots
            if nshift > 0:
                self.data.mask[:nshift] = True
                self.fold[:nshift] = 0
            elif nshift < 0:
                self.data.mask[nshift:] = True
                self.fold[nshift:] = 0

            # Update starttime (which propagates to update endtime)
            self.stats.starttime -= dt
        return self

    def to_mltrace(self):
        """
        Pass the contents of the .data, .stats, and .fold attributes 
        of this MLTraceBuffer to a new MLTrace object
        """
        self = MLTrace(data=self.data, header=self.stats, fold=self.fold)
        return self

## GRAVEYARD ##

    # def enforce_max_length(self, reference='endtime'):
    #     """
    #     Enforce the maximum length of the buffer using information
    #     """
    #     sr = self.stats.sampling_rate
    #     max_samp = int(self.max_length * sr + 0.5)
    #     if reference == 'endtime':
    #         te = self.stats.endtime
    #         ts = te - max_samp/sr
    #     elif reference == 'starttime':
    #         ts = self.stats.starttime
    #         te = ts + max_samp/sr
    #     self.trim(starttime=ts, endtime=te, pad=True, fill_value=None, nearest_sample=True)