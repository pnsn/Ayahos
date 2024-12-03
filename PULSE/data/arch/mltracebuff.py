import logging
import numpy as np
from obspy import Trace
from PULSE.data.arch.mltrace import MLTrace

Logger = logging.getLogger(__name__)


###################################################################################
# DISABLE _add_processing_info DECORATOR ##########################################
###################################################################################

@decorator
def _add_processing_info(func, *args, **kwargs):
    """
    Overwrite the _add_processing_info decorator from :class:`~obspy.core.trace.Trace`
    to prevent run-away documentation
    """
    return

class MLTraceBuff(MLTrace):

    def __init__(self,
        bufflen=1,
        add_method=1,
        pre_blinding=None,
        restricted_appends=True,
        reference_edge='end',
        dtype=np.float32,
        **options):
        """
        Initialize an MLTraceBuff object containing no data and default metadata

        :param bufflen: record length in seconds, defaults to 1.
            When the MLTraceBuff has its first data append, the **data** and **fold** vectors
            are allocated in memory for bufflen*sampling_rate samples, regardless of the number of
            samples in the data being appended. This differs from ObsPy's RTTrace, which allows spill-over.
        :type bufflen: float-like, optional
        :param pre_blinding: pre_blinding to apply to traces appended to this MLTraceBuff (including the initial trace), defaults to None.
            Also see :meth:`~PULSE.data.mltrace.MLTrace.apply_blinding`.
        :type pre_blinding: NoneType, positive int, or 2-tuple of positive int, optional
        :param restricted_appends: should restrictions on appends that would add chronologically older data to the buffer
            See :meth:`~PULSE.data.mltracebuff.MLTraceBuff.append` for more details
        :type restricted_appends: bool
        :param reference_edge: in the event of a first-append that over-fills the MLTraceBuff **data** and/or **fold**
            vectors, this specifies which edge of the data being appended is used as reference
            for trimming off data, defaults to "end".
            Supported values:
            - "end" -- 
        :param options: key word argument collector sent to calls of :meth:`~PULSE.data.mltrace.MLTrace.__add__` 
            within :meth:`~PULSE.data.mltracebuff.MLTraceBuff.append`. **options** are saved as the **options** attribute

        :var data: :class:`~numpy.ndarray` vector that holds waveform data values
        :var fold: :class:`~numpy.ndarray` vector that holds data fold values
        :var stats: :class:`~PULSE.data.mltrace.MLTraceStats` object that holds additional header information
        :var _other: dict that holds key-word arguments passed to :meth:`~PULSE.data.mltrace.MLTrace.__add__` each time it is called
        :var _pre_blinding: bool or 2-tuple that holds pre_blinding information passed to :meth:`~PULSE.data.mltrace.MLTrace.apply_pre_blinding`
        :var _restricted: bool flag for **restricted_appends** setting
        :var _has_data: bool flag for if this MLTraceBuff has ever had any data appended to it.

        TODO: Incorporate dtype into __init__
        
        """
        # Initialize as an MLTrace object
        super().__init__()
        # Compatability checks for bufflen
        if isinstance(bufflen, (int, float)):
            if bufflen > 0:
                if bufflen > 1200:
                    Logger.warning('MLTraceBuff bufflen > 1200 sec may take a lot of memory')
                self.bufflen = float(bufflen)
            else:
                raise ValueError('bufflen must be positive')
        else:
            raise TypeError('bufflen must be float-like')
        # pre_blinding compatability
        if pre_blinding is None or not pre_blinding:
            self._pre_blinding = False
        elif isinstance(pre_blinding, (list, tuple)):
            if len(pre_blinding) == 2:
                if all(int(_b) >= 0 for _b in pre_blinding):
                    self._pre_blinding = (int(pre_blinding[0]), int(pre_blinding[1]))
                else:
                    raise ValueError
            elif len(pre_blinding) == 1:
                if int(pre_blinding[0]) >= 0:
                    self._pre_blinding = (int(pre_blinding[0]), int(pre_blinding[0]))
        elif isinstance(pre_blinding, (int, float)):
            if int(pre_blinding) >= 0:
                self._pre_blinding = (int(pre_blinding), int(pre_blinding))
            else:
                raise ValueError
        else:
            raise TypeError
         

        # Compatability check for restrict past appends
        if not isinstance(restricted_appends, bool):
            raise TypeError
        else:
            self._restricted = restricted_appends
        # Capture kwargs for __add__
        self._options = options
        # Initialize _has_data private flag attribute
        self._has_data = False


    def append(self, other):
        """Core method for adding data to this MLTraceBuff object. This method applies a series of additional
        pre-append cross checks on new data being introduced to the **mltracebuff** that include timing and
        trace-size checks. It enforces the **restricted_appends** rule set when the **mltracebuff** was
        initialized and applies pre_blinding to incoming traces if this option is used. 

        
        context and buffer-size
        
        This method wraps the :meth:`~PULSE.data.mltrace.MLTrace.__add__` special method and
        provides some additional checks relevant to semi-sequential data streaming and buffering
        described in the following scenarios.

        .. rubric:: Append Scenarios

        * First Append
            If no data have been appended to this MLTraceBuff.
            Uses the :meth:`~PULSE.data.mltracebuff.MLTraceBuff._first_append` method to 
        

        * Internal Append 
            If **other** is fully contained within the current starttime and endtime of **mltracebuff**
            Uses :meth:`~PULSE.data.mltrace.MLTrace.__add__` to add **other** to **mltracebuff**.

        * Near Future Append 
            If some contents of the buffer and **other** coexist within the current **bufflen** window. 
            
            Data in the buffer are shifted to match **mltracebuff.stats.endtime** to **other.stats.endtime** and **other**
            is added using :meth:`~PULSE.data.mltrace.MLTrace.__add__`.

        * Far Future Append
            If **other.stats.starttime** is later than **mltracebuff.stats.endtime**.
            Data in **mltracebuff** are discarded and **other** is appended with :meth:`~PULSE.data.mltracebuff.MLTraceBuff._first_append`

        * Near Past Append **with restrictions** 
            **other** is trimmed to fit the available space (i.e., samples with 0 fold) in **mltracebuff**, if any exists.

        * Near Past Append **without restrictions** - **mltracebuff** contents are slid and trimmed as
            described in the Near Future Append scenario to accommodate the shape and timing of data
            in **other**

        * Far Past Append **with restrictions** - **other** is not appended to **mltracebuff**

        * Far Past Append **without restrictions** - **other** is apppended as described in the
            Far Future Append scenario.

        :param other: Trace-like object to append to this **mltracebuff**
        :type other: obspy.core.trace.Trace 
            also see :meth:`~PULSE.data.mltrace.MLTrace.__add__`


        :param kwargs: [kwargs] key-word arguments passed to MLTrace.__add__()
                    NOTE: any kwarg in kwargs that have a matching key to
                        self._options will be superceded by inputs to **kwargs
                        here.

        :: OUTPUT ::
        :return self: [ewflow.data.mltracebuff.MLTraceBuff] enable cascading
        """
        if self._has_data:
            if self.id != other.id:
                raise ValueError(f'trace ID\'s do not match {self.id} vs {other.id}')

        if isinstance(other, Trace):
            # If other is a trace, but not an MLTrace, convert
            if not isinstance(other, MLTrace):
                other = MLTrace(other)
        else:
            raise TypeError('input other must be type obspy.core.trace.Trace or a child-class thereof')
        
        # Apply pre_blinding (if specified) to incoming trace
        if self._pre_blinding:
            other.apply_pre_blinding(pre_blinding=self._pre_blinding)

        # If this is a first append
        if not self._has_data:
            self._first_append(other)
        # If this is a subsequent append 
        else:
            # (FUTURE APPEND) If other ends at or after self (FUTURE APPEND)
            if other.stats.endtime >= self.stats.endtime:
                # If other starts within buffer range of self end
                if other.stats.starttime - self.bufflen < self.stats.endtime:
                    # Conduct future append (always unrestricted)
                    # Logger.debug(f'sliding buffer endtime from {self.stats.endtime} to {other.stats.endtime}')
                    self._slide_buffer(other.stats.endtime, reference_type='endtime')
                    # Logger.debug(f'updated endtime {self.stats.endtime}')
                    self.__add__(other, **self._options)
                    # self.enforce_bufflen(reference='endtime')
                # If other starts later that self end + bufflen - big gap
                else:
                    # Run as a first append if id matches
                    if self.id == other.id:
                        self._has_data = False
                        self._first_append(other)

            # (PAST APPEND) If other starts at or before self
            elif other.stats.starttime <= self.stats.starttime:
                # FAR PAST
                if self.stats.starttime - other.stats.endtime >= self.bufflen:
                    # IF restriction in place
                    if self._restricted:
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
                    if self._restricted:
                        # Trim other
                        other.trim(starttime=self.stats.starttime)
                        self.__add__(other, **self._options)
                    # If not restricting past appends - slide buffer and append full other
                    else:
                        self._slide_buffer(other.stats.endtime, reference_type='endtime')
                        self.__add__(other, **self._options)

            # (INNER APPEND)
            else:
                self.__add__(other, **self._options)
                # # TODO: Make sure this is a copy
                # ftr = self.get_fold_trace().trim(starttime=other.stats.starttime, endtime=other.stats.endtime)
                # # If there are any 0-fold data in self that have information from other
                # if (ftr.data == 0 & other.fold >0).any():
                #     self.__add__(other, **kwargs)
                # else:
                #     pass
            
        return self
    
    # @_add_processing_info          
    def _first_append(self, other, reference_edge='endtime'):
        """
        PRIVATE METHOD

        Conduct the initial append of some obspy.Trace-like object to this MLTraceBuff
        object, scraping essential header data, and populating the MLTraceBuff.data and .fold
        attributes to the bufflen definied when initializing the the MLTraceBuff object

        :: INPUTS ::
        :param other: [obspy.Trace] or [ewflow.MLTrace] like object
                        data and metadata to append to this initialized MLTraceBuff object
        :param reference_edge: [str] in the event that the appended trace has more data than
                        bufflen allows, this specifies which endpoint of `other` is used
                        as a fixed referece (i.e., the end that is not truncated)
                        Supported arguments
                            'starttime' - use other.stats.starttime as the fixed reference
                            'endtime' - use other.stats.endtime as the fixed reference
        
        :: OUTPUT ::
        :return self: [ewflow.data.mltracebuff.MLTraceBuff] enable cascading
        """
        # Extra safety catch that this is a first append
        if not self._has_data:
            # Scrape SNCL, Model, Weight, sampling_rate, and starttime from `other`
            for _k in ['station','location','network','channel','model','weight','sampling_rate','calib']:
                if _k in other.stats.keys():
                    self.stats.update({_k:other.stats[_k]})
            # Inflate buffer to occupy memory allocation
            max_data = round(self.bufflen*self.stats.sampling_rate)
            # Initialize as a masked data array...
            self.data = np.ma.MaskedArray(np.full(max_data, fill_value=np.nan),
                                        mask=np.full(max_data, fill_value=True))
            # ... and a 0-fold array
            self.fold = np.full(max_data, fill_value=0)

            # If appended data is smaller than the buffer size
            if other.stats.npts < max_data:
                if reference_edge == 'starttime':
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
                elif reference_edge == 'endtime':
                    self.stats.starttime = other.stats.endtime - self.bufflen
                    if not isinstance(other.data, np.ma.MaskedArray):
                        self.data.data[-other.stats.npts:] = other.data
                        self.data.mask[-other.stats.npts:] = False
                    else:
                        self.data.data[-other.stats.npts:] = other.data.data
                        self.data.mask[-other.stats.npts:] = other.data.mask
                    if 'fold' in dir(other):
                        self.fold[-other.stats.npts:] = other.fold
                    else:
                        self.fold[-other.stats.npts:] = np.ones(shape=other.data.shape)

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
                if reference_edge == 'endtime':
                    self.stats.starttime = other.stats.endtime - self.bufflen
                    self.data = other.trim(starttime=self.stats.starttime).data
                    if 'fold' in dir(other):
                        self.fold = other.fold
                    else:
                        self.fold = np.ones(shape=self.data.shape, dtype=self.data.dtype)
                elif reference_edge == 'starttime':
                    self.stats.starttime = other.stats.starttime
                    self.data = other.trim(endtime = self.stats.endtime).data
                    if 'fold' in dir(other):
                        self.fold = other.fold
                    else:
                        self.fold = np.ones(shape=self.data.shape, dtype=self.data.dtype)
                else:
                    raise ValueError(f'reference_edge "{reference_edge}" not supported. See documentation')
            self._has_data = True
        else:
           raise AttributeError('This MLTraceBuff already contains data - canceling _first_append()')
        return self


    def _slide_buffer(self, reference_datetime, reference_type='endtime'):
        """
        PRIVATE METHOD

        Slide the contents of this MLTraceBuff's .data and .fold attributes
        relative to a specified reference datetime and a specified current 
        endpoint of the MLTraceBuff. Contents of .data and .fold that move
        out of the bounds of the new time window are discarded and emptied
        spaces (i.e. those shifted out of) are filled with masked values in
        self.data and 0-values in self.fold.

        :: INPUTS ::
        :param reference_datetime: [obspy.UTCDateTime] reference datetime object
                                    for shifting relative to the specified 
                                    `reference_type`
        :param reference_type: [str] initial time bound to use from this MLTraceBuff
                                    for determining the shift
                                    Supported Values:
                                    'starttime'
                                    'endtime'
        
        :: OUTPUT ::
        :return self: [ewflow.data.mltracebuff.MLTraceBuff] enables cascading
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
            self.stats.starttime += dt
        return self

    def to_mltrace(self):
        """
        Pass the contents of the .data, .stats, and .fold attributes 
        of this MLTraceBuff to a new MLTrace object
        """
        self = MLTrace(data=self.data, header=self.stats, fold=self.fold)
        return self