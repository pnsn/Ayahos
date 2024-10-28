import warnings
import numpy as np
from obspy.core.trace import Trace
from PULSE.data.foldtrace import FoldTrace

class FTBuff(FoldTrace):
    
    def __init__(
            self,
            maxlen=60.,
            method=3,
            fill_value=None,
            dtype=None):
        super().__init__(dtype=dtype)
        if isinstance(maxlen, (int, float)):
            if 1200 >= maxlen > 0:
                self.maxlen = maxlen
            else:
                raise ValueError('maxlen falls outside bounds of accepted max buffer length: (0, 1200] sec')
        else:
            raise TypeError(f'maxlen of type {type(maxlen)} not supported.')
        if method not in [0, 2, 3]:
            raise ValueError('method {method} not supported. See FoldTrace.__add__ for more information.')
        else:
            self.method = method
        self._empty = True

        self.fill_value = fill_value
    
    def _internal_add_processing_info(self, info):
        # Disable processing tracking for buffers
        return None


    def append(self, other):
        if isinstance(other, FoldTrace):
            pass
        else:
            raise TypeError(f'input "other" must be type PULSE.data.foldtrace.FoldTrace')
        if self._empty:
            self._first_append(other)
        else:
            self._subsequent_append(other)
    

    def _first_append(self, other):
        # Get end-time
        tf = other.stats.endtime
        # Trim/pad to buffer length
        t0 = tf - self.maxlen
        other.trim(starttime=t0, endtime=tf, pad=True, nearest_sample = False, fill_value=self.fill_value)
        # Overwrite stats, data, and fold from other
        self.stats = other.stats
        self.data = other.data
        self.fold = other.fold
        # Clear out processing
        self.stats.processing = []
        # Flip _empty flag
        self._empty = False

    def _subsequent_append(self, other):
        # Safety catch
        if self._empty:
            raise ValueError('Attempting to use _subsequent_append on an empty buffer.')
        # Run validation & pass along error messages
        try:
            self.validate_other(other)
        except ValueError as msg:
            raise ValueError(msg)

        ## CASE 0) FAR FUTURE APPENDS
        if other.stats.starttime > self.stats.endtime:
            


        ## CASE 0) PAST APPENDS
        # If other ends before buffer ends
        if to1 <= ts1:
            # If other ends after buffer starts
            if ts0 < to1:
                # Trim starttime to fit current buffer
                other = other.trim(starttime = ts0, endtime = ts1,
                                pad=False, nearest_sample=False)
                self.__iadd__(other, method=self.method, fill_value=self.fill_value)
                return
            # If other ends before buffer starts
            else:
                # Reject append
                raise UserWarning('Rejected appending other - data predates current buffer scope')
        
        ## CASE 1) INTERNAL APPENDS
        # If other ends 

        ## CASE 2) FUTURE APPENDS
        # If other ends after buffer
        else:
            # Get new buffer starttime
            new_ts0 = ts1 - self.maxlen
            # If the new starttime is outside current buffer
            if new_ts0 > ts1:
                # Re-initialize buffer with _first_append
                self._empty = True
                self._first_append(other)
            # Otherwise 
            else:

            




class FoldTraceBuff(FoldTrace):

    def __init__(
            self,
            bufflen=1.,
            method=3,
            restricted_appends=True,
            ref_edge='endtime',
            dtype=np.float32,
            **options):
        """_summary_

        :param bufflen: length of data to buffer in seconds, defaults to 1.
        :type bufflen: float-like, optional
        :param method: method to pass to :meth:`~PULSE.data.foldtrace.FoldTrace.__add__, defaults to 0.
            Supported values: 0, 2, 3
        :type method: int, optional
        :param restricted_appends: restrict ability to add data that pre-date data in the buffer?
            Defaults to True.
        :type restricted_appends: bool, optional
        :param ref_edge: When , defaults to 'endtime'
        :type ref_edge: str, optional
        :param dtype: _description_, defaults to np.float32
        :type dtype: _type_, optional
        """        
        super().__init__(dtype=dtype)
        if isinstance(bufflen, (int, float)):
            if np.inf > bufflen > 0:
                if bufflen > 1200:
                    warnings.warn("bufflen > 1200 seconds may result in excessive memory use")
                self.bufflen=float(bufflen)
            else:
                raise ValueError('bufflen must be positive valued')
        else:
            raise TypeError('bufflen must be float-like')

        self._options = options
        if method in [0, 2, 3]:
            self._options.update({'method':method})
        else:
            raise ValueError(f'method {method} not supported. Must be 0, 2, or 3. See documentation on FoldTrace.__add__')
        

        if not isinstance(restricted_appends, bool):
            raise TypeError('restricted_appends must be type bool')
        else:
            self._restricted = restricted_appends

        if ref_edge.lower() in ['endtime','starttime']:
            self.ref_edge = ref_edge.lower()
        else:
            raise ValueError(f'ref_edge {ref_edge} not supported.')
        # Flag as not having data
        self._has_data = False

    def append(self, other):
        """Core method for adding data to this FoldTraceBuff object. 
        
        This method applies pre-append cross checks on new data being introduced
        to the FoldTraceBuff object including relative timing and total sample sizes.
        
        It enforces the **restricted_appends** rule if set when the **foldtracebuff** was
        initialized

        context and buffer-size
        
        This method wraps the :meth:`~PULSE.data.foldtrace.FoldTrace.__add__` special method and
        provides some additional checks relevant to semi-sequential data streaming and buffering
        described in the following scenarios.

        .. rubric:: Append Scenarios

        * First Append
            If no data have been appended to this FoldTraceBuff.
            Uses the :meth:`~PULSE.data.foldtracebuff.FoldTraceBuff._first_append` method to 
        
        * Internal Append 
            If **other** is fully contained within the current starttime and endtime of **foldtracebuff**
            Uses :meth:`~PULSE.data.foldtrace.FoldTrace.__add__` to add **other** to **foldtracebuff**.

        * Near Future Append 
            If some contents of the buffer and **other** coexist within the current **bufflen** window. 
            
            Data in the buffer are shifted to match **foldtracebuff.stats.endtime** to **other.stats.endtime** and **other**
            is added using :meth:`~PULSE.data.foldtrace.FoldTrace.__add__`.

        * Far Future Append
            If **other.stats.starttime** is later than **foldtracebuff.stats.endtime**.
            Data in **foldtracebuff** are discarded and **other** is appended with :meth:`~PULSE.data.foldtracebuff.FoldTraceBuff._first_append`

        * Near Past Append **with restrictions** 
            **other** is trimmed to fit the available space (i.e., samples with 0 fold) in **foldtracebuff**, if any exists.

        * Near Past Append **without restrictions** - **foldtracebuff** contents are slid and trimmed as
            described in the Near Future Append scenario to accommodate the shape and timing of data
            in **other**

        * Far Past Append **with restrictions** - **other** is not appended to **foldtracebuff**

        * Far Past Append **without restrictions** - **other** is apppended as described in the
            Far Future Append scenario.

        :param other: Trace-like object to append to this **foldtracebuff**
        :type other: obspy.core.trace.Trace 
            also see :meth:`~PULSE.data.foldtrace.FoldTrace.__add__`


        :param kwargs: [kwargs] key-word arguments passed to FoldTrace.__add__()
                    NOTE: any kwarg in kwargs that have a matching key to
                        self._options will be superceded by inputs to **kwargs
                        here.

        :: OUTPUT ::
        :return self: [ewflow.data.foldtracebuff.FoldTraceBuff] enable cascading
        """
        if self._has_data:
            if self.id != other.id:
                raise ValueError(f'trace ID\'s do not match {self.id} vs {other.id}')

        if isinstance(other, Trace):
            # If other is a trace, but not an FoldTrace, convert
            if not isinstance(other, FoldTrace):
                other = FoldTrace(other)
        else:
            raise TypeError('input other must be type obspy.core.trace.Trace or a child-class thereof')
        
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
    def _first_append(self, other):
        """
        PRIVATE METHOD

        Conduct the initial append of some obspy.Trace-like object to this FoldTraceBuff
        object, scraping essential header data, and populating the FoldTraceBuff.data and .fold
        attributes to the bufflen definied when initializing the the FoldTraceBuff object

        :param other: FoldTrace object to append
        :type other: :class:`~PULSE.data.foldtrace.FoldTrace`
        """
        
        # Extra safety catch that this is a first append
        if not self._has_data:
            # Scrape SNCL, Model, Weight, sampling_rate, and starttime from `other`
            for _k in ['station','location','network','channel','model','weight','sampling_rate','calib']:
                if _k in other.stats.keys():
                    self.stats.update({_k:other.stats[_k]})

            # Inflate buffer to occupy memory allocation
            max_data = round(self.bufflen*self.stats.sampling_rate)
            self.data = np.full(shape=max_data, fill_value=np.nan)
            # Trim incoming to size
            if self.ref_edge == 'starttime':
                other.view_trim(starttime=None,
                                endtime=other.stats.starttime + self.bufflen,
                                pad=True,
                                fill_value=None)
            elif self.ref_edge == 'endtime':
                other.view_trim(starttime=other.stats.endtime - self.bufflen,
                                endtime=None,
                                pad=True,
                                fill_value=None)
            
            
                    
            
            # Initialize as a masked data array...
            self.data = np.ma.MaskedArray(np.full(max_data, fill_value=np.nan),
                                        mask=np.full(max_data, fill_value=True))
            # ... and a 0-fold array
            self.fold = np.full(max_data, fill_value=0)


            # If appended data fits entirely in the specified buffer length
            if other.stats.npts < max_data:
                if self.ref_edge == 'starttime':
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
                elif self.ref_edge == 'endtime':
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
                if self.ref_edge == 'endtime':
                    # Trim excess incoming data from the starttime side (_ltrim)
                    self.stats.starttime = other.stats.endtime - self.bufflen
                    self.data = other.trim(starttime=self.stats.starttime).data
                    if 'fold' in dir(other):
                        self.fold = other.fold
                    else:
                        self.fold = np.ones(shape=self.data.shape, dtype=self.data.dtype)
                elif self.ref_edge == 'starttime':
                    self.stats.starttime = other.stats.starttime
                    self.data = other.trim(endtime = self.stats.endtime).data
                    if 'fold' in dir(other):
                        self.fold = other.fold
                    else:
                        self.fold = np.ones(shape=self.data.shape, dtype=self.data.dtype)

            self._has_data = True
        else:
           raise AttributeError('This FoldTraceBuff already contains data - canceling _first_append()')
        return self


    def _slide_buffer(self, reference_datetime, reference_type='endtime'):
        """
        PRIVATE METHOD

        Slide the contents of this FoldTraceBuff's .data and .fold attributes
        relative to a specified reference datetime and a specified current 
        endpoint of the FoldTraceBuff. Contents of .data and .fold that move
        out of the bounds of the new time window are discarded and emptied
        spaces (i.e. those shifted out of) are filled with masked values in
        self.data and 0-values in self.fold.

        :: INPUTS ::
        :param reference_datetime: [obspy.UTCDateTime] reference datetime object
                                    for shifting relative to the specified 
                                    `reference_type`
        :param reference_type: [str] initial time bound to use from this FoldTraceBuff
                                    for determining the shift
                                    Supported Values:
                                    'starttime'
                                    'endtime'
        
        :: OUTPUT ::
        :return self: [ewflow.data.foldtracebuff.FoldTraceBuff] enables cascading
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