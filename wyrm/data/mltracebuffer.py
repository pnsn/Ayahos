"""
:module: wyrm.data.mltracebuffer
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose: This module provides a class definition for a waveform buffer that has the added
        attributes of the wyrm.data.mltrace.MLTrace class and works in an analogus way
        as the obspy.realtime.rttrace.RTTrace class is to the obspy.core.trace.Trace class.

        Additionally, this class provides some additional safety catch options to guard against
        appending data with spurious timing information and functionalities to use the machine learning
        "stack" style appends used on predicted values.
"""

from obspy import Trace
from wyrm.data.mltrace import MLTrace, _add_processing_info
from wyrm.util.compatability import bounded_floatlike
    
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

        # Compatability checks for default merge(**kwargs)
        if merge_kwargs:
            self.merge_kwargs = merge_kwargs
        else:
            self.merge_kwargs = {}
        # Initialize _has_data private flag attribute
        self._has_data = False

    def __add__(self, other):
        self.append(other)
        return self

    def append(self, other):
        if not isinstance(other, Trace):
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
            if 'fold' in dir(other):
                self.fold = other.fold
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

    def to_mltrace(self):
        self = MLTrace(data=self.data, header=self.stats, fold=self.fold)
        return self
