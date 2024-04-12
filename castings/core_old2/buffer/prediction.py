"""
:module: wyrm.core.buffer.prediction
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose: 
    This module provides the class definition for PredictionBuffer, which
    houses predicted values and metadata associated with continuous predicitons
    for a single seismic instrument. Initialization of a Prediction Buffer takes
    a minimal amout of parameteric inputs, and populates many of its attributes once
    a PredictionWindow object is appended to the buffer.

    Although this class is based on the obspy.realtime.RtTrace class, it notably
    diverges in the fact that it hosts a 2-D array (stack) to house multiple
    prediction time series and a 1-D array (fold) that tracks how many predicted
    values have been appended at a given time indexed point to facilitate stacking
    via averaging values. Fold also provides an idea of the data availability/density
    in the buffer and can be used to guide subsampling of prediction time series for
    subsequent triggering analyses.
"""
import os
import numpy as np
from obspy import Trace, Stream, read
from wyrm.core.window.prediction import PredictionWindow
import wyrm.util.compatability as wcc
from wyrm.util.semblance import shift_trim
from copy import deepcopy

class PredictionBuffer(object):
    """
    Class definition for a PredictionBuffer object. 
    
    This class is structured loosely after an obspy.realtime.RtTrace, hosting
    a 2-D numpy.ndarray in the self.stack attribute, rather than a data vector. 
    The purpose of this buffer is to house l-labeled, t-time-sampled, windowed
    predictions from machine learning models and provide methods for stacking
    methods (i.e. overlap handling) used for these applications (i.e., max value and
    average value) that differ from the overlap handing provided in obspy (interpolation
    or gap generation - see obspy.core.trace.Trace.__add__).

    Like the obspy.realtime.RtTrace object, all data, and most metadata are populated
    in a PredictionBuffer object (pbuff, for short) by the first append of a
    pwind object (wyrm.core.window.prediction.PredicitonWindow) to the buffer.

    
    In this initial version:
     1) metadata are housed in distinct attributes, rather than an Attribute Dictionary 
        as is the case with obspy Trace-like objects (this may change in a future version)
     2) non-future appends (i.e., appends that back-fill data) have conditions relative to
        buffer size and data timing for if an append is viable. 
        APPEND RULES:
        a) If pwind is later than the contents of pbuff (i.e. future append) - unconditional append
        b) If pwind is earlier than the contents of pbuff - pbuff must have enough space to fit
            the data in pwind without truncating data at the future (right) end of the buffer
        c) If pwind is within the contents of pbuff - pbuff cannot have a data fold greater than
            1 for all samples wher pwind overlaps pbuff.

    """
    
    def __init__(self, max_length=15000, stacking_method='max'):
        """
        Initialize a PredictionBuffer object that has predicted value
        arrays added to it throught the append method like TraceBuffer

        Unlike TraceBuffer, this class leverages the uniform sampling
        and timing of incoming values from assorted labels (channels)
        to accelerate stacking/buffer update operations via numpy's ufunc's

        :: INPUTS ::
        :param max_length: [int] maximum number of samples the buffer can contain
                        Once data are appended, this corresponds to the buffered
                        data "stack" 1-axis (or j-axis)
        :param stacking_method: [str] stacking method to apply to overlapping data
                        'max': maximum value at the jth overlapping sample
                        'avg': mean value at the jth overlapping sample
        
        :: POPULATED ATTRIBUTES ::
        :attr max_length: positive integer, see param max_length
        :attr stacking_method: 'max', or 'avg', see stacking_method
        :attr fold: (max_length, ) numpy.ndarray - keeps track of the number
                        of non-blinded samplesn stacked a particular time-point
                        in this buffer. dtype = numpy.float32
        :attr _has_data: PRIVATE - [bool] - flag indicating if data have been
                        appended to this pbuff via the pbuff.append() method
                        Default = False
        """
        # max_length compat. check
        self.max_length = wcc.bounded_intlike(
            max_length,
            name='max_length',
            minimum=10000,
            maximum=None,
            inclusive=True
        )
        # stacking_method compat. check
        if stacking_method not in ['max', 'avg']:
            raise ValueError('stacking method must be either "max" or "avg"')
        else:
            self.stacking_method = stacking_method
        # Create fold vector
        self.fold = np.zeros(self.max_length, dtype=np.float32)
        # Set _has_data flag
        self._has_data = False
    
    def validate_pwind(self, pwind):
        """
        Validate that metadata attributes in this PredictionBuffer and a candidate
        PredictionWindow. In the case where (meta) data have not been appended
        to this pbuff, attributes are assigned from the input pwind object
        (see :: ATTRIBUTES ::)

        :: INPUT ::
        :param pwind: [wyrm.core.window.prediction.PredictionWindow]
                    Candidate prediction window object
        :: OUTPUT ::
        :return status: [bool] is pwind compatable with this PredictionBuffer?

        :: ATTRIBUTES :: - only updated if self._has_data = False
        :attr id: [str] - instrument ID
        :attr t0: [float] - timestamp of first sample in pwind (seconds)
        :attr samprate: [float] - sampling rate of data in samples per second
        :attr model_name: [str] - machine learning model architecture name associated
                                with this pwind/pbuff
        :attr labels: [list] of [str] - names of data/prediction labels (axis-0 in self.stack)
        :attr stack: [numpy.ndarray] - (#labels, in_samples) array housing appended/stacked
                    data
        :attr blinding: [2-tuple] of [int] - number of blinding samples at either end
                     of a pwind
        """
        attr_list = ['id','samprate','model_name','weight_name','labels','blinding']
        if self._has_data:
            bool_list = []
            for _attr in attr_list:
                bool_list.append(eval(f'self.{_attr} == pwind.{_attr}'))
            if all(bool_list):
                status = True
            else:
                status = False
        else:
            # Scrape metadata
            self.id = pwind.id
            self.t0 = pwind.t0
            self.samprate = pwind.samprate
            self.model_name = pwind.model_name
            self.weight_name = pwind.weight_name
            self.labels = pwind.labels
            self.blinding = pwind.blinding
            # Populate stack
            self.stack = np.zeros(shape=(pwind._nl, self.max_length), dtype=np.float32)
            status = True
        return status

    def append(self, pwind, include_blinding=False):
        """
        Append a PredictionWindow object to this PredictionBuffer if it passes compatability
        checks and shift scenario checks that:
            1) unconditionally allow appends of data that add to the right end of the buffer
                i.e., appending future data
            2) allows internal appends if there are some samples that have fold > 1 in the
                merge space
                i.e., appending to fill gaps
            3) allows appends to left end of buffer if the corresponding sample shifts
                would not truncate right-end samples
                i.e. appending past data

        :: INPUTS ::
        :param pwind: [wyrm.core.window.prediction.PredicitonWindow] pwind object with
                    compatable data/metadata with this pbuff
        :param include_blinding: [bool] should blinded samples be included when calculating
                    the candidate shift/stack operation?
        
        :: OUTPUT ::
        :return self: [wyrm.core.buffer.prediction.PredictionBuffer] enable cascading
        """
        # pwind compat. check
        if not isinstance(pwind, PredictionWindow):
            raise TypeError('pwind must be type wyrm.core.window.prediction.PredictionWindow')
        # pwind validation/metadata scrape if first append
        elif not self.validate_pwind(pwind):
            raise BufferError('pwind and this PredictionBuffer are incompatable')
        
        # include_blinding compat. check
        if not isinstance(include_blinding, bool):
            raise TypeError('include_blinding must be type bool')
        
        # Get stacking instructions
        indices = self.get_stacking_indices(pwind, include_blinding=include_blinding)
        # Initial append - unconditional
        if not self._has_data:
            self._shift_and_stack(pwind, indices)
            self._has_data = True
        # Future append - unconditional
        elif indices['npts_right'] < 0:
            self._shift_and_stack(pwind, include_blinding=include_blinding)
        # Past append - conditional
        elif indices['npts_right'] > 0:
            # Raise error if proposed append would clip the most current predictions
            if any(self.fold[-indices['npts_right']:] > 0):
                raise BufferError('Proposed append would trim off most current predictions in this buffer - canceling append')
            else:
                self._shift_and_stack(pwind, indices)
        # Internal append - conditional
        else: #if indices['npts_right'] == 0:
            # If all sample points have had more than one append, cancel
            if all(self.fold[indices['i0_s']:indices['i1_s']] > 1):
                raise BufferError('Proposed append would strictly stack on samples that already have 2+ predictions - canceling append')
            else:
                self._shift_and_stack(pwind, indices)
        return self
    
    def get_stacking_indices(self, pwind, include_blinding=True):
        """
        Fetch the time-sample shift necessary to fit pwind into this
        pbuff object (npts_right), the indices of data to include
        from this pwind object [i0_p:i1_p], and the indices that the
        data will be inserted into after the time-sample shift has
        been applied to self.stack/.fold [i0_s:i1_s].

        :: INPUTS ::
        :param pwind: [wyrm.core.window.prediction.PredictionWindow] pwind object
                        to append to this pbuff.
        :param include_blinding: [bool] should the blinded samples be included
                        when calculating the shift and indices for this proposed append?

        :: OUTPUT ::
        :return indices: [dict] dictionary containing calculated values for
                            npts_right, i0_p, i1_p, i0_s, and i1_s as desecribed
                            above.
        """
        # Set prediction window sampling indices
        if include_blinding:
            indices = {'i0_p': 0, 'i1_p': None}
        else:
            indices = {'i0_p': self.blinding_samples[0],
                       'i1_p': -self.blinding_samples[1]}
        # If this is an initial append
        if not self._has_data:
            indices.update({'npts_right': 0, 'i0_s': None})
            if include_blinding:
                indices.update({'i1_s': self.window_samples})
            else:
                indices.update({'i1_s': self.window_samples - self.blinding_samples[0] - self.blinding_samples[1]})
        # If this is for a subsequen tappend
        else:
            dt = pwind.t0 - self.t0
            i0_init = dt*self.samprate
            # Sanity check that location is integer-valued
            if int(i0_init) != i0_init:
                raise ValueError('proposed new data samples are misaligned with integer sample time-indexing in this PredBuff')
            # Otherwise, ensure i0 is type int
            else:
                i0_init = int(i0_init)
            # Get index of last sample in candidate prediction window
            i1_init = i0_init + self.window_samples
            # If blinding samples are removed, adjust the indices
            if not include_blinding:
                i0_init += self.blinding_samples[0]
                i1_init -= self.blinding_samples[1]
                di = self.window_samples - self.blinding_samples[0] - self.blinding_samples[1]
            else:
                di = self.window_samples

            # Handle data being appended occurs before the current buffered data
            if i0_init < 0:
                # Instruct shift to place the start of pred at the start of the buffer
                indices.update({'npts_right': -i0_init,
                                     'i0_s': None,
                                     'i1_s': di})

            # If the end of pred would be after the end of the current buffer timing
            elif i1_init > self.buff_samples:
                # Instruct shift to place the end of pred at the end of the buffer
                indices.update({'npts_right': self.buff_samples - i1_init,
                                     'i0_s': -di,
                                     'i1_s': None})
            # If pred entirely fits into the current bounds of buffer timing
            else:
                # Instruct no shift and provide in-place indices for stacking pred into buffer
                indices.update({'npts_right': 0,
                                     'i0_s': i0_init,
                                     'i1_s': i1_init})

        return indices


    def _shift_and_stack(self, pwind, indices):
        """
        Apply specified npts_right shift to self.stack and self.fold and
        then stack in pwind.data at specified indices with specified
        self.stack_method, and update self.fold. Internal shifting routine
        is wyrm.util.stacking.shift_trim()
        
        :: INPUTS ::
        :param pwind: [wyrm.core.window.prediction.PredictionWindow] 
                        validated prediction window object to append to this prediction buffer
        :param indices: [dict] - stacking index instructions from self.get_stacking_indices()
        
        :: OUTPUT ::
        :return self: [wyrm.core.buffer.prediction.PredictionBuffer] to enable cascading
        """

        # Shift stack along 1-axis
        self.stack = shift_trim(
            self.stack,
            indices['npts_right'],
            axis=1,
            fill_value=0.,
            dtype=self.stack.dtype)
        
        # Shift fold along 0-axis
        self.fold = shift_trim(
            self.fold,
            indices['npts_right'],
            axis=0,
            fill_value=0.,
            dtype=self.fold.dtype)
        
        # # ufunc-facilitated stacking # #
        # Construct in-place prediction slice array
        pred = np.zeros(self.stack.shape, dtype=self.stack.dtype)
        pred[:, indices['i0_s']:indices['i1_s']] = pwind.data[:, indices['i0_p']:indices['i1_p']]
        # Construct in-place fold update array
        nfold = np.zeros(shape=self.fold.shape, dtype=self.stack.dtype)
        nfold[indices['i0_s']:indices['i1_s']] += 1
        # Use fmax to update
        if self.stack_method == 'max':
            # Get max value for each overlapping sample
            np.fmax(self.stack, pred, out=self.stack); #<- Run quiet
            # Update fold
            np.add(self.fold, nfold, out=self.fold); #<- Run quiet
        elif self.stack_method == 'avg':
            # Add fold-scaled stack/prediction arrays
            np.add(self.stack*self.fold, pred*nfold, out=self.stack); #<- Run quiet
            # Update fold
            np.add(self.fold, nfold, out=self.fold); #<- Run quiet
            # Normalize by new fold to remove initial fold-rescaling
            np.divide(self.stack, self.fold, out=self.stack, where=self.fold > 0); #<- Run quiet
        
        # If a shift was applied, update t0 <- NOTE: this was missing in version 1!
        if indices['npts_right'] != 0:
            self.t0 -= indices['npts_right']/self.samprate
        
    # I/O AND DUPLICATION METHODS #

    def copy(self):
        """
        Return a deepcopy of this PredictionBuffer
        """
        return deepcopy(self)
        
    def to_stream(self, min_fold=1, fill_value=None):
        """
        Create a representation of self.stack as a set of labeled
        obspy.core.trace.Trace objects contained in an obspy.core.stream.Stream
        object. Traces are the full length of self.stack[_i, :] and masked using
        a threshold value for self.fold.

        Data labels are formatted as follows
        for _l in self.labels
            NLSC = self.id + _l[0].upper()
        
        i.e., the last character in the NSLC station ID code is the first, capitalized
              letter of the data label. 
        e.g., for UW.GNW.--.BHZ, 
            self.id is generally trucated by the RingWyrm or DiskWyrm data ingestion
            to UW.GNW.--.BH
            UW.GNW.--.BH with P-onset prediction becomes -> UW.GNW.--.BHP

        The rationale of this transform is to allow user-friendly interface with the 
        ObsPy API for tasks such as thresholded detection and I/O in standard seismic formats.

        :: INPUTS ::
        :param min_fold: [int-like] minimum fold required for a sample in stack to be unmasked
        :param fill_value: compatable value for numpy.ma.masked_array()'s fill_value kwarg

        :: OUTPUT ::
        :return st: [obspy.core.stream.Stream]
        """
        # Create stream
        st = Stream()
        # Compose boolean mask
        mask = self.fold < min_fold
        # Use default fill value
        if fill_value is None:
            fill_value = self.fill_value
        # Compose generic header
        n,s,l,bi = self.id.split('.')
        # If the ID happens to be using the full 3-character SEED code, truncate
        if len(bi) == 3:
            bi = bi[:2]
        header = {'network': n,
                  'station': s,
                  'location': l,
                  'starttime': self.t0,
                  'sampling_rate': self.samprate}
        # Construct specific traces with label names
        for _i, _l in enumerate(self.labels):
            header.update({'channel':f'{bi}{_l[0].upper()}'})
            _tr = Trace(data=np.ma.masked_array(data=self.stack[_i,:],
                                                mask=mask, fill_value=fill_value),
                        header=header)
            st.append(_tr)
        
        # Construct fold trace
        header.update({'channel': f'{bi}f'})
        _tr = Trace(data=self.fold, header=header)
        st.append(_tr)
        return st
    

    def to_mseed(self, save_dir='.', min_fold=1, fill_value=None):
        """
        EXAMPLE WORKFLOW

        This serves as an example wrapper around to_stream() for 
        saving outputs to miniSEED format as individual traces

        :: INPUTS ::
        :param save_dir: [str] - path to directory in which to save
                        the output miniSEED files (exclude trailing \ or /)
        :param min_fold: [int] - see self.to_stream()
        :param fill_value: - see self.to_stream()
        """

        st = self.to_stream(min_fold=min_fold, fill_value=fill_value)
        labels = ''
        for _l in self.labels:
            labels += f'{_l}_'
        labels = labels[:-1]
        of_str = '{model}_{weight}_{id}_{t0:.6f}_{labels}.mseed'.format(
            model=self.model_name,
            weight=self.weight_name,
            id=self.id,
            t0=self.t0,
            labels=labels
        )
        out_fp = os.path.join(save_dir, of_str)
        st.write(out_fp, fmt='MSEED')

    def from_stream(self, st, model_name=None, weight_name=None, tol=1e-5):
        """
        Populate a PredictionBuffer object from a stream object 
        """
        # Only run "append" from stream if no data are present
        if self._has_data:
            raise BufferError('This PredictionBuffer already has data - cannot populate from Stream')

        # Convert labels into channel names
        channels = [_tr.stats.channel for _tr in st]
        labels = [_tr.stats.channel[-1] for _tr in st]
        
        # Run cross-checks on data traces and fold trace
        for _i, tr_i in enumerate(st):
            for _j, tr_j in enumerate(st):
                if _i < _j:
                    if abs(tr_i.stats.starttime - tr_j.stats.starttime) > tol:
                        raise AttributeError(f'starttime mismatch: {tr_i.id} vs {tr_j.id}')
                    if tr_i.stats.npts != tr_j.stats.npts:
                        raise AttributeError(f'npts mismatch: {tr_i.id} vs {tr_j.id}')
                    if tr_i.stats.sampling_rate != tr_j.stats.sampling_rate:
                        raise AttributeError(f'sampling_rate mismatch: {tr_i.id} vs {tr_j.id}')
                    if tr_i.id[-1] != tr_j.id[:-1]:
                        raise AttributeError(f'instrument code (id[:-1]) mismatch: {tr_i.id} vs {tr_j.id}')
                    
        # populate attributes
        self.t0 = st[0].stats.starttime.timestamp
        self.labels = labels
        self.id = st[0].id[:-1]
        self.weight_name = weight_name
        self.model_name = model_name
        self._has_data = True

        # populate stack
        self.stack = np.zeros(shape=(len(labels), st[0].stats.npts), dtype=np.float32)
        # get data
        _i = 0
        for _c in (channels): 
            _tr = st.select(channel=_c)[0]
            if _c[-1] != 'f':
                self.stack[_i, :] = _tr.data
                _i += 1
            else:
                self.fold = _tr.data
        
        return self

    def from_mseed(self, infile ,tol=1e-5):
        """
        Populate this PredictionBuffer from a version saved as a MSEED file using
        the to_mseed() method above. This can only be executed on a pbuff object that
        has not had data appended to it.

        Wraps the PredictionBuffer.from_stream() method.

        :: INPUTS ::
        :param infile: [str] path and file name string - formatting must conform to the 
                     format set by PredictionBuffer.to_mseed()
        :param tol: [float] mismatch tolerance between trace starttimes in seconds

        :: OUTPUT ::
        :return self: [wyrm.core.buffer.prediction.PredictionBuffer] enable cascading
        """
        # Only run "append" from stream if no data are present
        if self._has_data:
            raise BufferError('This PredictionBuffer already has data - cannot populate from mSEED')

        # Parse input file name for metadata
        # Strip filename from infile
        _, file = os.path.split(infile)
        # Remove extension
        file = os.path.splitext(file)[0]
        # Split by _ delimiter
        fparts = file.split('_')
        if len(fparts) != 4:
            raise SyntaxError(f'infile filename {file} does not conform with the format "model_weight_id_t0_l1-l2-l3"')
        # Alias metadata from file name
        model = fparts[0]
        weight = fparts[1]
        # id = fparts[2]
        # ch = id.split('.')[-1]
        # t0 = float(fparts[3])
        # labels = fparts[4].split('-')
        
        # Load data
        st = read(infile).merge()
        # Run data ingestion
        self.from_stream(st, model_name=model, weight_name=weight, tol=tol)
        

        return self