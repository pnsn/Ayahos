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
import numpy as np
from wyrm.core.window.prediction import PredictionWindow
import wyrm.util.compatability as wcc
from wyrm.util.stacking import shift_trim
from copy import deepcopy

class PredictionBuffer(object):
    
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
        PredictionWindow. In the case where 

        :: INPUT ::
        :param pwind: [wyrm.core.window.prediction.PredictionWindow]
                    Candidate prediction window object
        :: OUTPUT ::
        :param
        """
        attr_list = ['id','samprate','model_name','weight_name','labels','blinding']
        if self._has_data:
            bool_list = []
            for _attr in attr_list:
                bool_list.append(eval(f'self.{_attr} == pwind.{_attr}'))
            if all(bool_list):
                return True
            else:
                return False
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
            return True

    def append(self, pwind, include_blinding=False):
        # pwind compat. check
        if not isinstance(pwind, PredictionWindow):
            raise TypeError('pwind must be type wyrm.core.window.prediction.PredictionWindow')
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

        if include_blinding:
            indices = {'i0_p': 0, 'i1_p': None}
        else:
            indices = {'i0_p': self.blinding_samples[0],
                       'i1_p': -self.blinding_samples[1]}
        
        if not self._has_data:
            indices.update({'npts_right': 0, 'i0_s': None})
            if include_blinding:
                indices.update({'i1_s': self.window_samples})
            else:
                indices.update({'i1_s': self.window_samples - self.blinding_samples[0] - self.blinding_samples[1]})
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
        stack_method
        
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
        




class PredictionBuffer(object):

    def __init__(self,
                 model=sbm.EQTransformer(),
                 weight_name='pnw',
                 buff_samples=15000,
                 stack_method='max',
                 blinding_samples=(500,500)):
        """
        Initialize a Prediction Buffer v2 (PredBuff2) object

        :: INPUTS ::
        :param model: [seisbench.models.WaveformModel] child-class of this baseclass
                        used to generate predictions. 
        :param weight_name: [str] name of pretrained weights used for prediction. Must
                        be in the list of pretrained model names
        :param buff_samples: [int] length of this buffer in samples. Must be between
                        2 and 100x model.in_samples
        :param stack_method: [str] stacking method for overlapping predicted samples
                        Supported: 'avg', 'max'
        :param blinding_samples: [2-tuple] number of samples at the front and end of
                        an appended PredArray object to suppres (sets to 0)
    
        """
        # model compat. check
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError('intput model must be type seisbench.models.WaveformModel')
        else:
            self.model_name = model.name
            self.labels = {_l: _i for _i, _l in enumerate(model.labels)}
            self.window_samples = model.in_samples

        # weight_name compat. check
        pt_list = model.list_pretrained()
        if isinstance(weight_name, str):
            if weight_name in pt_list:
                self.weight_name 
            else:
                raise ValueError
        else:
            raise TypeError
        
        # buff_samples compat. check
        self.buff_samples = icc.bounded_intlike(
            buff_samples,
            name='buff_samples',
            minimum = 2*self.window_samples,
            maximum = 100*self.window_samples,
            inclusive=True
        )

        # stack_method compat. check
        if not isinstance(stack_method, str):
            raise TypeError
        elif stack_method not in ['max', 'avg']:
            raise ValueError
        else:
            self.stack_method = stack_method
        

        self.shape = (len(self.labels), self.buff_samples)

        # Blinding Samples compatability check and formatting
        if isinstance(blinding_samples, (list, tuple)):
            if not all(isinstance(_b, (int,float)) for _b in blinding_samples):
                raise TypeError
            if len(blinding_samples) == 2:
                self.blinding_samples = (int(_b) for _b in blinding_samples)
            elif len(blinding_samples) == 1:
                self.blinding_samples = (int(blinding_samples[0]), int(blinding_samples[0]))
            else:
                raise SyntaxError('blinding_samples must be a single number, or a 1-/2-tuple of int-like values')
        elif isinstance(blinding_samples, (int, float)):
            self.blinding_samples = (int(blinding_samples), int(blinding_samples))
        else:
            raise TypeError

        # Initialize data and fold
        self.stack = np.zeros(shape=self.shape, dtype=np.float32)
        self.fold = np.zeros(shape=self.buff_samples, dtype=np.float32)
        # Initialize blinding array 
        self.blinding = np.ones(shape=(self.shape[1], self.shape[2]))
        self.blinding[..., :self.blinding_samples[0]] = 0
        self.blinding[..., -self.blinding_samples[1]:] = 0

        ## PLACEHOLDERS -- POPULATED DURING FIRST APPEND ##
        self.id = None
        self.t0 = None
        self.samprate = None
        
        # Flag for if buffer has had an append or not
        self._has_data = False


    def copy(self):
        return deepcopy(self)
    
    def append(self, predwind, include_blinding=False):
        # Basic compatability checks
        if not isinstance(predwind, PredictionWindow):
            raise TypeError
        if not isinstance(include_blinding, bool):
            raise TypeError
        
        # Handle metadata scrape/crosscheck 
        if self._has_data:
            # Run validation in raise_error mode
            _ = self.validate_prearray_metadata(predwind, raise_error=True)
        else:
            # Run PredArray validation
            self.scrape_predarray_metadata(predwind)
    
        # Get append indexing instructions
        indices = self.get_stacking_indices(predwind, include_blinding=include_blinding)
        
        # Run append with scenario-dependent safety catches.
        # First append - unconditional stacking
        if not self._has_data:
            self._shift_and_stack(predwind, indices)
            self._has_data = True
        # Unconditionally stack future appends
        elif indices['npts_right'] < 0:
            self._shift_and_stack(predwind, indices)
        # Conditionally apply past appends
        elif indices['npts_right'] > 0:
            # Raise error if proposed append would clip the most current predictions
            if any(self.fold[-indices['npts_right']:] > 0):
                raise BufferError('Proposed append would trim off most current predictions in this buffer - canceling append')
            else:
                self._shift_and_stack(predwind, indices)
        else: #if indices['npts_right'] == 0:
            # If all sample points have had more than one append, cancel
            if all(self.fold[indices['i0_s']:indices['i1_s']] > 1):
                raise BufferError('Proposed append would strictly stack on samples that already have 2+ predictions - canceling append')
            else:
                self._shift_and_stack(predwind, indices)
        return self
        
    def scrape_predwind_metadata(self, predwind):
        if not isinstance(predwind, PredictionWindow):
            raise TypeError('predarray must be type wyrm.buffer.prediction.PredArray')
        
        if self._has_data:
            raise RuntimeError('Data have already been appended to this array - will not overwrite metadata')
        
        if self.model_name != predarray.model_name:
            raise ValueError('model_name for this predbuff and input predarray do not match')

        self.id = predarray.id
        self.samprate = predarray.samprate
        self.t0 = predarray.t0

    def validate_prearray_metadata(self, predarray, raise_error=False):
        """
        Check the compatability of metadata between this PredBuff and a
        PredArray object. 
        
        Required matching attributes are:
         @ id - Instrument Code, generally Net.Sta.Loc.BandInst (e.g., UW.GNW..BH)
         @ samprate - sampling rate in samples per second (Hz)
         @ model_name - name of ML model architecture
        
        Attributes checked for type:
         @ t0 - timestamp for 0th sample
         @ weight_name - 
        

        :: INPUTS ::
        :param predarray: [wyrm.buffer.prediction.PredArray]
                            Prediction Array object for which metadata will be compared
        :param raise_error: [bool] 
                            If a mismatch/error is raised internally, raise error externally?
                            True - raise errors
                            False - return False
        
        """
        # Input compatability checks
        if not isinstance(predarray, PredArray):
            raise TypeError('predarray must be type wyrm.buffer.prediction.PredArray')
        
        if not isinstance(raise_error, bool):
            raise TypeError('raise_error must be type bool')
        
        # Check ID match
        if self.id != predarray.id:
            if raise_error:
                raise ValueError('id for this predbuff and predarray do not match')
            else:
                return False    
            
        # Check samprate match
        elif self.samprate != predarray.samprate:
            if raise_error:
                raise ValueError('samprate for this predbuff and predarray do not match')
            else:
                return False
            
        # Check model_name match
        elif self.model_name != predarray.model_name:
            if raise_error: 
                raise ValueError('model_name for this predbuff and predarray do not match')
            else:
                return False
            
        # Check t0 type
        elif not isinstance(predarray.t0, float):
            if raise_error:
                raise TypeError('predarray.t0 must be type float')
            else:
                return False
    
        # Check weight_name type
        elif not isinstance(predarray.weight_name, str):
            if raise_error:
                raise TypeError('predarray.weight_name must be type str')
            else:
                return False
        
        else:
            return True
        
    def get_stacking_indices(self, predarray, include_blinding=True):
        _ = self.validate_prearray_metadata(predarray, raise_error=True)

        if include_blinding:
            indices = {'i0_p': 0, 'i1_p': None}
        else:
            indices = {'i0_p': self.blinding_samples[0],
                       'i1_p': -self.blinding_samples[1]}
        
        if not self._has_data:
            indices.update({'npts_right': 0, 'i0_s': None})
            if include_blinding:
                indices.update({'i1_s': self.window_samples})
            else:
                indices.update({'i1_s': self.window_samples - self.blinding_samples[0] - self.blinding_samples[1]})
        else:
            dt = predarray.t0 - self.t0
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

    def _shift_and_stack(self, predarray, indices):
        """
        Apply specified npts_right shift to self.stack and self.fold and
        then stack in predarray.data at specified indices with specified
        stack_method
        
        :: INPUTS ::
        :param predarray: [wyrm.buffer.prediction.PredArray] prediction array
                            to stack into this PredBuff
        :param indices: [dict] - stacking index instructions from self.get_stacking_indices()
        
        :: OUTPUT ::
        :return self: [wyrm.buffer.prediction.PredBuff] to enable cascading
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
        pred[:, indices['i0_s']:indices['i1_s']] = predarray.data[:, indices['i0_p']:indices['i1_p']]
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

        return self
        
    def to_stream(self, min_fold=1, fill_value=None):
        # Create stream
        st = Stream()
        # Compose boolean mask
        mask = self.fold < min_fold
        # Use default fill value
        if fill_value is None:
            fill_value = self.fill_value
        # Compose generic header
        n,s,l,bi = self.id.split('.')
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
        return st