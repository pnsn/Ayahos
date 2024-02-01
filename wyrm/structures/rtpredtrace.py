from obspy import Trace, UTCDateTime
import numpy as np
from collections import deque
import wyrm.util.input_compatability_checks as icc
import torch
import seisbench.models as sbm

class RtPredBuff(object):
    """
    Realtime Prediction Buffer - this class acts as a buffer for overlapping
    predictions 
    """
    def __init__(self, max_windows=20, stack_method='max'):
        """
        Initialize a Realtime Prediction Stack object

        :: INPUTS ::
        :param max_windows: [int] maximum buffer length in number of windows
        :param blinding: [tuple of int] number of samples to blind
                        on either end of an input prediction
        :param overlap: [int] number of samples each successive 
                        prediction window overlaps by
        :param window: [int] number of samples in a prediction window
        """
        ### Static parameters (once assigned) ###
        # max_windows compatability checks
        # Maximum number of windows allowed in buffer
        self.max_windows = icc.bounded_intlike(
            max_windows,
            name='max_windows',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # stack_method compatability checks
        # Stacking method for overlapping, unblinded prediction values
        if not isinstance(stack_method, str):
            raise TypeError("stack_method must be type str")
        elif stack_method.lower() in ['max','peak','maximum']:
            self.stack_method = 'max'
        elif stack_method.lower() in ['mean','avg','average']:
            self.stack_method = 'avg'
        else:
            raise ValueError(f'stack method "{stack_method}" is unsupported. Supported values: "max", "avg"')
        # Instrument Metadata
        self.inst_code = None
        # ML Model Metadata
        self.model_name = None
        self.weight_name = None
        self.label_codes = None
        self.label_names = None
        # Window Indexing Scalars
        self.window_npts = None
        self.overlap_npts = None
        self.blinding_npts = None        
        # Sampling Rate (float)
        self.samprate = None
        # Placeholder for binary array for blinding samples
        self.blinding_array = None

        ### Dynamic parameters ###
        # NOTE: Reference Index Number / Start Time / array j_index
        # refer to the first sample in the chronologically most-recent
        # window successfully appended to the stack and fold arrays
        # Reference Window Index Number (int)
        self.ref_index = None
        # Reference Window Start Time (UTCDateTime)
        self.ref_time = None
        self.ref_j = None

        # Dynamic data arrays
        self.stack = None
        self.fold = None


        self._has_data = False

    def initial_append(self, pred, meta, model, wgt_name):
        # If data have already been appended - raise error
        if self._has_data:
            raise RuntimeError('This RtPredBuff already contains data - canceling initial_append')
        
        else:
            # Run validation checks on metadata
            vmeta = self._validate_metadata(meta)
            # Run validation checks on input model
            vwgt_name = self._validate_model_info(vmeta, model, wgt_name)
            # Assign attribute values from metadata and model info
            self.inst_code = vmeta['inst_code']
            self.model_name= vmeta['model_name']
            self.weight_name = vwgt_name
            self.label_names = model.labels
            self.label_codes = [_c[0].upper() for _c in model.labels]
            self.window_npts = vmeta['window_npts']
            self.overlap_npts = vmeta['overlap_npts']
            self.blinding_npts = vmeta['blinding_npts']
            self.samprate = vmeta['samprate']
            self.ref_index = vmeta['index']
            self.ref_time = vmeta['starttime']
            self.ref_j = 0

            # Get advance samples
            self.adv_npts = self.window_npts - self.overlap_npts
            # Populate stack and fold arrays
            shp = (len(self.label_codes), 
                   self.window_npts * self.max_windows - self.overlap_npts * (1 - self.max_windows))
            self.stack = np.full(shape=shp, fill_value=np.nan, dtype=np.float32)
            self.fold = np.full(shape=shp, fill_value=np.nan, dtype=np.float32)
            self.blinding_array = np.ones(shape=(len(self.label_codes), self.window_npts))
            self.blinding_array[:self.blinding_npts] = 0
            self.blinding_array[-self.blinding_npts:] = 0

            # Validate prediction
            vpred = self._validate_prediction(pred)

            # Insert prediction into stack
            self.stack[:, self.ref_j:self.ref_j + self.window_npts] = vpred * self.blinding_array
            # Place information into fold
            self.fold[:, self.ref_j:self.ref_j + self.window_npts] += self.blinding_array
            
            # Update _has_data flag
            self._has_data = True
        return self

    def subsequent_append(self, pred, meta):
        vmeta = self._validate_metadata(meta)
        vpred = self._validate_prediction(pred)
        d_index = abs(self.ref_index - vmeta['index'])
        if d_index < self.max_windows:
            candidate_k = self._get_candidate_k(vmeta['index'])
            if candidate_k


    def _validate_metadata(self, meta):
        """
        Validate input metadata dictionary against expected
        keys and value formats as defined in: 
        wyrm.structures.window.get_metadata()
        """
        window_metadata_keys=(
            'inst_code',
            'model_name',
            'samprate',
            "window_npts",
            "overlap_npts",
            "blinding_npts",
            'fill_rule',
            'fill_status',
            'fill_value',
            "starttime",
            'index')
        window_metadata_dtypes=(
            str,
            str,
            float,
            int,
            int,
            int,
            str,
            (bool, str),
            UTCDateTime,
            int
        )
        def_dict = dict(zip(window_metadata_keys, window_metadata_dtypes))
        # Ensure all keys, and only these keys, are present.
        # But don't be picky about order.
        if not isinstance(meta, dict):
            raise TypeError('meta must be type dict')
        # If there are mismatched keys or missing keys, compose
        # detailed error message
        if not def_dict.keys() == meta.keys():
            emsg = 'Misfit keys in "meta":'
            for _k in meta.keys():
                if _k not in window_metadata_keys:
                    emsg += f'\n{_k} (extraneous key in meta)'
            for _k in def_dict.keys():
                if _k not in meta.keys():
                    emsg == f'\n{_k} (missing from meta)'
            raise KeyError(emsg)
        # If there are mismatched datatypes in meta, compose
        # a detailed error message
        if not all(isinstance(meta[_k], def_dict[_k]) for _k in def_dict.keys()):
            emsg = 'meta.values() dtype mismatches:'
            for _k in def_dict.keys():
                if not isinstance(meta[_k], def_dict[_k])
                    emsg += f'\n{_k} is {type(meta[_k])} - {def_dict[_k]} expected'
            raise TypeError(emsg)
        
        # If this is not the RtPredBuff's first rodeo... (i.e., already has data appended)
        if self._has_data:
            # Check that static values match
            self_list = [self.inst_code, self.model_name, self.window_npts, self.overlap_npts, self.blinding_npts, self.samprate]
            meta_list = ['inst_code','model_name','window_npts','overlap_npts','blinding_npts','samprate']
            check_tuples = list(zip(self_list, meta_list))
            for _s, _k in check_tuples:
                if _s != meta[_k]:
                    emsg = f'{_k} in meta ({meta[_k]}) does not match'
                    emsg += f' value recorded in this RtPredBuff ({_s})'
                    raise ValueError(emsg)

            # Get delta time compared to reference time
            dt = self.ref_time - meta['starttime']
            # Get delta samples equivalent
            dnpts = dt * self.samprate
            # Check if the delta npts is an integer scalar
            # of window advance steps
            residual_npts = dnpts % (self.window_npts - self.overlap_npts)
             # if residual is 0
            if residual_npts == 0:
                # do final check that the candidate index is correct
                if self.ref_index - meta['index'] == dnpts // (self.window_npts - self.overlap_npts):
                    


                pass
            else:
                raise IndexError(f'Candidate prediction is misaligned by {residual_npts}')
                # TODO: include an append method where stacking is arbitrary, finding the 
                # nearest sample to append to (computationally cheap) or conduct interpolation
                # to shift sampling points (computationally more expensive)
            # If alignment check is passed
            # Determine the index placement of the data
            

        vmeta = meta.copy()
        return vmeta
    
    def _validate_prediction(self, pred):
        if self.window_npts is None:
            raise AttributeError('Initial metadata have not been populated')
        else:
            shp = (len(self.label_codes), self.window_npts)
            try:
                vpred = pred.copy().reshape(shp)
            except ValueError:
                raise ValueError(f'pred cannot be reshaped into shape {shp}')
        return vpred


        
    def _crosscheck_model_info(self, model, wgt_name):
        # Raise errors if model and metadata have basic information mismatches
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError('model must be a child-class of seisbench.models.WaveformModel')
        elif model.name != self.model_name:
            raise ValueError('model.name and self.model_name have different values')
        elif model.in_samples != meta['window_npts']:
            raise ValueError('model and self.window_npts have different values')
        
        if not isinstance(wgt_name, str):
            raise TypeError()
        
        # If all checks are passed, alias meta as vmeta
        vmeta = meta.copy()
        vmeta.update({'model_wgt'})
        return vmeta
       
    def _validate_prediction(self, pred, vmeta):
        """
        Validate candidate prediction 
        """
    
    def model_stacking_fold(self):
        """
        Create a model of data stacking fold using pre-defined
        window length, blinding samples, and window overlaps.


        :def fold: number of non-blinded samples that share 
                    a given time index from overlapping prediction
                    windows
        
        NOTE:   Depending on the blinding and overlap parameters provided,
                there may be some saw-toothing in the maximum number of unblinded
                samples from consecutive windows, as such we calculate both the 
                maximum data fold (max_fold) attained and the maximum continuous
                data fold (cont_fold) values. 
        
        :: INPUTS ::
        :self: 
        :: OUTPUTS ::
        :return max_fold: [int] maximum fold value attained 
                        by stacking of _burnin_ct + 1 + n
                        consecutive prediction windows
        :return cont_fold: [int] maximum continuously maintained
                        fold value from stacking _burnin_ct + 1 + n
                        consecutive prediction windows

        NOTE: Generally, max_fold - cont_fold \in [0, 1]
        """
         
        # Construct single window weight vector
        self._wind_wgts = np.ones(self.window)
        self._wind_wgts[:self.blinding[0]] = 0
        self._wind_wgts[-self.blinding[1]:] = 0
        # Iterate across two window lengths
        i_, j_ = 0, self.window
        model_stack = np.ones(self.window*(2*self._burnin_stack_ct + 1))
        while i_ < len(model_stack):
            if j_ < len(model_stack):
                model_stack[i_:j_] += self._wind_wgts
            else:
                model_stack[i_:] += self._wind_wgts[:len(model_stack)-i_]
            i_ += (self.window - self.overlap)
            j_ = i_ + self.window

        pi_ = self._burnin_stack_ct*self.window
        pj_ = pi_+self.window
        stack_sample = model_stack[pi_:pj_]
        max_fold = np.max(stack_sample)
        cont_fold = np.min(stack_sample)

        return max_fold, cont_fold

    # ###################################################### #
    # Input model prediction and metadata validation methods #
    # ###################################################### #
    def _validate_pred(self, pred):
        
        if not isinstance(pred, (torch.Tensor, np.ndarray))
            raise TypeError(f'pred is type {type(pred)} - unsupported. Supported types: torch.Tensor, numpy.ndarray')
        
        if len(pred.shape) != 1:
            try:
                pred.reshape(self.window)
            except ValueError:
                raise ValueError(f'pred {pred.shape} cannot be reshaped into ({self.window},)')

        # If pred hasn't already been detached and 
        # converted into numpy, do so now
        if isinstance(pred, torch.Tensor):
            if pred.device.type != 'cpu':
                vpred = pred.copy().detach().cpu().numpy()
            else:
                vpred = pred.copy().detach().numpy()
        else:
            vpred = pred.copy()
        # Finally, if reshape is needed to peel off extra
        # singleton axes, do so now
        if vpred.shape != (self.window,):
            vpred = vpred.reshape(self.window)        
        return vpred        





    def _populate_data_from_stack(self, fill_value=0, fold_thresh=None):
        """
        Populate the self.data attribute and change-sensitive
        fields in self.stats inherited from obspy.Trace using
        stacked values in self.stack and data fold information
        contained in self.fold, converting self.fold into a 
        data mask where self.fold < fold_thresh are masked

        :: INPUT ::
        :param fill_value: [float-like] or [None]
                        fill_value to assign to the masked_array written
                        to self.data by this method
        :param fold_thresh: [int-like] or [None]
                        [int-like] - data with fold g.e. fold_thresh
                        are mapped as unmasked values in self.data
                        [None] - fold_thresh is set to _cont_fold
        
        mapped values from stack to data have the stacking method
        (self.method) applied as follows:
            self.method = 'max'
            stack[_i] = data[_i]
            self.method = 'mean'
            stack[_i]/fold[_i] = data[_i]

        """
        if fold_thresh is None:
            fold_thresh = self._cont_fold

        if self._data_appended:
            # Convert fold into mask array using fold_thresh
            _mask = np.zeros(self._max_npts)
            _mask[np.argwhere(self.fold >= fold_thresh)] += 1
            # Compose data array with stack-method scaling
            if self.method == 'max':
                _data = self.stack.copy()
            elif self.method == 'mean':
                _data = self.stack.copy() / self.fold.copy()
            # Composed masked array
            _marray = np.ma.masked_array(
                data=_data,
                mask=_mask,
                fill_value=fill_value
            )
            # TODO: need to account for buffering aspects.
                # DO that here?

    def crosscheck_metadata(self, meta):
        """
        Systematically check contents of a metadata
        dictionary generated by an InstWindow object
        against the same attributes for this RtPredBuff
        object. Return a dictionary 
        """

        meta_tests = {}
        # Check inst_code
        if self.inst_code is None:
            self.inst_code = meta['inst_code']
            meta_tests.update({'inst_code': True})
        elif self.inst_code != meta['inst_code']:
            meta_tests.update({'inst_code': False})
        else:
            meta_tests.update({'inst_code': True})
        
        # Check model_name
        if self.model_name is None:
            self.model_name = meta['model_name']
            meta_tests.update({'model_name': True})
        elif self.model_name != meta['model_name']:
            meta_tests.update({'model_name': False})
        else:
            meta_tests.update({'model_name': True})

        # Check weight_name
        if self.weight_name is None:
            self.weight_name = meta['weight_name']
            meta_tests.update({'weight_name': True})
        elif self.weight_name != meta['weight_name']:
            meta_tests.update({'weight_name': False})
        else:
            meta_tests.update({'weight_name': True})

        # Handle advance_npts update/check
        if self.advance_npts is None:
            self.advance_npts = meta['advance_npts']
            meta_tests.update({'advance_npts': True})
        elif self.advance_npts != meta['advance_npts']:
            meta_tests.update({'advance_npts': False})
        else:
            meta_tests.update({'advance_npts': True})

        # Handle samprate
        if self.samprate is None:
            self.samprate = meta['samprate']
            meta_tests.update({'samprate': True})
        elif self.advance_npts != meta['samprate']:
            meta_tests.update({'samprate': False})
        else:
            meta_tests.update({'samprate': True})
        
        
        
        # Handle index
        if meta['index'] in self.index_queue:
            index_status='duplicate'
        else:
            # Handle case where new metadata occurs before reference index
            if meta['index'] < self.ref_index:
                # If backward step gap is larger than max_windows, cancel append
                if self.ref_index - meta['index'] < self.max_windows:
                    index_status='outsized_lead'
                else:
                    index_status ='before reference'

            
            # Handle case where new metadata occurs after the reference index

        return index_status, meta_tests


, model_name='EQTransformer', weight_name='pnw', window=6000, blinding=(500,500), overlap=1800, method='max'):
        

    def first_append(self, pred, meta):
        # Scrape metadata for first append
        index_status, meta_tests = self.crosscheck_metadata(meta)
        if all(meta_tests.values):
            _i0 = self.ref_index*
            # Assign prediction values to 
            self.stack[:, :self.window] = pred
            lblind = self.blinding[0]
            rblind = self.window - self.blinding[1]
            self.fold[lblind:rblind] = 1
            self.fold[:lblind] = 0
            self.fold[rblind:self.window] = 0
        else:
            emsg = 'Errors in crosschecks of:'
            for _k in status.keys():
                if not status[_k]:
                    emsg += f' {_k}'
            raise ValueError(emsg)
        return self

    def in_range_append(self, pred, meta):
        """
        Append a new window of predicted values to already
        initialized stack and fold arrays.
        """
        status = self.crosscheck_metadata(meta)
        # Get new 

    def validate_new_index(self, index, allow_backslide=False):
        if len(self.index_queue) == 0:
            return True
        elif index in self.indeq_queue:
            return False
        elif index > max(self.index_queue):


    def is_index_in_bounds(self, index, allow_backslide=False):
    
        if index 

    def _index_to_samples(self, index):
        d_idx = index - self.ref_index
        ref_index = self.ref_index
        ref_time = self.ref_time
        samp_rate = self.samprate
        advance_npts = self.window - self.overlap
    
    def _index_to_timestamp(self, index):
        """
        Convert an input index into a
        modeled time offset given the ref_index
        ref_time and samprate attributes of this
        RtPredBuff
        """
        d_idx = index - self.ref_index
        adv_npts = self.window - self.overlap
        d_sec = d_idx*adv_npts*self.samprate
        datetime = self.ref_datetime + d_sec
        return datetime
            


    def append(self, pred, meta):
        index_status, meta_tests = self.cross




    def append(self, pred, meta):
        if 









        # # model_name compatability checks
        if not isinstance(model_name, str):
            raise TypeError('model_name must be type str')
        elif model_name not in ['EQTransformer','PhaseNet']:
            raise ValueError(f'model_name "{model_name}" not presently supported. Supported model_name values: "EQTransformer", "PhaseNet"')
        else:    
            self.model_name = model_name

        # # weight_name compatability check
        if not isinstance(weight_name, str):
            raise TypeError('weight_name must be type str')
        elif self.model_name == 'EQTransformer':
            if weight_name not in ['ethz', 'geofon', 'instance', 'iquique', 'lendb', 'neic', 'obs', 'original', 'original_nonconservative', 'pnw', 'scedc', 'stead']:
                raise ValueError(f'weight_name "{weight_name}" is not a currently supported SeisBench pretrained weight for EQTransformer')
            else:
                self.weight_name = weight_name
        elif self.model_name == 'PhaseNet':
            if weight_name not in ['diting', 'ethz', 'geofon', 'instance', 'iquique', 'lendb', 'neic', 'obs', 'original', 'scedc', 'stead']:
                raise ValueError(f'weight_name "{weight_name}" is not a currently supported SeisBench pretrained weight for PhaseNet')
            else:
                self.weight_name = weight_name

        # # window compatability checks
        self.window = icc.bounded_intlike(
            window,
            name='window',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # # blinding compatability checks
        # checks for (2-)tuple input
        if isinstance(blinding, tuple):
            blinding_l = icc.bounded_intlike(
                blinding[0],
                name='blinding[0]',
                minimum=0,
                maximum=self.window/2,
                inclusive=True
            )
            blinding_r = icc.bounded_intlike(
                blinding[-1],
                name='blinding[-1]',
                minimum=0,
                maximum=self.window/2,
                inclusive=True
            )
            self.blinding = (blinding_l, blinding_r)
        # checks for single value input
        else:
            val = icc.bounded_intlike(
                blinding,
                name='blinding',
                minimum=0,
                maximum=self.window/2,
                inclusive=True
            )
            self.blinding = (val, val)
        # # overlap compatability checks
        self.overlap = icc.bounded_intlike(
            overlap,
            name='overlap',
            minimum=0,
            maximum=self.window-1,
            inclusive=True
        )
        # # method compatablity checks
        if not isinstance(method, str):
            raise TypeError('method must be type str')
        elif method.lower() not in ['max','mean']:
            raise ValueError(f'method {method.lower()} not supported. Supported: "max", "mean"')
        # # Calculate stacking attributes
        # The maximum number of data this prediction buffer can hold
        self._max_npts = (self.max_windows - 1)*(self.window - self.overlap)
        # The number of consecutive stacked windows required to hit max/continuous stacking fold
        self._burnin_ct = self.window//(self.window - self.overlap) + 1
        # Model continuous and maximum stacking fold
        self._max_fold, self._cont_fold = self.model_stacking_fold()
        # Set the next expected window index counter as 0
        self.ref_index = 0
        # Create deque for current window index holdings
        self.index_queue = deque([])

        # Initialize stacked data and fold arrays
        # NOTE: these are the definitive data attributes in this object
        self.stack = np.full(shape=(self._max_npts), fill_value=np.nan, dtype=np.float32)
        self.fold = np.full(shape=(self._max_npts), fill_value=0, dtype=np.int32)


        # NTS: I keep waffling back and forth thinking this should
        # either be included as an inheritance case to handle timing
        # or simply as an output class-method (i.e., RtPredBuff.to_trace())
        # that can have some 
        # # initialize Trace representation
        # # NOTE: the trace representaion is a mirror to the stack and fold data attributes
        # super().__init__(data=[], header={})
        # self._data_appended = False








    def _get_candidate_k(self, index):
        



    def _shift_stack_k_samples(self, k_samp, safemode=True):
        """
        Shift and trim the contents of self.stack by k_samp samples
        with a negative k_samp value indicating a left shift and
        a positive k_samp value indicating a right shift.

        Provides a safemode option (default True) that prevents
        fully shifting contents of self.stack out of range

        :: INPUTS ::
        :param k_samp: [int] number of samples to shift the contents of
                        self.stack by over axis 2 (k-axis), adhering to
                        the [i, j, k] axis structure of SeisBench WaveformModel
                        inputs and outputs
        :param safemode: [bool] True - raise warning of k_samp > self.stack.shape[2]
                                False - overwrite self.stack with a np.nan-filled
                                        array matching self.stack.shape.
        :: UPDATE ::
        :attr stack: 
        
        :: OUTPUT ::
        :return self:
        """

        if not isinstance(k_samp, int):
            raise TypeError('k_samp must be type int')
        elif abs(k_samp) > self.stack.shape[1]:
            if safemode:
                raise ValueError(f'k_samp step is larger than stack length ({self.stack.shape[1]})')
            else:
                self.stack = np.full(shape=self.stack.shape, fill_value=np.nan, dtype=np.float32)
                return self
        else:
            pass
        
        tmp_stack = np.full(shape=self.stack.shape, fill_value=np.nan, dtype=np.float32)
        if k_samp < 0:
            tmp_stack[:,:-k_samp] = self.stack[:, k_samp:]
        elif k_samp > 0:
            tmp_stack[:, k_samp:] = self.stack[:, :-k_samp]
        else:
            tmp_stack = self.stack.copy()

        self.stack = tmp_stack
        return self


    def _shift_fold_k_samples(self, k_samp, safemode=True):
        """
        Shift the contents of the self.fold vector by k_samp with
        a negative k_samp value indicating a leftward shift of contents
        and a positive k_samp value indicating a rightward shift.

        Provides a safemode (default True) that prevents applying
        shifts that would completely move 
        """
        if not isinstance(k_samp, int):
            raise TypeError('k_samp must be type int')
        elif abs(k_samp) > self.fold.shape[0]:
            if safemode:
                raise ValueError(f'k_samp step is larger than fold length ({self.stack.shape[0]})')
            else:
                self.fold = np.full(shape=self.fold.shape, fill_value=np.nan, dtype=np.float32)
        else:
            pass
        
        tmp_fold = np.full(shape=self.fold.shape, fill_value=np.nan, dtype=np.float32)
        if k_samp < 0:
            tmp_fold[:,:-k_samp] = self.fold[:, k_samp:]
        elif k_samp > 0:
            tmp_fold[:, k_samp:] = self.fold[:, :-k_samp]
        else:
            tmp_fold = self.fold.copy()

        self.fold = tmp_fold
        return self




















    # def _first_stack(self, vpred, vmeta):
    #     current_stack = self.data.copy()
    #     first_window = vpred


    #     return self
        


        
    # def _contiguous_stack(self, vpred, vmeta):
    #     delta = vmeta['window_index'] - self.index_queue[0]
    #     if 0 < delta <= self._
    #     new_npts = self.data.shape[0] + delta*(self.window - self.overlap)
    #     new_stack = np.full(shape=(new_shape,), fill_value=np.nan, dtype=np.float32)
    
    #     ci_start = current_stack.shape[0] - self.overlap
    #     ni_start = 0
    #     ci_end = current_stack.shape[0]
    #     ni_end = self.overlap


        
        
# ## ALSO SCRATCHPAD ###
#         # Initialize placeholder arrays
#         self._placeholder_pred = np.ma.masked_array(
#             data=np.zeros(shape=(self.window,)),
#             mask=np.full(shape=(self.window), fill_value=True),
#             fill_value=np.nan)
#         self._placeholder_meta = 
#         # Initialize pred_queue - this holds raw predictions
#         self.pred_queue = deque([])
#         self.meta_queue = deque([])
#         self.index_queue = deque([])
#         # Initialize obspy.Trace representation 
#         super().__init__(data=[], header={})







### SCRATCHPAD ###

    # def stack(self, method='max'):
    #     stack = np.zeros(shape=(self._max_npts))

    # def append(self, pred, meta):
    #     # Run validation checks on incoming data and metadata
    #     vpred, vmeta = self.validate_incoming(pred, meta)
    #     # If (pred, meta) represent the next contiguous prediction window        
    #     if vmeta['window_index'] > self.index_queue[0]:
    #         while vmeta['window_index'] > self.index_queue[0] + 1:
    #             self.pred_queue.appendleft(self._placeholder_pred)
    #             self.meta_queue.appendleft(self._placeholder_meta)
    #             self.index_queue.appendleft(self.index_queue[0] + 1)
    #             if len(self.index_queue) > self.max_windows:
    #                 while len(self.index_queue) > self.max_windows:
    #                     self.pred_queue.pop()
    #                     self.meta_queue.pop()
    #                     self.index_queue.pop()
    #         self.pred_queue.append(vpred)
    #         self.meta_queue.append(vmeta)
    #         self.index_queue.append(vmeta['window_index'])
    #     elif        




    #     # Initialize data vector - this holds stacked data
    #     self.data = np.full(shape=(self._max_npts,), fill_value=np.nan, dtype=np.float32)
    #     # Initialize fold vector - this will hold integer counts
    #     self.fold = np.full(shape=(self._max_npts,), fill_value=0., dtype=np.int32)
    #     # Initialize empty dictionary for header information
    #     self.header = {}
    #     # 
    #     self.indices = np.full(shape=(self.max_windows,) fill_value=np.nan, dtype=np.int32)
    #     # # # initialize obspy.Trace inheritance, starting with empty trace
    #     # super().__init__(data=[], header={})
    #     # # Initialize stacking attributes
    #     self.calc_stacking_attributes()
    #     self._appended_indices = deque([])



    # def 


    # def append(self, pred, meta, method='max'):
    #     # ensure predicted values have correct dimensionality
    #     vpred, vmeta = self.validate_new_window(pred, meta)

        


    #     # ensure metadata has the right information and formatting
    #     vmeta = self.validate_meta(meta)
    #     # If this is the first append
    #     if len(self._appended_indices) == 0:
    #         self._first_append(vpred, vmeta):
    #     # If candidate window overlaps with the last data
    #     elif vmeta['window_index'] <= self._max_fold + max(self._appended_indices)
    #         self._contiguous_append(vpred, vmeta, method=method)
    #     # If candidate window does not overlap with last data
    #     elif vmeta['window_index'] > self._max_fold + max(self._appended_indices)


    #     # If the appended index is already in the _appended_indices attribute 
    #     elif vmeta['window_index'] in self._appended_indices:
    #         print('candidate window index is already in appended list - canceling append')
    #         pass
    #     # If the candidate window has an index lower than the minimum index in this buffer
    #     elif vmeta['window_index'] < min(self._appended_indices):
    #         print('candidate window occurs before current buffered windows - canceling append')
    #         pass

    #     if vmeta['window_index'] > self.
        

    # def

    # def _stack_max(self,data,index):
    #     stack = np.full(shape=())


    # def validate_pred(self, pred):
    #     """
    #     Run validation checks on pred and return
    #     a (slightly) altered pred that conforms to
    #     the expected shape (self.window,)

    #     :: INPUT ::
    #     :param pred: [numpy.ndarray] candidate predicted values array
    #                     that should have a shape that matches
    #                     the specified window length in samples, plus
    #                     up to singular element axes that lead the data
    #                     axis.
    #     """

    #     # Run check that pred is a numpy.ndarray
    #     if not isinstance(pred, np.ndarray):
    #         raise TypeError(f'pred must be type numpy.ndarray, not {type(pred)}')
    #     # Run check on dimensions, reshaping if needed
    #     if pred.shape == (1, 1, self.window):
    #         opred = pred.copy().reshape(self.window)
    #     elif pred.shape == (1, self.window):
    #         opred = pred.copy().reshape(self.window)
    #     elif pred.shape == (self.window,):
    #         opred = pred.copy()
    #     else:
    #         raise IndexError(f'dims of pred {pred.shape} do not match expected values ({self.window},)')
        



    # def __add__(self, trace, method='max')
        
    #     status = self._run_trace_crosschecks(trace)
    #     if isinstance(status, str):
    #         raise TypeError(status)
        
    #     if self.stats.starttime <= trace.stats.starttime:
    #         lt = self
    #         rt = trace
    #     else:
    #         lt = trace
    #         rt = self
        

        
    #     # Check for gap
    #     sr = self.stats.sampling_rate
    #     delta = (rt.stats.starttime - lt.stats.endtime)*sr
    #     delta = int(round_away(delta)) - 1
    #     delta_endtime= lt.stats.endtime - rt.stats.endtime
    #     out = self.__class__(header=deepcopy(lt.stats))

    #     if delta < 0 and delta_endtime < 0:
    #         # overlap
    #         delta = abs(delta)
    #         if np.all(np.equal(lt.data[-delta:], rt.data[:delta]))):
                

    #     if delta < 0:
    #         overlap = True
    #     else:
    #         overlap = False
        