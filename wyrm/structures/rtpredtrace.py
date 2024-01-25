from obspy import Trace
import numpy as np
from obspy.core.compatibility import round_away
import wyrm.util.input_compatability_checks as icc

class RtPredTrace(Trace):
    """
    Adaptation of the obspy.RtTrace and wyrm.
    """
    def __init__(self, max_windows=20, window=6000, blinding=(500,500), overlap=1800, method='max'):
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
        # # max_windows compatability checks
        self.max_windows = icc.bounded_intlike(
            max_windows,
            name='max_windows',
            minimum=0,
            maximum=None,
            inclusive=False
        )
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
        self.calc_stacking_attributes()

        self.stack = np.full(shape=(self._max_npts), fill_value=np.nan, dtype=np.float32)
        self.fold = np.full(shape=(self._max_npts), fill_value=0, dtype=np.int32)
        # initialize Trace representation
        super().__init__(data=[], header={})
        self._data_appended = False

    def calc_stacking_attributes(self):
        """
        Calculate the window stacking attributes
        for this buffer based on pre-defined
        windowing attributes

        :: INPUT ::
        :param pad_windows: [int] number of window lengths to pad
                        the stacking model by for calculating
                        the stack fold attributes. Must be
                        in the range [1, 5]

        NOTE:   Depending on the blinding and overlap parameters provided,
                there may be some saw-toothing in the maximum number of unblinded
                samples from consecutive windows, as such we calculate both the 
                maximum data fold (_max_fold) attained and the maximum continuous
                data fold (_cont_fold) values. 
        
        Populates
        self._burnin_count - number of consecutive stacks needed
                         to attain _max_fold
        self._max_fold - absolute maximum data stacking fold
        self._cont_fold - continuous maximum data stacking fold

        self._wind_wgts - bindary vector for data blinding
                          that can be passed to a 
                          numpy.ma.masked_array as a mask
        """
        # Get the maximum number of points there should be in this buffer
        self._max_npts = (self.max_windows - 1)*(self.window - self.overlap) + self.window
        self._burnin_stack_ct = self.window//(self.window - self.overlap) + 1
        
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
        self._max_fold = np.max(stack_sample)
        self._cont_fold = np.min(stack_sample)


        return self
    
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
        