from obspy import Trace, Stream, UTCDateTime
import numpy as np
import wyrm.util.input_compatability_checks as icc
from wyrm.util.semblance import shift_trim, roll_trim_rows
import torch
import seisbench.models as sbm
import matplotlib.pyplot as plt
from time import time



class RtPredBuff(object):
    """
    Realtime Prediction Buffer - this class acts as a buffer for overlapping
    predictions 
    """
    def __init__(self, max_length=120, stack_method='max', fill_value=0., dtype=np.float32, model=None, debug=False):
        """
        Initialize a Realtime Prediction Buffer object. Most attributes
        will be populated by the first data/metadata append process
        using the self.initial_append(pred, meta) method.

        :: INPUTS ::

        :param max_length: [float] maximum buffer length in seconds
        :param stack_method: [str] method for stacking non-blinded
                        overlapping predictions.
                        Supported methods:
                            'max': using np.nanmax, update stack
                                   for overlapping label samples
                                   (stack[i,j], pred[i,j])
                            'avg': update stack for overlapping samples as:
                                  stack[i,j] = (stack[i,j]*fold[i,j] + 
                                                pred[i,j])/(fold[i,j] + 1)
        :param fill_value: [scalar] placeholder value `fill_value` in 
                        numpy.full() - used for empty data entry positions
                        in self.stack and self.fold
                        -- for compatability with ufunc this needs to be
                        set to 0. (OBSOLITE AS OPTION)
        :param model: [seisbench.models.WaveformModel] or None
                        Model from which to scrape metadata (optional)        
        """
        if not isinstance(model, (sbm.WaveformModel, type(None))):
            raise TypeError('input model must be type seisbench.models.WaveformModel or None')
        elif isinstance(model, sbm.WaveformModel):
            self.model_name = model.name
            self.label_names = model.labels
            if self.label_names is not None:
                self.label_codes = [_l[0].upper() for _l in model.labels]
            else:
                self.label_codes = None
            self.window_npts = model.in_samples
        else:
            self.model_name = None
            self.label_names = None
            self.label_codes = None
            self.window_npts = None

        if not isinstance(debug, bool):
            raise TypeError('debug must be type bool')
        else:
            self.debug = debug

        ### Static parameters (once assigned) ###
        # max_length compatability checks
        # Maximum number of windows allowed in buffer
        self.max_length = icc.bounded_floatlike(
            max_length,
            name='max_length',
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
        
        if not isinstance(fill_value, (int, float, type(None))):
            raise TypeError('fill_value must be type int, float or None')
        else:
            self.fill_value = fill_value
        
        # Instrument Metadata
        self.inst_code = None
        # Window Indexing Scalars
        self.overlap_npts = None
        self.blinding_npts = None        
        # Sampling Rate (float)
        self.samprate = None
        # Placeholder for binary array for blinding samples
        self.blinding_array = None
        # Placeholder for timestamp corresponding to index 0
        self.t0 = None
        # Dynamic data arrays
        self.stack = None
        self.fold = None
        # Has data appended flag
        self._has_data = False

    def append(self, pred, meta, include_blinding=False):
        """
        Generalized wrapper for the methods:
            inital_append(pred, meta, model, wgt_name)
            subsequent_append(pred, meta)

        :: INPUTS ::
        :param pred: [numpy.ndarray] or [torch.Tensor]
                    predicted value array output from SeisBench-hosted
                    PyTorch model (a WaveformModel-child class)
        :param meta: [dict]
                    Dictionary containing metadata associated with
                    pred
        :param model: [seisbench.models.WaveformModel]
                    Model object used to generate pred
        :param wgt_name: [str]
                    Shorthand name for the training weight set loaded
                    in `model`.
        """
        if self.debug:
            tick0 = time()
        # Run metadata validation 
        vmeta = self.validate_metadata(meta)
        if self.debug:
            tick1 = time()
            print(f'validate metadata {tick1 - tick0:.3f} sec')
            
        # If this is an initial append, scrape metadata to fill out attributes
        if not self._has_data:
            self._meta2attrib(vmeta)
            if self.debug:
                tick2 = time()
                print(f'scraped metadata {tick2 - tick1:.3f} sec')
        # Run prediction validation
        elif self.debug:
            tick2 = time()
        vpred = self.validate_prediction(pred)
        if self.debug:
            tick3 = time()
            print(f'validate prediction {tick3 - tick2:.3f} sec')
        # Get append instructions
        instr = self.calculate_stacking_instructions(vmeta, include_blinding=include_blinding)
        if self.debug:
            tick4 = time()
            print(f'calc stacking instr {tick4 - tick3:.3f} sec')
        # Handle different proposed shift cases
        # If initial append
        if not self._has_data:
            # Apply instructions without restrictions
            self._apply_stack_instructions(instr, vpred)
            self._has_data = True
            if self.debug:
                tick5 = time()
                print(f'apply stacking {tick5 - tick4:.3f}sec') 
        else:
            # If proposal is a leftward shift of the stack (right-end append)
            if instr['npts_right'] < 0:
                # Apply instructions without restrictions
                self._apply_stack_instructions(instr, vpred)
                if self.debug:
                    tick5 = time()
                    print(f'apply stacking {tick5 - tick4:.3f}sec') 
            # if proposal is a no-shift stacking
            elif instr['npts_right'] == 0:
                # Check that the proposed stack still should have room to grow
                if all(self.fold[instr['i0_s']:instr['i1_s']] >= self.cont_fold):
                    raise BufferError('Attempting to append to a completely filled buffer zone - canceling append')
                else:
                    self._apply_stack_instructions(instr, vpred)
                    if self.debug:
                        tick5 = time()
                        print(f'apply stacking {tick5 - tick4:.3f}sec') 
            elif instr['npts_right'] > 0:
                # Check if the proposed shift would truncate rightward (leading) samples
                if any(self.fold[-instr['npts_right']:] > 0):
                    raise BufferError('Attempting to append in a manner that would clip leading prediction values - canceling append')
                else:
                    self._apply_stack_instructions(instr, vpred)
                    if self.debug:
                        tick5 = time()
                        print(f'apply stacking {tick5 - tick4:.3f}sec') 
        return self
    
    def validate_metadata(self, meta):
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
            (float, int, type(None)),
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
            if all(_k in meta.keys() for _k in def_dict.keys()):
                pass
            else:
                emsg = 'Misfit keys in "meta":'
                # for _k in meta.keys():
                #     if _k not in window_metadata_keys:
                #         emsg += f'\n{_k} (extraneous key in meta)'
                for _k in def_dict.keys():
                    if _k not in meta.keys():
                        emsg == f'\n{_k} (missing from meta)'
                raise KeyError(emsg)
        # If there are mismatched datatypes in meta, compose
        # a detailed error message
        if not all(isinstance(meta[_k], def_dict[_k]) for _k in def_dict.keys()):
            emsg = 'meta.values() dtype mismatches:'
            for _k in def_dict.keys():
                if not isinstance(meta[_k], def_dict[_k]):
                    emsg += f'\n{_k} is {type(meta[_k])} - {def_dict[_k]} expected'
            raise TypeError(emsg)
        
        vmeta = meta.copy()
        # If this is not the RtPredBuff's first rodeo... (i.e., already has data appended)
        # Run additional checks to match instrument and model parameters
        if self._has_data:
            # Compose check tuples
            self_list = [self.inst_code, self.model_name, self.window_npts, self.overlap_npts, self.blinding_npts, self.samprate]
            meta_list = ['inst_code','model_name','window_npts','overlap_npts','blinding_npts','samprate']
            check_tuples = list(zip(self_list, meta_list))
            # Run check tuples with f-string error message composition
            for _s, _k in check_tuples:
                if _s != meta[_k]:
                    emsg = f'{_k} in meta ({meta[_k]}) does not match'
                    emsg += f' value recorded in this RtPredBuff ({_s})'
                    raise ValueError(emsg)
        return vmeta

    def _meta2attrib(self, vmeta, include_blinding=False, dtype=np.float32):
        """
        Using a pre-validated meta(data) dictionary (see validate_metadata())
        to populate most attributes in this RtPredBuff object

        :: INPUTS ::
        :param vmeta: [dict] dictionary formatted with prediction window metadata
        :param fill_value: [scalar] value to pass to numpy.full as fill_value

        :: ATTRIB ::
        :inst_code: [str] instrument code used for validating subsequent appends
        :model_name: [str] ML model name used for validating subsequent appends
        :window_npts: [int] number of data points in each prediction array
                        - This is the last axis in SeisBench formatting standards
        :overlap_npts: [int] number of overlapping window samples expected
        :blinding_npts: [int] number of samples on each side of a prediction that
                    will be blinded (i.e. assigned fill_value)

        """
        # Metadata direct mappings
        self.inst_code = vmeta['inst_code']
        if self.model_name is None:
            self.model_name= vmeta['model_name']
        else:
            if self.model_name != vmeta['model_name']:
                raise ValueError('vmeta model_name does not match assigned model_name')
        
        if self.window_npts is None:
            self.window_npts = vmeta['window_npts']
        else:
            if self.window_npts != vmeta['window_npts']:
                raise ValueError('vmeta window_npts does not match assigned window_npts')
        
        self.overlap_npts = vmeta['overlap_npts']
        self.blinding_npts = vmeta['blinding_npts']
        self.samprate = vmeta['samprate']
        if include_blinding:
            self.t0 = vmeta['starttime']
        else:
            self.t0 = vmeta['starttime'] + (self.blinding_npts/self.samprate)
        
        # Get advance samples
        self.adv_npts = self.window_npts - self.overlap_npts
        # Get expected fold ceiling
        self.max_fold = np.ceil((self.window_npts - 2*self.blinding_npts)/self.adv_npts)
        # Get expected continuous maximum fold
        self.cont_fold = np.floor((self.window_npts - 2*self.blinding_npts)/self.adv_npts)

        # Populate stack and fold arrays
        # Get mapping to max_length in samples
        self.max_length_npts = int(np.round(self.max_length*self.samprate))
        shp = (len(self.label_codes), self.max_length_npts)
        # Populate stack and fold attributes
        self.stack = np.full(shape=shp, fill_value=self.fill_value, dtype=dtype)
        self.fold = np.full(shape=self.max_length_npts, fill_value=0, dtype=self.stack.dtype)
        # Compose blinding array
        self.blinding_array = np.ones(shape=self.window_npts)
        self.blinding_array[:self.blinding_npts] = self.fill_value
        self.blinding_array[-self.blinding_npts:] = self.fill_value
        return self

    def validate_prediction(self, pred):
        """
        Validate candidate predicted value array against expected shape
        based on metadata scraped from the first 
        """
        # Initial check if there are sufficient attribute metadata to conduct
        # the validation
        if self.window_npts is None:
            raise ValueError('Initial metadata have not been populated - cannot validate pred input')
        # Check for valid pred type
        if not isinstance(pred, (torch.Tensor, np.ndarray)):
            raise TypeError('input "pred" must be type torch.Tensor or numpy.ndarray')


        # Array shape checks against object attribute metadata
        target_shp = (len(self.label_codes), self.window_npts)
        if pred.shape != target_shp:
            # Attempt reshape
            try:
                vpred = pred.reshape(target_shp)
            # If reshape kicks ValueError, populate specific ValueError message
            except ValueError:
                emsg = f'input "pred" shape ({pred.shape}) does not match '
                emsg += f'expected shape ({len(self.label_codes)}, {self.window_npts})'
                raise ValueError(emsg)
        else:
            vpred = pred
        # Handle conversion to numpy array if not already done
        if isinstance(vpred, torch.Tensor):
            # Handle case where input is still on non-cpu hardware
            if pred.device.type != 'cpu': 
                vpred = vpred.detach().cpu().numpy()
            # Otherwise, standard detach -> numpy array
            else:
                vpred = vpred.detach().numpy()
  
        # Return altered data object
        return vpred

    def calculate_stacking_instructions(self, vmeta, include_blinding=True):
        """
        Calculate the instructions for stacking a new prediction window
        to this RtPredBuff given pre-validated metadata (vmeta) and
        attributes of this RtPredBuff object.

        :: INPUT ::
        :param vmeta: [dict] dictionary output by validate_metadata()
        (IN DEVELOPMENT)
        :param include_blinding: [bool] should blinded samples be considered
                        when calculating shifts?
        :: OUTPUT ::
        :return instructions: [dict] instructions for shifting/trimming
                    and stacking in new window. 
                    Keys:
                        'npts_right': number of points to shift samples
                            along their data axes. 
                            Value should be passed to wyrm.util.stacking.shift_trim()
                        'i0_p': None or positive [int]
                            first sample from pred along data axis to include in
                            stacking
                        'i0_s': None or positive [int]
                            first index value that the candidate prediction
                            window will occupy AFTER the stack and fold have
                            been shifted
                        'i1_p': None or positive [int]
                            last sample from pred along data axis to include
                            in stacking
                        'i1_s': None or positive [int]
                            last index value that the candidate prediction
                            window will occupy AFTER the stack and fold have
                            been shifted
        """
        # Start construction of instructions
        # If blinding is included, result in an indexer of [:]
        if include_blinding:
            instructions = {'i0_p': 0,
                            'i1_p': None}
        # Otherwise, provide an indexer of [blinding_npts:-blinding_npts]
        else:
            instructions = {'i0_p': self.blinding_npts,
                            'i1_p': -self.blinding_npts}
        # If this is for the first append
        if not self._has_data:
            # Do not apply a shift and pin the first pred sample to the 0 index
            instructions.update({'npts_right': 0,
                                 'i0_s': None})
            # If blinding is included, place the entire
            if include_blinding:
                instructions.update({'i1_s': self.window_npts})
            else:
                instructions.update({'i1_s': self.window_npts - 2*self.blinding_npts})

        # If this is a subsequent append
        elif self._has_data:
            # Get second-scaled time shift between the reference time and the
            # starttime of the new data
            dt = vmeta['starttime'] - self.t0
            # Calculate the integer location of the first sample of the prediction window
            i0_init = dt*self.samprate
            # Sanity check that location is integer-valued
            if int(i0_init) != i0_init:
                raise ValueError('proposed new data samples are misaligned with integer sample time-indexing in this RtPredBuff')
            # Otherwise, ensure i0 is type int
            else:
                i0_init = int(i0_init)
            # Get index of last sample in candidate prediction window
            i1_init = i0_init + self.window_npts
            # If blinding samples are removed, adjust the indices
            if not include_blinding:
                i0_init += self.blinding_npts
                i1_init -= self.blinding_npts
                di = self.window_npts - 2*self.blinding_npts
            else:
                di = self.window_npts

                
            # Handle data being appended occurs before the current buffered data
            if i0_init < 0:
                # Instruct shift to place the start of pred at the start of the buffer
                instructions.update({'npts_right': -i0_init,
                                     'i0_s': None,
                                     'i1_s': di})

            # If the end of pred would be after the end of the current buffer timing
            elif i1_init > self.max_length_npts:
                # Instruct shift to place the end of pred at the end of the buffer
                instructions.update({'npts_right': self.max_length_npts - i1_init,
                                     'i0_s': -di,
                                     'i1_s': None})
            # If pred entirely fits into the current bounds of buffer timing
            else:
                # Instruct no shift and provide in-place indices for stacking pred into buffer
                instructions.update({'npts_right': 0,
                                     'i0_s': i0_init,
                                     'i1_s': i1_init})

        return instructions

    def _apply_stack_instructions(self, instructions, vpred):
        """
        Apply stacking instructions for validated prediction
        array `vpred`, using general instructions set when this
        RtPredBuff was initialized (i.e., stack_method & fill_value)
        and window-specific instructions contained in `instructions`

        :: INPUTS ::
        :param instructions: dictionary output by the calculate_stacking_instructions()
                        class method for this specific prediction window
        :param vpred: validated prediction window array
                        see class method validate_prediction()
        
        :: ALTERS ::
        :attr stack:
        :attr fold:
        :attr t0: updates reference time for data-index position 0
        """
        instr = instructions
        print(f'shift will be {instr["npts_right"]}')
        # Shift stack along 1-axis
        self.stack = shift_trim(
            self.stack,
            instr['npts_right'],
            axis=1,
            fill_value=self.fill_value,
            dtype=self.stack.dtype)
        
        # Shift fold along 0-axis
        self.fold = shift_trim(
            self.fold,
            instr['npts_right'],
            axis=0,
            fill_value=self.fill_value,
            dtype=self.fold.dtype)
        
        # Benchmarking suggests that this method is slower by 3x
        # if self.debug:
        #     time1 = time()
        # if instr['npts_right'] != 0:
        #     self.stack = roll_trim_rows(self.stack, instr['npts_right'], fill_value=self.fill_value)
        #     self.fold = roll_trim_rows(self.fold, instr['npts_right'], fill_value=self.fill_value)
        #     self.t0 -= (instr['npts_right']/self.samprate)
        # if self.debug:
        #     time2 = time()
        #     print(f'    rolled buffers {time2 - time1: .3f} sec')
        
        # OBSOLITE - runs several orders of magnitude slower than ufunc-facilitated operations
        # Update stack and fold entries with local variables
        # Extract/copy predicted values
        # pred = vpred[:, instr['i0_p']:instr['i1_p']]
        # # Extract
        # stack = self.stack[:, instr['i0_s']:instr['i1_s']]
        # fold = self.fold[instr['i0_s']:instr['i1_s']]
        # # This is slow as a loop - vectorize with ufuncs
        # for _i in range(pred.shape[0]):
        #     for _j in range(pred.shape[1]):
        #         # Get individual samples
        #         try:
        #             sij = stack[_i, _j]
        #         except:
        #             breakpoint()
        #         if sij == self.fill_value:
        #             sij = 0
        #         pij = pred[_i, _j]
        #         fj = fold[_j]
        #         # Apply stacking_method instruction
        #         if self.stack_method == 'max':
        #             stack[_i, _j] = np.nanmax([sij, pij])
        #         elif self.stack_method == 'avg':
        #             try:
        #                 stack[_i, _j] = np.nansum([sij * fj, pij])/np.nansum([fj, 1])
        #             except:
        #                 breakpoint()
        #         # Update fold entry for this sample once per cycle across labels
        #         if _i == 0:
        #             fold[_j] = np.nansum([fj, 1])
        # # Re-assign values to stack and fold
        # self.stack[:, instr['i0_s']:instr['i1_s']] = stack
        # self.fold[instr['i0_s']:instr['i1_s']] = fold
            
        # # ufunc-facilitated stacking # #
        # Construct in-place prediction slice array
        pred = np.full(self.stack.shape, fill_value=0., dtype=self.stack.dtype)
        pred[:, instr['i0_s']:instr['i1_s']] = vpred[:, instr['i0_p']:instr['i1_p']]
        # Construct in-place fold update array
        nfold = np.zeros(shape=self.fold.shape, dtype=self.stack.dtype)
        nfold[instr['i0_s']:instr['i1_s']] += 1
        # Use fmax to update
        if self.stack_method == 'max':
            np.fmax(self.stack, pred, out=self.stack); #<- Run quiet
            np.add(self.fold, nfold, out=self.fold);
        elif self.stack_method == 'avg':
            breakpoint()
            # Add fold-scaled stack/prediction arrays
            np.add(self.stack*self.fold, pred*nfold, out=self.stack);
            # Update fold
            np.add(self.fold, nfold, out=self.fold);
            # Normalize by new fold to remove initial fold-rescaling
            np.divide(self.stack, self.fold, out=self.stack, where=self.fold > 0);
        # if self.stack_method == 'max':
        #     self.stack = self.stack
        if self.debug:
            time3 = time()
            print(f'    stacking ({self.stack_method}) {time3 - time2:.3f}sec')
        return self
    
    def plot(self):
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=1, nrows=self.stack.shape[0] + 1)
        axf = fig.add_subplot(gs[-1])
        axs = [fig.add_subplot(gs[_i], sharex=axf) for _i in range(self.stack.shape[0])]
        axs.append(axf)
        t_vect = np.arange(0,self.fold.shape[0])/self.samprate
        names = self.label_names + ['fold']
        for _i in range(self.stack.shape[0]):
            axs[_i].plot(t_vect, self.stack[_i, :])
            axs[_i].set_ylabel(self.label_names[_i])
            axs[_i].xaxis.set_visible(False)
            if _i == 0:
                axs[_i].set_title(f'{self.inst_code}\nModel: {self.model_name} | Stacking: {self.stack_method}')
        axf.fill_between(t_vect, np.zeros(self.fold.shape), self.fold)
        axf.set_ylabel('Data fold')
        axf.set_xlabel(f'Elapsed time since {self.t0} [sec]')
        out_dict = {'fig': fig}
        out_dict.update(dict(zip(names, axs)))

        return out_dict

    def to_stream(self, minimum_fold=1, fill_value=None):
        st = Stream()
        mask = self.fold < minimum_fold
        if fill_value is None:
            fill_value = self.fill_value
        n,s,l,bi = self.inst_code.split('.')
        header = {'network': n,
                  'station': s,
                  'location': l,
                  'starttime': self.t0,
                  'sampling_rate': self.samprate}
        for _i, _l in enumerate(self.label_codes):
            header.update({'channel':f'{bi}{_l}'})
            _tr = Trace(data=np.ma.masked_array(data=self.stack[_i,:],
                                                mask=mask, fill_value=fill_value),
                        header=header)
            st.append(_tr)
        return st

    def __str__(self):
        rstr = f'Instrument: {self.inst_code} Pred: {self.label_codes}\n'
        rstr += f'Model: {self.model_name}\nOut_Samp: {self.window_npts} '
        rstr += f'Blind_Samp: {self.blinding_npts} Over_Samp: {self.overlap_npts} Stack_Rule: {self.stack_method}\n'
        return rstr
    
    def __repr__(self):
        rstr = self.__str__()
        return rstr

        
    # def _validate_model_info(self, vmeta, model, wgt_name):
    #     # Raise errors if model and metadata have basic information mismatches
    #     if not isinstance(model, sbm.WaveformModel):
    #         raise TypeError('model must be a child-class of seisbench.models.WaveformModel')
    #     elif model.name != self.model_name:
    #         raise ValueError('model.name and self.model_name have different values')
    #     elif model.in_samples != self.window_npts:
    #         raise ValueError('model.in_samples and self.window_npts have different values')
    #     elif model.labels != self.label_names:
    #         raise ValueError('model.labels and self.model_names')
    #     if not isinstance(wgt_name, str):
    #         raise TypeError()

       
#  def subsequent_append(self, pred, meta, include_blinding=False):
#         """
#         Append an additional pred / meta pair to this RtPredBuff
#         object if it passes data checks. 
        
#         :: INPUTS ::
#         :param pred: [numpy.ndarray] or [torch.Tensor]
#                     predicted value array output from SeisBench-hosted
#                     PyTorch model (a WaveformModel-child class)
#         :param meta: [dict]
#                     Dictionary containing metadata associated with
#                     pred
    
#         """
#         # Validate metadata (and calculate index locations)
#         vmeta = self.validate_metadata(meta)
#         instructions = self.calculate_stacking_instructions(vmeta, include_blinding=include_blinding)
#         # Validate prediction
#         vpred = self.validate_prediction(pred)

#         # If proposed shift to data arrays is 0
#         if vmeta['delta i'] == 0:
#             print('noshift')
#             # Check that the candidate pred doesn't appear to be a repeat append
#             if vmeta['i0'] is None:
#                 raise IndexError('proposed append appears to be repeat data - canceling append')
#             # Otherwise, append
#             else:
#                 self._append_core_process(vpred, vmeta)
#         # If proposed shift to data arrays is leftward
#         elif vmeta['delta i'] < 0:
#             print('leftshift')
#             # Shift data unconditionally
#             breakpoint()
#             self.shift_data_arrays(vmeta['delta i'])
#             # And append
#             self._append_core_process(vpred, vmeta)
#         # If proposed shift to data arrays is rightward
#         elif vmeta['delta i'] > 0:
#             print('rightshift')
#             # Check that the shift will not truncate valid leading edge samples
#             last_non_nan_idx = np.max(np.argwhere(np.isfinite(self.fold)))
#             if self.max_length_npts - vmeta['delta i'] > last_non_nan_idx:
#                 self.shift_data_arrays(vmeta['delta i'])
#                 self._append_core_process(vpred, vmeta)
#             # If it would, kick error
#             else:
#                 breakpoint()
#                 raise IndexError('proposed append would result in truncation of leading edge data - canceling append')
            
#         return self
    # def _append_core_process(self, vpred, vmeta):
    #     """
    #     Core process for append methods that handles stack_method choice
    #     """
    #     blinded_pred = vpred*self.blinding_array
    #     i0 = vmeta['i0']
    #     if i0 is None:
    #         i0 = 0
    #     i1 = vmeta['i1']
    #     if i1 is None:
    #         i1 = self.max_length_npts
    #     for _i in range(len(self.label_codes)):
    #         for _k, _j in enumerate(np.arange(i0, i1)):
    #             # Get stack sample
    #             sij = self.stack[_i, _j]
    #             # Get fold sample
    #             fj = self.fold[_j]
    #             # Get blinding array sample
    #             bk = self.blinding_array[_k]
    #             # Get pred sample
    #             pik = blinded_pred[_i, _k]
    #             # If max stacking - apply that
    #             if self.stack_method == 'max':
    #                 self.stack[_i, _j] = np.nanmax([sij, pik])
    #             # If avg stacking - apply that
    #             elif self.stack_method == 'avg':
    #                 self.stack[_i, _j] = np.nansum([sij*fj, pik*bk])/np.nansum([fj, bk])
    #             # Update data fold
    #             self.fold[_j] = np.nansum([fj, bk])
       
    # # ##################################### #
    # # Data array shift and trim subroutines #
    # # ##################################### #
    # def shift_data_arrays(self, samps, safemode=True):
    #     dt = samps/self.samprate
    #     # Shift data
    #     self._shift_stack_j_samples(samps, safemode=safemode)
    #     self._shift_fold_i_samples(samps, safemode=safemode)
    #     # Update referenceframes
    #     self.t0 += dt
    #     return self
        

    # def _shift_stack_j_samples(self, j_samp, safemode=True):
    #     """
    #     Shift and trim the contents of self.stack by j_samp samples
    #     with a negative j_samp value indicating a left shift and
    #     a positive j_samp value indicating a right shift.

    #     Provides a safemode option (default True) that prevents
    #     fully shifting contents of self.stack out of range

    #     :: INPUTS ::
    #     :param j_samp: [int] number of samples to shift the contents of
    #                     self.stack by over axis 1 (k-axis), adhering to
    #                     the [i, j] axis structure of individual prediction
    #                     windows with i-axis for label type and j-axis for
    #                     data point indexing
    #     :param safemode: [bool] True - raise warning of j_samp > self.stack.shape[1]
    #                             False - overwrite self.stack with a np.nan-filled
    #                                     array matching self.stack.shape.
    #     :: UPDATE ::
    #     :attr stack: 
        
    #     :: OUTPUT ::
    #     :return self:
    #     """

    #     if not isinstance(j_samp, int):
    #         raise TypeError('j_samp must be type int')
    #     elif abs(j_samp) > self.stack.shape[1]:
    #         if safemode:
    #             raise ValueError(f'j_samp step is larger than stack length ({self.stack.shape[1]})')
    #         else:
    #             self.stack = np.full(shape=self.stack.shape, fill_value=np.nan, dtype=np.float32)
    #             return self
    #     else:
    #         pass
        
    #     tmp_stack = np.full(shape=self.stack.shape, fill_value=np.nan, dtype=np.float32)
    #     # Left shift along j-axis
    #     if j_samp < 0:
    #         tmp_stack[:, :-j_samp] = self.stack[:, j_samp:]
    #     # Right shift along j-axis
    #     elif j_samp > 0:
    #         tmp_stack[:, j_samp:] = self.stack[:, :-j_samp]
    #     # 0-shift handling
    #     else:
    #         tmp_stack = self.stack.copy()
    #     # Update
    #     self.stack = tmp_stack
    #     return self


    # def _shift_fold_i_samples(self, i_samp, safemode=True):
    #     """
    #     Shift the contents of the self.fold vector by i_samp with
    #     a negative i_samp value indicating a leftward shift of contents
    #     and a positive i_samp value indicating a rightward shift.

    #     Provides a safemode (default True) that prevents applying
    #     shifts that would completely move 
    #     """
    #     if not isinstance(i_samp, int):
    #         raise TypeError('i_samp must be type int')
    #     elif abs(i_samp) > self.fold.shape[0]:
    #         if safemode:
    #             raise ValueError(f'i_samp step is larger than fold length ({self.stack.shape[0]})')
    #         else:
    #             self.fold = np.full(shape=self.fold.shape, fill_value=np.nan, dtype=np.float32)
    #     else:
    #         pass
        
    #     tmp_fold = np.full(shape=self.fold.shape, fill_value=np.nan, dtype=np.float32)
    #     # Left shift along i-axis
    #     if i_samp < 0:
    #         tmp_fold[:-i_samp] = self.fold[i_samp:]
    #     # Right shift along i-axis
    #     elif i_samp > 0:
    #         tmp_fold[i_samp:] = self.fold[:-i_samp]
    #     # 0-shift handling
    #     else:
    #         tmp_fold = self.fold.copy()

    #     self.fold = tmp_fold
    #     return self




    # def initial_append(self, pred, meta, model, wgt_name, include_blinding=False):
    #     """
    #     Append the first data and metadata to this RtPredBuff and populate
    #     most of its attributes.

    #     :: INPUTS ::
    #     :param pred: [numpy.ndarray] or [torch.Tensor]
    #                 predicted value array output from SeisBench-hosted
    #                 PyTorch model (a WaveformModel-child class)
    #     :param meta: [dict]
    #                 Dictionary containing metadata associated with
    #                 pred
    #     :param model: [seisbench.models.WaveformModel]
    #                 Model object used to generate pred
    #     :param wgt_name: [str]
    #                 Shorthand name for the training weight set loaded
    #                 in `model`.
    #     """
        
    #     # If data have already been appended - raise error
    #     if self._has_data:
    #         raise RuntimeError('This RtPredBuff already contains data - canceling initial_append')
        
    #     else:
    #         # Run validation checks on metadata
    #         vmeta = self.validate_metadata(meta)
    #         # Run validation checks on input model
    #         if not isinstance(wgt_name, str):
    #             raise TypeError('"wgt_name" must be type str')
    #         else:
    #             self.weight_name = wgt_name
    #         if not isinstance(model, sbm.WaveformModel):
    #             raise TypeError('input "model" must be a seisbench.models.WaveformModel class object')
    #         else:
    #             self.label_names = model.labels
    #             self.label_codes = [_l[0].upper() for _l in model.labels]
                
    #         # Assign attribute values from metadata contents
    #         self._meta2attrib(vmeta)
    #         # Validate prediction
    #         vpred = self.validate_prediction(pred)
    #         # Insert prediction into stack
    #         if include_blinding:
    #             self.stack[:, :self.window_npts] = vpred * self.blinding_array
    #             # Place information into fold
    #             self.fold[:self.window_npts] = self.blinding_array
    #         else:
    #             nbidx = self.window_npts - 2*self.blinding_npts
    #             self.stack[:, :nbidx] = vpred[:, self.blinding_npts:-self.blinding_npts]
    #             self.fold[:nbidx] = 1
    #         # Update _has_data flag
    #         self._has_data = True
    #     return self



        # # If data need to left shift (future append)
        # elif vmeta['delta i'] < 0:
            





        #             self.stack[:, vmeta['i0']:vmeta['i1']] = substack
                
        #         elif self.stack_method == 'avg': 
        #             for _i in range

        #         raise IndexError('proposed append would shift the leading edge of stack/fold data outside the bounds of preallocated space')



        # # Raise error if attempting a duplicate position append
        # if vmeta['relative_pos'] == 0:
        #     raise IndexError('proposed append appears to be repeat data - canceling append')
        # # Appending future data
        # elif vmeta['relative_pos'] > 0:
        #     # If future append fits in current buffer
        #     if vmeta['relative_pos'] + self.window_npts < len(self.fold):
        #         append_start = vmeta['relative_pos']
        #     else:
        #         shift_npts = len(self.fold) - vmeta['relative_pos'] + self.window_npts - 1
        #         append_start = vmeta['relative_pos'] - shift_npts

        # # Assess indexing/storage for a past data append 
        # elif vmeta['relative_pos'] < 0:
        #     last_non_nan = np.max(np.argwhere(np.isfinite(self.stack)))
        #     if last_non_nan - vmeta['relative_pos'] < self.max_length_npts:
        #         pass
        #     else:
        # else:
        #     # Appending future data
        #     pass
        
        # # If checks are passed, shift, trim, and append data to stack and update fold
        # # First, run shift/trim
        # breakpoint()


        # self._shift_stack_j_samples(-vmeta['relative_pos'], safemode=safemode)
        # self._shift_fold_i_samples(-vmeta['relative_pos'], safemode=safemode)
        # blinded_pred = vpred * self.blinding_array
            
        # # Max Stack method
        # if self.stack_method == 'max':
        #     substack = self.stack[:,:self.window_npts].copy()
        #     for _i in range(len(self.label_codes)):
        #         for _j in range(self.window_npts):
        #             substack[_i, _j] = np.nanmax([substack[_i, _j], blinded_pred[_i, _j]])
        
        # # Average Stack method
        # elif self.stack_method == 'avg':
        #     # Get subset of fold vector
        #     subfold = self.fold[:self.window_npts]
        #     # Get subset of stack array
        #     substack = self.stack[:, :self.window_npts].copy()
        #     breakpoint()
        #     # Multiply substack elements by subfold values 
        #     substack *= subfold
        #     # Add blinded prediction values
        #     substack += blinded_pred
        #     # Iterate across each element of substack
        #     for _i in range(len(self.label_codes)):
        #         for _j in range(self.window_npts):
        #             # Get new data fold for this element
        #             new_subfold_samp = np.nansum([subfold[_j], self.blinding_array[_j]])
        #             # If fold is 0, assign 0 value to stack (work-around for div by 0 error)
        #             if new_subfold_samp == 0:
        #                 substack[_i, _j] = 0
        #             # Otherwise calculated updated average value
        #             else:
        #                 substack[_i, _j] /= new_subfold_samp
        # breakpoint()
        # # Update stack
        # self.stack[:, :self.window_npts] = substack
        # # Update fold
        # self.fold[:self.window_npts] += self.blinding_array



    # def model_stacking_fold(self):
    #     """
    #     Create a model of data stacking fold using pre-defined
    #     window length, blinding samples, and window overlaps.


    #     :def fold: number of non-blinded samples that share 
    #                 a given time index from overlapping prediction
    #                 windows
        
    #     NOTE:   Depending on the blinding and overlap parameters provided,
    #             there may be some saw-toothing in the maximum number of unblinded
    #             samples from consecutive windows, as such we calculate both the 
    #             maximum data fold (max_fold) attained and the maximum continuous
    #             data fold (cont_fold) values. 
        
    #     :: INPUTS ::
    #     :self: 
    #     :: OUTPUTS ::
    #     :return max_fold: [int] maximum fold value attained 
    #                     by stacking of _burnin_ct + 1 + n
    #                     consecutive prediction windows
    #     :return cont_fold: [int] maximum continuously maintained
    #                     fold value from stacking _burnin_ct + 1 + n
    #                     consecutive prediction windows

    #     NOTE: Generally, max_fold - cont_fold \in [0, 1]
    #     """
         
    #     # Construct single window weight vector
    #     self._wind_wgts = np.ones(self.window)
    #     self._wind_wgts[:self.blinding[0]] = 0
    #     self._wind_wgts[-self.blinding[1]:] = 0
    #     # Iterate across two window lengths
    #     i_, j_ = 0, self.window
    #     model_stack = np.ones(self.window*(2*self._burnin_stack_ct + 1))
    #     while i_ < len(model_stack):
    #         if j_ < len(model_stack):
    #             model_stack[i_:j_] += self._wind_wgts
    #         else:
    #             model_stack[i_:] += self._wind_wgts[:len(model_stack)-i_]
    #         i_ += (self.window - self.overlap)
    #         j_ = i_ + self.window

    #     pi_ = self._burnin_stack_ct*self.window
    #     pj_ = pi_+self.window
    #     stacj_sample = model_stack[pi_:pj_]
    #     max_fold = np.max(stacj_sample)
    #     cont_fold = np.min(stacj_sample)

    #     return max_fold, cont_fold












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
        