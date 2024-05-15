"""
:module: wyrm.processing.mldetect
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module hosts class definitions for a Wyrm class that operates PyTorch
    based earthquake body phase detector/labeler models.

    PredictWyrm - a submodule for runing predictions with a particular PyTorch/SeisBench model
                architecture with pretrained weight(s) on preprocessed waveform data
                    PULSE:
                        input: deque of preprocessed WindowStream objects
                        output: deque of MLTrace objects
"""

import torch, copy
import numpy as np
import seisbench.models as sbm
from collections import deque
from ayahos.core.trace.mltrace import MLTrace
from ayahos.core.stream.dictstream import DictStream
from ayahos.core.stream.windowstream import WindowStream
from ayahos.core.wyrms.wyrm import Wyrm, add_class_name_to_docstring


###################################################################################
# MLDETECT WYRM CLASS DEFINITION - FOR BATCHED PREDICTION IN A PULSED MANNER ####
###################################################################################

@add_class_name_to_docstring
class MLDetectWyrm(Wyrm):
    """
    Conduct ML model predictions on preprocessed data ingested as a deque of
    WindowStream objects using one or more pretrained model weights. Following
    guidance on model application acceleration from SeisBench, an option to precompile
    models on the target device is included as a default option.

    This Wyrm's pulse() method accepts a deque of preprocessed WindowStream objects
    and outputs to another deque (self.queue) of MLTrace objects that contain
    windowed predictions, source-metadata, and fold values that are the sum of the
    input data fold vectors
        i.e., data with all 3 data channels has predictions with fold = 3 for all elements, 
              whereas data with both horizontal channels missing produce predictions with 
              fold = 1 for all elements. Consequently data with gaps may have fold values
              ranging \in [0, 3]

    This functionality allows tracking of information density across the prediction stage
    of a processing pipeline.
    """


    def __init__(
        self,
        model=sbm.EQTransformer(),
        weight_names=['pnw',
                      'instance',
                      'stead'],
        devicetype='cpu',
        compiled=True,
        max_batch_size=256,
        max_pulse_size=1):
        """
        Initialize a ayahos.core.wyrms.mldetectwyrm.MLDetectWyrm object

        :: INPUTS ::
        :param model: seisbench WaveformModel child class object, default is seisbench.models.EQTransformer()
        :type model: seisbench.models.WaveformModel
        :param weight_names: names of pretrained model weights included in the model.list_pretrained() output
                        default is ['pnw','instance','stead']
                        NOTE: This object holds distinct, in-memory instances of
                            all model-weight combinations, allowing rapid cycling
                            across weights and storage of pre-compiled models
        :type weight_names: list-like of str or str
        :param devicetype: name of a device compliant with a torch.device(), default is 'cpu'
                                (e.g., on Apple M1/2 'mps' becomes an option)
        :type devicetype: str
        :param compiled: should the model(s) be precompiled on initialization using the torch.compile() method?, default is False
                        NOTE: This is suggested in the SeisBench documentation as
                            a way to accelerate model application
        :type compiled: bool
        :param max_batch_size: maximum batch size for windowed data in each pulse, default is 256
        :type max_batch_size: int
        :param max_pulse_size: maximum number of iterations (batches) to run per pulse, default is 1
        :type max_pulse_size: int
        """
        super().__init__(max_pulse_size=max_pulse_size)
        
        # max_batch_size compatability checks
        if isinstance(max_batch_size, int):
            if 0 < max_batch_size <= 2**15:
                self.max_batch_size = max_batch_size
            else:
                raise ValueError(f'max_batch_size {max_batch_size} falls out of bounds should be \in(0, 2**15]')
        else:
            raise TypeError(f'max_batch_size must be type int, not {type(max_batch_size)}')

        # model compatability checks
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError('model must be a seisbench.models.WaveformModel object')
        elif model.name == 'WaveformModel':
            raise TypeError('model must be a child-class of the seisbench.models.WaveformModel class')
        else:
            self.model = model
        
        # Model weight_names compatability checks
        # pretrained_list = model.list_pretrained() # This error catch is now handled with the preload/precopile setp
        if isinstance(weight_names, str):
            weight_names = [weight_names]
        elif isinstance(weight_names, (list, tuple)):
            if not all(isinstance(_n, str) for _n in weight_names):
                raise TypeError('not all listed weight_names are type str')
        # else:
        #     for _n in weight_names:
        #         if _n not in pretrained_list:
        #             raise ValueError(f'weight_name {_n} is not a valid pretrained model weight_name for {model}')
        self.weight_names = weight_names

        # device compatability checks
        if not isinstance(devicetype, str):
            raise TypeError('devicetype must be type str')
        else:
            try:
                device = torch.device(devicetype)
            except RuntimeError:
                raise RuntimeError(f'devicetype {devicetype} is an invalid device string for PyTorch')
            try:
                self.model.to(device)
            except RuntimeError:
                raise RuntimeError(f'device type {devicetype} is unavailable on this installation')
            self.device = device
        
        # Preload/precompile model-weight combinations
        if isinstance(compiled, bool):    
            self.compiled = compiled
        else:
            raise TypeError(f'"compiled" type {type(compiled)} not supported. Must be type bool')

        self.cmods = {}
        for wname in self.weight_names:
            if self.debug:
                print(f'Loading {self.model.name} - {wname}')
            cmod = self.model.from_pretrained(wname)
            if compiled:
                if self.debug:
                    print(f'...pre compiling model on device type "{self.device.type}"')
                cmod = torch.compile(cmod.to(self.device))
            else:
                cmod = cmod.to(self.device)
            self.cmods.update({wname: cmod})


    # def __str__(self):
    #     rstr = f'ayahos.core.wyrms.mldetectwyrm.MLDetectWyrm('
    #     rstr += f'model=sbm.{self.model.name}, weight_names={self.weight_names}, '
    #     rstr += f'devicetype={self.device.type}, compiled={self.compiled}, '
    #     rstr += f'max_pulse_size={self.max_pulse_size}, debug={self.debug})'
    #     return rstr

    #################################
    # PULSE POLYMORPHIC SUBROUTINES #
    #################################

    # def _continue_iteration(self, stdin, iterno):
    #     if len(stdin) == 0:

    # Inherit from Wyrm
    # _continue_iteration() - stdin must be a non-empty deque and iterno +1 < len(stdin)

    def _get_obj_from_input(self, stdin):
        """ _get_obj_from_input method for MLDetectWyrm

        Create batched window data for input to ML prediction

        :param stdin: collection of input objects
        :type stdin: collections.deque of ayahos.core.stream.windowstream.WindowStream(s)
        :return: batch_data, batch_fold, and batch_meta objects
        :rtype: 3-tuple
        """        
        if not isinstance(stdin, deque):
            raise TypeError('input `obj` must be type collections.deque')        

        batch_data = []
        batch_fold = []
        batch_meta = []
        # Compose Batch
        for j_ in range(self.max_batch_size):
            # Check if there are still objects to assess (inherited from Wyrm)
            status = super()._continue_iteration(stdin, j_)
            # If there are 
            if status:
                _x = stdin.popleft()
                if not isinstance(_x, WindowStream):
                    self.logger.critical('type mismatch')
                    raise TypeError
                # Check if windowstream is ready for conversion to torch.Tensor
                if _x.ready_to_burn(self.model):
                    # Get data tensor
                    _data = _x.to_npy_tensor(self.model).copy()
                    # Get data fold vector
                    _fold = _x.collapse_fold().copy()
                    # Get WindowStream metadata
                    _meta = _x.stats.copy()
                    # Explicitly delete the source window from memory
                    del _x
                    # Append coppied (meta)data to batch collectors
                    batch_data.append(_data)
                    batch_fold.append(_fold)
                    batch_meta.append(_meta)
                else:
                    self.logger.error(f'WindowStream for {_x.stats.common_id} is not sufficiently processed - skipping')
                    pass
            # If we've run out of objects to assess, stop creating batch
            else:
                break
        obj = (batch_data, batch_fold, batch_meta)
        return obj
    
    def _unit_process(self, obj):
        """unit_process of ayahos.core.wyrms.mldetectwyrm.MLDetectWyrm

        This unit process batches data, runs predictions, reassociates
        predicted values and their source metadata, and attaches prediction
        containing objects to the output attribute

        :param obj: tuple containing batched data, fold, and metadata objects
        :type obj: 3-tuple
        :return unit_out: unit process output
        :rtype unit_out: dict of ayahos.core.stream.dictstream.DictStream objects
        """
        # unpack obj
        batch_data, batch_fold, batch_meta = obj
        # If we have at least one tensor to predict on, proceed
        if len(batch_data) > 0:
            # Convert list of 2d numpy.ndarrays into a 3d numpy.ndarray
            batch_data = np.array(batch_data)
            # Catch case where we have a single window (add the window axis)
            if batch_data.ndim == 2:
                batch_data = batch_data[np.newaxis, :, :]
            # Convert int
            batch_data = torch.Tensor(batch_data)
            # Create output holder for all predictions
            unit_out = {i_: DictStream() for i_ in range(len(batch_meta))}
            # Iterate across preloaded (possibly precompiled) models
            for wname, weighted_model in self.cmods.items():
                # RUN PREDICTION
                batch_pred = self.__run_prediction(weighted_model, batch_data, batch_meta)
                # Reassociate metadata
                self.__batch2dst_dict(wname, batch_pred, batch_fold, batch_meta, unit_out)
        return unit_out
    
    def _capture_unit_out(self, unit_out): 
        """_capture_unit_out

        Iterate across DictStreams in unit_out and append each to the output attribute

        :param unit_out: unit output from _unit_out
        :type unit_out: dict of ayahos.core.stream.dictstream.DictStream objects
        """                       
        # Attach DictStreams to output
        for _v in unit_out.values():
            self.output.append(_v)

    #############################
    # _unit_process subroutines #
    #############################
    def __run_prediction(self, weighted_model, batch_data, reshape_output=True):
        """
        Run a prediction on an input batch of windowed data using a specified model on
        self.device. Provides checks that batch_data is on self.device and an option to
        enforce a uniform shape of batch_preds and batch_data.

        :: INPUT ::
        :param weighted_model: ML model with pretrained weights loaded (and potentialy precompiled) with
        :type weighted_model: seisbench.models.WaveformModel
        :param batch_data: data array with scaling appropriate to the input layer of `weighed_model` 
        :type batch_data: torch.Tensor or numpy.ndarray
        :param reshape_output: if batch_preds has a different shape from batch_data, should batch_preds be reshaped to match?
        :type reshape_output: bool
        :return detached_batch_preds: prediction outputs, detached from non-cpu processor if applicable
        :rtype detached_batch_preds: numpy.ndarray
        """
        # Ensure input data is a torch.tensor
        if not isinstance(batch_data, (torch.Tensor, np.ndarray)):
            raise TypeError('batch_data must be type torch.Tensor or numpy.ndarray')
        elif isinstance(batch_data, np.ndarray):
            batch_data = torch.Tensor(batch_data)

        # RUN PREDICTION, ensuring data is on self.device
        if batch_data.device.type != self.device.type:
            batch_preds = weighted_model(batch_data.to(self.device))
        else:
            batch_preds = weighted_model(batch_data)

        # If operating on EQTransformer
        nwind = batch_data.shape[0]
        nlbl = len(self.model.labels)
        nsmp = self.model.in_samples
        if self.model.name == 'EQTransformer':
            detached_batch_preds= np.full(shape=(nwind, nlbl, nsmp), fill_value=np.nan, dtype=np.float32)
            for _l, _p in enumerate(batch_preds):
                if _p.device.type != 'cpu': 
                    detached_batch_preds[:, _l, :] = _p.detach().cpu().numpy()
                else:
                    detached_batch_preds[:, _l, :] = _p.detach().numpy()
        elif self.model.name == 'PhaseNet':
            if batch_preds.device.type != 'cpu':
                detached_batch_preds = batch_preds.detach().cpu().numpy() 
            else:
                detached_batch_preds = batch_preds.detach().numpy()
        else:
            self.logger.critical(f'model "{self.model.name}" prediction initial unpacking not yet implemented')
            raise NotImplementedError
        # breakpoint()
        # # Check if output predictions are presented as some list-like of torch.Tensors
        # if isinstance(batch_preds, (tuple, list)):
        #     # If so, convert into a torch.Tensor
        #     if all(isinstance(_p, torch.Tensor) for _p in batch_preds):
        #         batch_preds = torch.concat(batch_preds)
        #     else:
        #         raise TypeError('not all elements of preds is type torch.Tensor')
        # # # If reshaping to batch_data.shape is desired, check if it is required.
        # # if reshape_output and batch_preds.shape != batch_data.shape:
        # #     batch_preds = batch_preds.reshape(batch_data.shape)

        return detached_batch_preds

    def __batch2dst_dict(self, weight_name, batch_preds, batch_fold, batch_meta, dst_dict):
        """
        Reassociated batched predictions, batched metadata, and model metadata to generate MLTrace objects
        that are appended to the output deque (self.queue). The following MLTrace ID elements are updated
            component = 1st letter of the model label (e.g., "Detection" from EQTranformer -> "D")
            model = model name
            weight = pretrained weight name
        

        :param weight_name: name of the pretrained model weight used
        :type weight_name: str
        :param batch_preds: predicted values with expected axis assignments:
                                axis 0: window # - corresponding to the axis 0 values in batch_fold and batch_meta
                                axis 1: label - label assignments from the model architecture used
                                axis 2: values
        :type batch_preds: torch.Tensor
        :param batch_fold: vectors of summed input data fold for each input window
        :type batch_fold: list of numpy.ndarray
        :param batch_meta: metadata corresponding to input data for each prediction window
        :type batch_meta: list of wyrm.core.WindowStream.WindowStreamStats
        :param dst_dict: prediction output holder object that will house reassociated (meta)data
        :type dst_dict: dict of ayahos.core.stream.dictstream.DictStream objects

        """
        # Reshape sanity check
        if batch_preds.ndim != 3:
            if batch_preds.shape[0] != len(batch_meta):
                batch_preds = batch_preds.reshape((len(batch_meta), -1, self.model.in_samples))

        # TODO - change metadata propagation to take procesing from component stream, but still keep
        # timing and whatnot from reference_streams
        # Iterate across metadata dictionaries
        for _i, _meta in enumerate(batch_meta):
            # Split reference code into components
            # breakpoint()
            n,s,l,c,m,w = _meta.common_id.split('.')
            # Generate new MLTrace header for this set of predictions
            _header = {'starttime': _meta.reference_starttime,
                      'sampling_rate': _meta.reference_sampling_rate,
                      'network': n,
                      'station': s,
                      'location': l,
                      'channel': c,
                      'model': m,
                      'weight': weight_name,
                      'processing': copy.deepcopy(_meta.processing)}
            # Update processing information to timestamp completion of batch prediction
     
            # _header['processing'].append([time.time(),
            #                               'Wyrm 0.0.0',
            #                               'PredictionWyrm',
            #                               'batch2dst_dict',
            #                               '<internal>'])
            # Iterate across prediction labels
            for _j, label in enumerate(self.cmods[weight_name].labels):
                # Compose output trace from prediction values, input data fold, and header data
                _mlt = MLTrace(data = batch_preds[_i, _j, :], fold=batch_fold[_i], header=_header)
                # Update component labeling
                _mlt.set_comp(label)
                # if self._timestamp:
                #     _mlt.stats.processing.append(['PredictionWyrm','batch2dst',f'{_i+1} of {len(batch_meta)}',time.time()])
                # Append to window-indexed dictionary of WyrmStream objects
                if _i not in dst_dict.keys():
                    dst_dict.update({_i, DictStream()})
                dst_dict[_i].__add__(_mlt, key_attr='id')
                # Add mltrace to dsbuffer (subsequent buffering to happen in the next step)
                



    # def pulse(self, x):
    #     """
    #     Execute a pulse on input deque of WindowStream objects `x`, predicting
    #     values for each model-weight-window combination and outputting individual
    #     predicted value traces as MLTrace objects in the self.queue attribute

    #     :: INPUT ::
    #     :param x: [deque] of [wyrm.core.WyrmStream.WindowStream] objects
    #                 objects must be 
        

    #     TODO: Eventually, have the predictions overwrite the windowed data
    #           values of the ingested WindowStream objects so predictions
    #           largely work as a in-place change
    #     """
    #     if not isinstance(x, deque):
    #         raise TypeError('input "x" must be type deque')
        
    #     qlen = len(x)
    #     # Initialize batch collectors for this pulse
    #     batch_data = []
    #     batch_fold = []
    #     batch_meta = []

    #     for _i in range(self.max_pulse_size):
    #         if len(x) == 0:
    #             break
    #         if _i == qlen:
    #             break
    #         else:
    #             _x = x.popleft()
    #             if not(isinstance(_x, WindowStream)):
    #                 x.append(_x)
    #             # Check that WindowStream is ready to split out, copy, and be eliminated
    #             if _x.ready_to_burn(self.model):
    #                 # Part out copied data, metadata, and fold objects
    #                 _data = _x.to_npy_tensor(self.model).copy()
    #                 _fold = _x.collapse_fold().copy() 
    #                 _meta = _x.stats.copy()
    #                 # Attach processing information for split
    #                 # _meta.processing.append([time.time(),
    #                 #                          'Wyrm 0.0.0',
    #                 #                          'PredictionWyrm',
    #                 #                          'split_for_ml',
    #                 #                          '<internal>'])
    #                 if self._timestamp:
    #                     _meta.processing.append(['PredictionWyrm','split_for_ml',str(_i), time.time()])
    #                 # Delete source WindowStream object to clean up memory
    #                 del _x
    #                 # Append copied (meta)data to collectors
    #                 batch_data.append(_data)
    #                 batch_fold.append(_fold)
    #                 batch_meta.append(_meta)
    #             # TODO: If not ready to burn, kick error
    #             else:
    #                 breakpoint()
    #                 raise ValueError('WindowStream is not sufficiently preprocessed - suspect an error earlier in the tube')
       
    #     # IF there are windows to process
    #     if len(batch_meta) > 0:
    #         # Concatenate batch_data tensor list into a single tensor
    #         batch_data = torch.Tensor(np.array(batch_data))
    #         batch_dst_dict = {_i: WyrmStream() for _i in range(len(batch_meta))}
    #         # Iterate across preloaded (and precompiled) models
    #         for wname, weighted_model in self.cmods.items():
    #             if self._timestamp:
    #                 batch_meta = batch_meta.copy()
    #                 for _meta in batch_meta:
    #                     _meta.processing.append(['PredictionWyrm','pulse','batch_start',time.time()])
    #             # Run batch prediction for a given weighted_model weight
    #             if batch_data.ndim != 3:
    #                 breakpoint()
    #             batch_pred = self.run_prediction(weighted_model, batch_data, batch_meta)
    #             # Reassociate window metadata to predicted values and send MLTraces to queue
    #             self.batch2dst_dict(wname, batch_pred, batch_fold, batch_meta, batch_dst_dict)
    #         # Provide access to queue as pulse output
    #         for _v in batch_dst_dict.values():
    #             self.queue.append(_v)

    #     # alias self.queue to output
    #     y = self.queue
    #     return y