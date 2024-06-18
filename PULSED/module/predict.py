import torch, copy, logging, sys
import numpy as np
import seisbench.models as sbm
from collections import deque
from obspy import UTCDateTime
from PULSED.data.mltrace import MLTrace
from PULSED.data.mlstream import MLStream
from PULSED.data.mlwindow import MLWindow
from PULSED.module._base import _BaseMod

Logger = logging.getLogger(__name__)
###################################################################################
# MLDETECT WYRM CLASS DEFINITION - FOR BATCHED PREDICTION IN A PULSED MANNER ####
###################################################################################

# @add_class_name_to_docstring
class SeisBenchMod(_BaseMod):
    """
    Conduct ML model predictions on preprocessed data ingested as a deque of
    MLWindow objects using one or more pretrained model weights. Following
    guidance on model application acceleration from SeisBench, an option to precompile
    models on the target device is included as a default option.

    This Wyrm's pulse() method accepts a deque of preprocessed MLWindow objects
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
        mclass,
        weight_names,
        devicetype='cpu',
        compiled=True,
        min_batch_size=1,
        max_batch_size=256,
        max_pulse_size=1,
        meta_memory=3600,
        max_output_size=1e9,
        report_period=False):
        """
        Initialize a PULSED.data.wyrms.mldetectwyrm.MLDetectWyrm object

        :: INPUTS ::
        :param model: seisbench WaveformModel child class object, e.g., seisbench.models.EQTransformer()
        :type model: seisbench.models.WaveformModel
        :param weight_names: names of pretrained model weights included in the model.list_pretrained() output
                        e.g., ['pnw','instance','stead']
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
        super().__init__(max_pulse_size=max_pulse_size,
                         meta_memory=meta_memory,
                         report_period=report_period,
                         max_output_size=max_output_size)
        
        # max_batch_size compatability checks
        if isinstance(max_batch_size, int):
            if 0 < max_batch_size <= 2**15:
                self.max_batch_size = max_batch_size
            else:
                raise ValueError(f'max_batch_size {max_batch_size} falls out of bounds should be \in(0, 2**15]')
        else:
            raise TypeError(f'max_batch_size must be type int, not {type(max_batch_size)}')

        if isinstance(min_batch_size, int):
            if 0 < min_batch_size < self.max_batch_size:
                self.min_batch_size = min_batch_size
            else:
                raise ValueError(f'min_batch_size {min_batch_size} not \in (0, {max_batch_size})')
        else:
            raise TypeError(f'min_batch_size must be type int, not {type(max_batch_size)}')

        # Load model class object
        model = self.import_class(mclass)()
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
            self.Logger.debug(f'Loading {self.model.name} - {wname}')
            cmod = self.model.from_pretrained(wname)
            if compiled:
                self.Logger.debug(f'...pre compiling model on device type "{self.device.type}"')
                cmod = torch.compile(cmod.to(self.device))
            else:
                cmod = cmod.to(self.device)
            self.cmods.update({wname: cmod})

        self.junk_drawer = deque()

    # def __str__(self):
    #     rstr = f'PULSED.data.wyrms.mldetectwyrm.MLDetectWyrm('
    #     rstr += f'model=sbm.{self.model.name}, weight_names={self.weight_names}, '
    #     rstr += f'devicetype={self.device.type}, compiled={self.compiled}, '
    #     rstr += f'max_pulse_size={self.max_pulse_size}, debug={self.debug})'
    #     return rstr

    #################################
    # PULSE POLYMORPHIC SUBROUTINES #
    #################################

    # def _continue_iteration(self, input, iterno):
    #     if len(input) == 0:

    # Inherit from Wyrm
    # _continue_iteration() - input must be a non-empty deque and iterno +1 < len(input)

    def _should_this_iteration_run(self, input, input_measure, iterno):
        status = False
        # if input is deque
        if isinstance(input, deque):
            # and input has at least min_batch_size elements
            if len(input) >= self.min_batch_size:
                # and iteration number + 1 is l.e. the length of input
                if iterno + 1 <= input_measure:
                    # Then proceed with iteration
                    status = True
        return status

    def _unit_input_from_input(self, input):
        """
        POLYMORPHIC
        Last update with :class:`~PULSED.wyrms.sbmwyrm.SBMWyrm`

        Create batched window data for input to ML prediction

        :param input: collection of input objects
        :type input: collections.deque of PULSED.data.MLWindow.MLWindow(s)
        :returns: 
            - **unit_input** (*3-tuple of lists*) -- tuple containing
                - **batch_data** -- batch of windowed, preprocessed data tensors
                - **batch_fold** -- fold information for each preprocessed data tensor window
                - **batch_meta** -- metadata associated with each window in **batch_data**
        :rtype: 3-tuple
        """        
        if not isinstance(input, deque):
            raise TypeError('input `obj` must be type collections.deque')        

        batch_data = []
        batch_fold = []
        batch_meta = []
        # Compose Batch
        measure = len(input)
        for j_ in range(self.max_batch_size):
            # Check if there are still objects to assess (inherited from Wyrm)
            status = super()._should_this_iteration_run(input, measure, j_)
            # If there are 
            if status:
                _x = input.popleft()
                if not isinstance(_x, MLWindow):
                    self.Logger.critical('type mismatch')
                    raise TypeError
                # Check if MLWindow is ready for conversion to torch.Tensor
                if _x.ready_to_burn(self.model):
                    # Get data tensor
                    _data = _x.to_npy_tensor(self.model).copy()
                    # Get data fold vector
                    _fold = _x.collapse_fold().copy()
                    # Get MLWindow metadata
                    _meta = _x.stats.copy()
                    _meta.processing.append([self.__name__(), 'batched', UTCDateTime()])
                    # Explicitly delete the source window from memory
                    del _x
                    # Append coppied (meta)data to batch collectors
                    batch_data.append(_data)
                    batch_fold.append(_fold)
                    batch_meta.append(_meta)
                else:
                    self.Logger.error(f'MLWindow for {_x.stats.common_id} is not sufficiently processed - skipping')
                    # self.junk_drawer.append(_x)
                    pass
            # If we've run out of objects to assess, stop creating batch
            else:
                break
        unit_input = (batch_data, batch_fold, batch_meta)
        return unit_input
    
    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last update with :class:`~PULSED.wyrms.sbmwyrm.SBMWyrm`

        This unit process batches data, runs predictions, reassociates
        predicted values and their source metadata, and attaches prediction
        containing objects to the output attribute

        :param unit_input: tuple containing batched data, fold, and metadata objects
        :type unit_input: (list of numpy.ndarray, list of numpy.ndarray, list of dict)
        :returns
            - **unit_output** (*dict of PULSED.data.mlstream.MLStream*) -- output predictions reassociated with their fold-/meta-data
        """
        # unpack unit_input
        batch_data, batch_fold, batch_meta = unit_input
        # Create holder for outputs
        unit_output = {'pred': {}, 'meta': batch_meta, 'fold': batch_fold}
        # If we have at least one tensor to predict on, proceed
        if len(batch_data) > 0:
            # self.Logger.info(f'prediction on batch of {len(batch_data)} windows')
            # Convert list of 2d numpy.ndarrays into a 3d numpy.ndarray
            if isinstance(batch_data, np.ndarray):
                batch_data = [batch_data]
            batch_data = torch.Tensor(batch_data)
                
            # batch_data = np.array(batch_data, dtype=np.float32)
            # # Catch case where we have a single window (add the window axis)
            # if batch_data.ndim == 2:
            #     batch_data = batch_data[np.newaxis, :, :]
            # # Convert into torch tensor
            # batch_data = torch.Tensor(batch_data)
            # Iterate across preloaded (possibly precompiled) models
            for wname, weighted_model in self.cmods.items():
                # RUN PREDICTION
                batch_pred = self.__run_prediction(weighted_model, batch_data, batch_meta)
                # Capture model-weight output
                unit_output['pred'].update({wname: batch_pred})
        else:
            unit_output = None
            del batch_data
        return unit_output
    
    def _capture_unit_output(self, unit_output):
        if unit_output is None:
            return
        else:
            bm = unit_output['meta']
            bf = unit_output['fold']
            for wname in self.cmods.keys():
                bp = unit_output['pred'][wname]
                if bp is None:
                    continue
                else:
                    pass
                _traces = []
                for _n, _meta in enumerate(bm):
                    n,s,l,c,m,w = _meta.common_id.split('.')
                    # Generate new general MLTrace header for this set of predictions
                    _header = {'starttime': _meta.reference_starttime,
                            'sampling_rate': _meta.reference_sampling_rate,
                            'network': n,
                            'station': s,
                            'location': l,
                            'channel': c,
                            'model': m,
                            'weight': wname,
                            'processing': copy.deepcopy(_meta.processing)}
                    for _o, _label in enumerate(self.cmods[wname].labels):
                        # Compose output trace from prediction values, input data fold, and header data
                        _mlt = MLTrace(data = bp[_n, _o, :], fold=bf[_n], header=_header)
                        # Update component labeling
                        _mlt.set_comp(_label)
                        _traces.append(_mlt)
                    mls = MLStream(traces=_traces,
                                   header=_header,
                                   key_attr='id')
                    mls.stats.processing.append([self.__name__(), 'debatched', UTCDateTime()])
                    
                    self.output.append(mls)

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

        nwind = batch_data.shape[0]
        nlbl = len(self.model.labels)
        nsmp = self.model.in_samples

        # If using EQTransformer
        if self.model.name == 'EQTransformer':
            detached_batch_preds= np.full(shape=(nwind, nlbl, nsmp), fill_value=np.nan, dtype=np.float32)
            for _l, _p in enumerate(batch_preds):
                if _p.device.type != 'cpu': 
                    detached_batch_preds[:, _l, :] = _p.detach().cpu().numpy()
                else:
                    detached_batch_preds[:, _l, :] = _p.detach().numpy()
        # If using PhaseNet (original)
        elif self.model.name == 'PhaseNet':
            if batch_preds.device.type != 'cpu':
                detached_batch_preds = batch_preds.detach().cpu().numpy() 
            else:
                detached_batch_preds = batch_preds.detach().numpy()
        # Safety catch
        else:
            self.Logger.critical(f'model "{self.model.name}" prediction initial unpacking not yet implemented')
            sys.exit(1)
        
        return detached_batch_preds
