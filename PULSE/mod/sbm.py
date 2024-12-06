import typing
from collections import deque
import numpy as np
import seisbench.models as sbm
import torch

from PULSE.mod.base import BaseMod
from PULSE.data.foldtrace import FoldTrace
from PULSE.data.window import Window
from PULSE.data.dictstream import DictStream

class SBMMod(BaseMod):
    """This :class:`~.BaseMod` child class hosts a single :class:`~seisbench.models.WaveformModel`
    architecture with one or more pretrained weights and orchestrates model predictions using
    inputs of pre-processed :class:`~.Window` objects. Like the :meth:`~.sbm.WaveformModel.annotate`
    method, this class facilitates metadata book-keeping around the prediction step, generating
    :class:`~.DictStream` objects that share the same source :class:`~.Window` object, and have
    updated metadata reflecting the model architecture, weight, and prediction type.

    This multi-weight-prediction functionality is based on the methods of :cite:`Yuan2023`, who
    demonstrated its computational efficiency as part of their assessment of prediction semblance
    efficacy.

    Parameters
    ----------
    :param model: full import name of the model class to use,
        defaults to 'seisbench.models.EQTransformer'
    :type model: str, optional
    :param weight_names: list of pretrained weight names applicable
        to the model being imported, defaults to ['pnw']
    :type weight_names: list, optional
    :param compiled: should the models be precompiled, defaults to False
    :type compiled: bool, optional
    :param device: name of the device the model will be run on, defaults to 'cpu'
        see :class:`~torch.device`
    :type device: str, optional
    :param batch_sizes: minimum and maximum batch size for each iteration
        of the pulse method, defaults to None
        None sets the minimum batch size to 1 and maximum batch size
        to the batch_size in the **model._annotate_args** entry
        Tuple input must be a 2-tuple, e.g., (1, 128)
    :type batch_sizes: NoneType or 2-tuple of int, optional
    :param max_pulse_size: maximum number of iterations per call of
        :meth:`~.SBMMod.pulse`, defaults to 1
    :type max_pulse_size: int, optional
    :param maxlen: maximum size of the output deque, defaults to None
    :type maxlen: None or int, optional
    :param name: suffix to add to the **name** attribute of this
        mod, defaults to None
    :type name: str, int, NoneType, optional
    """    
    def __init__(
            self,
            model=sbm.EQTransformer(),
            weight_names=['pnw'],
            compiled=False,
            device='cpu',
            batch_sizes=None,
            max_pulse_size=8,
            maxlen=None,
            name=None):
        """Initialize a :class:`~.SBMMod` object

        :param model: full import name of the model class to use,
            defaults to 'seisbench.models.EQTransformer'
        :type model: str, optional
        :param weight_names: list of pretrained weight names applicable
            to the model being imported, defaults to ['pnw']
        :type weight_names: list, optional
        :param compiled: should the models be precompiled, defaults to False
        :type compiled: bool, optional
        :param device: name of the device the model will be run on, defaults to 'cpu'
            see :class:`~torch.device`
        :type device: str, optional
        :param batch_sizes: minimum and maximum batch size for each iteration
            of the pulse method, defaults to None
            None sets the minimum batch size to 1 and maximum batch size
            to the batch_size in the **model._annotate_args** entry
            Tuple input must be a 2-tuple, e.g., (1, 128)
        :type batch_sizes: NoneType or 2-tuple of int, optional
        :param max_pulse_size: maximum number of iterations per call of
            :meth:`~.SBMMod.pulse`, defaults to 1
        :type max_pulse_size: int, optional
        :param maxlen: maximum size of the output deque, defaults to None
        :type maxlen: None or int, optional
        :param name: suffix to add to the **name** attribute of this
            mod, defaults to None
        :type name: str, int, NoneType, optional
        """        
        # Inherit from BaseMod
        super().__init__(max_pulse_size=max_pulse_size, maxlen=maxlen, name=name)

        # Compatability check for model
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError('model must be type seisbench.model.WaveformModel')
        elif model.name == 'WaveformModel':
            raise TypeError('model must be a child-class of seisbench.models.WaveformModel')
        elif model.name not in ['EQTransformer','PhaseNet']:
            raise NotImplementedError('PULSE.mod.sbm.SBMMod currently supports only EQTransformer and PhaseNet')
        else:
            self.model = model

        self.name = f'{self.name}_{self.model.name}'

        pretrained = self.model.list_pretrained()

        # Compatability check for model weight_names
        if isinstance(weight_names, str):
            weight_names = [weight_names]
        
        for _wn in weight_names:
            if _wn not in pretrained:
                raise ValueError(f'weight {weight_names} is not a pretrained weight for {self.model.name}')
        self.weight_names = weight_names


        # Compatability check for device
        try:
            self.device = torch.device(device)
        except RuntimeError:
            raise
        
        ## THIS SHOULD REALLY BE SET OUTSIDE THIS CLASS
        # # Parse thread_limit
        # max_threads = torch.get_num_threads()
        # if thread_limit == 'half':
        #     torch_threads = max_threads//2
        # elif thread_limit is None:
        #     torch_threads = max_threads
        # elif isinstance(thread_limit, (int, float)):
        #     if max_threads >= int(thread_limit) > 0:
        #         torch_threads = int(thread_limit)
        #     else:
        #         self.Logger.warning(f'specified thread_limit {thread_limit} outside hardware bounds - using {max_threads//2}')
        #         torch_threads = max_threads//2
        # else:
        #     raise ValueError(f'thread_limit "{thread_limit}" not supported')
        # self.Logger.info(f'Set torch thread limit to {torch_threads}')
        # torch.set_num_threads(torch_threads)

        # Parse batch_sizes
        if batch_sizes is None:
            self.batch_sizes = (1, self.model._annotate_args['batch_size'][1])
        elif not isinstance(batch_sizes, tuple):
            raise TypeError('batch_sizes must be type tuple')
        elif len(batch_sizes) != 2:
            raise ValueError('batch_sizes must have 2 elements')
        elif not all(isinstance(_e, int) for _e in batch_sizes):
            raise TypeError('batch_sizes entries must be type int')
        elif not all(_e > 0 for _e in batch_sizes):
            raise ValueError('batch_sizes entries must be non-negative')
        else:
            self.batch_sizes = (min(batch_sizes),
                                max(batch_sizes))
        
        # Load Pretrained Models
        self.cmods = {_wn: self.model.from_pretrained(_wn) for _wn in weight_names}

        # Compile model(s) if specified
        if compiled:
            for _v in self.cmods.values():
                _v = torch.compile(_v.to(self.device))


    def get_unit_input(self, input: deque) -> typing.Union[dict,None]:
        """Detach :class:`~.Window` objects from **input**
        and use their data, fold, and metadata to create
        a batch input for a model prediction. This method
        triggers early stopping if there are an insufficient
        number of objects in **input** to meet the minimum
        batch size (this is specified with the **batch_sizes** attribute/
        argument in :meth:`~.SBMMod.__init__`).

        POLYMORPHIC: last updated with :class:`~.SBMMod`

        :param input: double-ended queue of :class:`~.Window` objects
        :type input: deque

        :returns: **unit_input** (*dict* or *None*) -- dictionary with:
         - 'data': a `batch_size` x `components` x `samples` numpy.ndarray
               that contains preprocessed waveform data
         - 'fold': a `batch_size` x `samples` numpy.ndarray that contains
                the summed fold of each window in 'data'
         - 'meta': a list containing a copy of the **stats** attribute object
                of the primary component from each :class:`~.Window` processed
        """        
        batch_size = min((self.measure_input(input), self.batch_sizes[1]))
        # If batch_size is too small, return none and signal early stopping
        if batch_size < self.batch_sizes[0]:
            self._continue_pulsing = False
            return None
        # Otherwise compose unit_input dictionary
        unit_input = {'data': np.full(shape=(batch_size, self.model.in_channels, self.model.in_samples),
                                      fill_value=0., dtype=np.float32),
                      'fold': np.full(shape=(batch_size, self.model.in_samples),
                                      fill_value=0., dtype=np.float32),
                      'meta': []}
                             
        # Iterate to pop off a batch of windows
        for _e in range(batch_size):
            if len(input) == 0:
                break
            # Get window from input deque
            window = input.pop()
            # Confirm necessary attributes
            if not isinstance(window, Window):
                raise TypeError(f'input presented a {type(window)} object, expected PULSE.data.window.Window')
            # Ensure window targets match model
            if window.stats.target_sampling_rate != self.model.sampling_rate:
                raise AttributeError(f'target_sampling_rate != model.sampling_rate')
            if window.stats.target_npts != self.model.in_samples:
                raise AttributeError(f'in_samples != target_npts')
            # Check component order
            if all(_e in window.keys for _e in self.model.component_order):
                order = self.model.component_order
            else:
                order = None
            # Compose numpy tensor entry
            unit_input['data'][_e, :, :] = window.to_npy_tensor(components=order)
            # Compose collapsed fold entry
            unit_input['fold'][_e, :] = window.collapse_fold(components=order)
            # Appendleft metadata to mdq
            unit_input['meta'].append(window.primary.stats.copy())

        return unit_input
    
    def run_unit_process(self, unit_input: dict) -> dict:
        """Run prediction(s) on a batch of pre-processed waveform
        data, allowing for multiple model weights to be applied
        to the same pre-processed data. Returns an updated
        **unit_input** with an added entry 'pred' that hosts
        a dictionary keyed with pretrained weight names and values
        that are the associated predictions

        POLYMORPHIC: last updated with :class:`~.SBMMod`

        :param unit_input: dictionary of data, metadata, and fold
            from a batch of windows
        :type unit_input: dict
        :returns: **unit_output** (*dict*) -- updated **unit_input**
            that has a 'pred' entry as described above
        """        
        # If passed a None (somehow _continue_pulsing didn't kick to OFF)
        if not isinstance(unit_input, dict):
            raise TypeError(f'unit_input must be type dict. Got type "{type(unit_input)}"')
        # Map unit_input to unit_output
        unit_output = unit_input

        
        # Create holder dictionary for predictions
        unit_output.update({'pred': {}})

        # Convert data to torch.Tensor
        data = torch.Tensor(unit_output['data'])
                
        # Iterate across model weights
        for _wn, _mod in self.cmods.items():
            # RUN PREDICTION
            if data.device.type != self.device:
                raw_preds = _mod(data.to(self.device))
            else:
                raw_preds = _mod(data)
            # OFFLOAD PREDICTIONS
            preds = np.full(shape=data.shape, fill_value=np.nan, dtype=np.float32)
            if self.model.name == 'EQTransformer':
                for _e, _p in enumerate(raw_preds):
                    if _p.device.type != 'cpu':
                        preds[:,_e,:] = _p.detach().cpu().numpy()
                    else:
                        preds[:,_e,:] = _p.detach().numpy()
            elif self.model.name == 'PhaseNet':
                if preds.device.type != 'cpu':
                    preds[:,:,:] = raw_preds.detach().cpu().numpy()
                else:
                    preds[:,:,:] = raw_preds.detach().numpy()
            unit_output['pred'].update({_wn: preds})

        return unit_output
    
    def put_unit_output(self, unit_output: dict) -> None:
        """Reassociate metadata and fold information with
        predictions from :meth:`~.SBMMod.run_unit_process`
        and produce :class:`~.DictStream` objects that contain
        the predictions (and metadata) arising from an input
        :class:`~.Window` object

        POLYMORPHIC: last updated with :class:`~.SBMMod`

        :param unit_output: dictionary containing
         - 'data' - input waveform data presented to the ML model
         - 'meta' - metadata from the primary component of the source window
         - 'fold' - collapsed fold for each waveform window
         - 'pred' - predictions (sub-dictionary)
        :type unit_output: dict
        """        
        meta = unit_output['meta']
        fold = unit_output['fold']
        pred = unit_output['pred']
        # Iterate across window number
        for _wn in range(fold.shape[0]):
            # Get metadata
            _meta = meta[_wn]
            _fold = fold[_wn, :]
            out = DictStream()
            # Iterate across weight
            for _weight, _p in pred.items():
                for _l, _label in enumerate(self.model.labels):
                    # Get particular predictions
                    _pred = _p[_wn, _l, :]
                    # Update copy of metadata
                    __meta = _meta.copy()
                    __meta.channel = __meta.channel[:-1] + _label[0].upper()
                    __meta.model = self.model.name
                    __meta.weight = _weight
                    # Compose FoldTrace
                    _ft = FoldTrace(data=_pred, fold=_fold,
                                    header=__meta)
                    # Append to out
                    out.extend(_ft)
            # Attach out to output
            self.output.appendleft(out)

                    

