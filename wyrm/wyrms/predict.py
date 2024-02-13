"""
:module: wyrm.wyrms.prediction
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:


"""

from collections import deque
from wyrm.wyrms.base import Wyrm
import wyrm.util.input_compatability_checks as icc
from wyrm.structures.window import InstWindow
from wyrm.buffer.prediction import PredBuff
from wyrm.buffer.structures import TieredBuffer
import seisbench.models as sbm
import numpy as np
import torch
from tqdm import tqdm
from time import time

class MachineWyrm(Wyrm):
    """
    Wyrm for housing and operating pulsed prediction by a WaveformModel from
    SeisBench and passing metadata from model inputs to outputs

    input - deque([tuple(tensor, metadata), tuple(tensor, metadata),...])
        with each tuple created as the output of a InstWindow.to_torch()
        instance
    output - TieredBuffer terminating in wyrm.buffer.prediction.PredBuff objects
            with tier-0 keys corresponding to instrument codes [N.S.L.bi] and
            tier-1 keys corresponding to model type. This tier-1 selection results
            in a single tier at this stage, but becomes useful when applying multiple
            model architectures to the same data
    """
    def __init__(self, model, weight_names, devicetype='cpu', max_samples=12000, max_pulse_size=1000, stack_method='max', debug=False):
        """
        Initialize a SeisBenchWyrm object

        :: INPUTS ::
        :param model: [seisbench.models.WaveformModel-like] a seisbench model
                        that has the WaveformModel parent class.
        :param weight_names: [str] or [list] model name(s) to pass to 
                              model.from_pretrained()
        :param device: [torch.device] device to run predictions on
        :param max_samples: [int] maximum prediction buffer length in samples
                        NOTE: __init__ enforces a 2-window minimum length based on 
                        the model.in_samples attribute.
        :param max_pulse_size: [int] maximum number of data windows to assess
                    for a pulse (synonymous with batch size)
        :param debug: [bool] run in debug mode?
        """
        # Initialize Wyrm inheritance
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)
        # model compatability checks
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError('model must be a seisbench.models.WaveformModel object')
        elif model.name == 'WaveformModel':
            raise TypeError('model must be a child-class of the seisbench.models.WaveformModel class')
        else:
            self.model = model
        # Map model labels to model codes
        self.label_codes = ''
        for _l in self.model.labels:
            self.label_codes += _l[0]

        # Model weight_names compatability checks
        pretrained_list = model.list_pretrained()
        if isinstance(weight_names, str):
            weight_names = [weight_names]
        elif isinstance(weight_names, (list, tuple)):
            if not all(isinstance(_n, str) for _n in weight_names):
                raise TypeError('not all listed weight_names are type str')
        else:
            for _n in weight_names:
                if _n not in pretrained_list:
                    raise ValueError(f'weight_name {_n} is not a valid pretrained model weight_name for {model}')
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

        # max_samples compatability checks
        self.max_samples = icc.bounded_floatlike(
            max_samples,
            name='max_samples',
            minimum=2*self.model.in_samples,
            maximum=None,
            inclusive=False
        )

        # stack_method compatability checks
        if not isinstance(stack_method, str):
            raise TypeError('stack_method must be type str')
        elif stack_method.lower() in ['max','maximum']:
            self.stack_method='max'
        elif stack_method.lower() in ['avg','mean']:
            self.stack_method='avg'
        else:
            raise ValueError(f'stack_method {stack_method} not supported. Must be "max", "avg" or select aliases')
      
        # Initialize TieredBuffer terminating in PredBuff objects
        self.buffer = TieredBuffer(
            buff_class=PredBuff,
            model=self.model,
            weight_names=self.weight_names,
            max_samples=self.max_samples,
            stack_method=self.stack_method
            )

    def pulse(self, x):
        """
        :: INPUTS ::
        :param x: [deque] of InstWindow objects

        :: OUTPUT ::
        :return y: [RtInstStream] access to RtInstStream composed
                of RtPredTrace objects.
        """
        # Construct prediction inputs
        window_concat_tensor, meta_list = self._instwindows2tensor(x)
        # Preallocate numpy array for outputs
        pred_nwlt = np.full(shape=(len(self.nwgts),
                                   len(meta_list),
                                   len(self.label_codes),
                                   self.model.in_samples),
                            fill_value=0.,
                            dtype=np.float32)
        
        # Iterate across weights
        for _n, wname in enumerate(self.weight_names):
            if self.debug:
                tick = time()
                print('Starting {self.model_name}, {wname} weighted prediction')
            # Load n-th model weight
            self.model = self.model.from_pretrained(wname)
            # Run Prediction
            raw_preds = self._run_prediction(window_concat_tensor)
            # Append output to numpy array
            pred_nwlt[_n, ...] = self._raw_preds2numpy(window_concat_tensor, raw_preds)
            # Report runtime
            if self.debug:
                print(f'Prediction runtime ({time() - tick})')
            
            self._distribute_preds_to_buffer(pred_nwlt, meta_list)
            

            # Iterate across windows and append to tree
            for _i, _meta in enumerate(meta_list):
                # self._append_to_tree(_meta, npy_preds[_i, :, :], _n)
                self._append_to_predbuff_tree(_meta, npy_preds[_i, ...], _n)
                
        # self._sort_tree_indices()
        y = self.tree
        return y

    def _instwindows2tensor(self, x):
        """
        Convert a deque of InstWindow objects into
        a concatenated torch.Tensor of up to self.max_pulse_size
        windows of pre-processed instrument data

        :: INPUT ::
        :param x: [deque] of [wyrm.structures.window.InstWindow] objects
                  that have been pre-processed
        :: OUTPUTS ::
        :return window_concat_tensor: [torch.Tensor] with SeisBench dimensions
                                    (window#, component, data)
        :return meta_list: [list] of [dict] containing window information
                            that correspond with the 0-axis indexing of
                            window_concat_tensor
        """
        if not isinstance(x, deque):
            raise TypeError('input x must be a deque')
        # Concatenate input tensors and metadata
        tensor_list = []
        meta_list = []
        for _ in range(self.max_pulse_size):
            if len(x) > 0:
                _x = x.pop()
                if not isinstance(_x, InstWindow):
                    x.appendleft(_x)
                tensor = _x.to_torch()
                meta = _x.get_metadata()
                tensor_list.append(tensor)
                meta_list.append(meta)
        
        window_concat_tensor = torch.concat(tensor_list)
        return window_concat_tensor, meta_list
    
    def _run_prediction(self, tensor):
        """
        Run prediction on an torch.Tensor composed of
        concatenated, windowed tensors

        :
        """
        if self.model.device.type != self.device.type:
            self.model.to(self.device)
        if tensor.device.type == self.device.type:
            raw_preds = self.model(tensor)
        else:
            raw_preds = self.model(tensor.to(self.device))
        return raw_preds
    
    def _raw_preds2numpy(self, window_concat_tensor, raw_preds):
        npy_preds = np.full(window_concat_tensor.shape,
                            fill_value=np.nan,
                            dtype=np.float32)
        # Handle tuple output of EQTransformer
        if isinstance(raw_preds, tuple):
            raw_preds = torch.concat(raw_preds)
        # Reshape if needed
        if raw_preds.shape != window_concat_tensor.shape:
            raw_preds = raw_preds.reshape(window_concat_tensor.shape)
        # Convert back to numpy, shift off non-CPU device if needed
        if raw_preds.device.type != 'cpu':
            npy_preds = raw_preds.detach().cpu().numpy()
        else:
            npy_preds = raw_preds.detach().numpy()
        return npy_preds
    

    def _append_preds_to_buffer(self, pred_nwlt, meta_list):
        for _w in range(pred_nwlt.shape[1]):
            nwind = pred_nwlt[_w]
            # Fetch metadata for source window
            meta = meta_list[_w]
            # Update with i and j indices of weight and label codes
            mnu = {f'iw{_i}': _n for _i, _n in enumerate(self.weight_names)}
            mlu = {f'jw{_i}': _l for _i, _l in enumerate(self.label_codes)}
            meta.update(mnu)
            meta.update(mlu)
            # Get tier-0 key as inst_code
            k0 = meta['inst_code']
            # Get tier-1 key as model_name
            k1 = self.model.name
            # Append to tiered buffer using its 
            self.buffer.append(nwind, k0, k1)

    def __str__(self, extended=False):
        # rstr = super().__repr__()
        rstr = f'model: {self.model.name} | '
        rstr += f'shape: (1, {self.model.in_channels}, {self.model.in_samples}) ->'
        rstr += f'(1, {len(self.model.labels)}, {self.model.in_samples}) | '
        rstr += f'labels: {self.model.labels} (codes: {self.label_codes})\n'
        rstr += f'batch size: {self.max_pulse_size} | '
        # rstr += f'queue length: {len(self.queue)}\n'
        rstr += f' model weight names: '
        for _n in enumerate(self.weight_names):
            rstr += f'{_n}'
        rstr += f'\n{self.buffer.__str__(extended=extended)}'
        return rstr

    def __repr__(self):
        """
        Representative string of the parameters used to initialize this MachineWyrm
        """
        rstr = f'wyrm.wyrms.predict.MachineWyrm(model={self.model}, weight_names={self.weight_names}, '
        rstr += f'devicetype={self.device}, max_samples={self.max_samples}, max_pulse_size={self.max_pulse_size}, '
        rstr += f'stack_method={self.stack_method}, debug={self.debug})'
        return rstr
    