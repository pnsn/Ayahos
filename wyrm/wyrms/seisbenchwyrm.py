from collections import deque
from wyrm.wyrms.wyrm import Wyrm
# from wyrm.structures.rtinststream import RtInstStream
# from wyrm.structures.rtpredtrace import RtPredTrace
import wyrm.util.input_compatability_checks as icc
from wyrm.structures.window import InstWindow
import seisbench.models as sbm
import numpy as np
import torch
from tqdm import tqdm
from time import time

class WaveformModelWyrm(Wyrm):
    """
    Wyrm for housing and operating pulsed prediction by a WaveformModel from
    SeisBench and passing metadata from model inputs to outputs

    input - deque([tuple(tensor, metadata), tuple(tensor, metadata),...])
        with each tuple created as the output of a InstWindow.to_torch()
        instance
    output - RtPredStream containing stacked predictions
    """
    def __init__(self, model, weight_names, devicetype='cpu', max_length=180., max_pulse_size=1000, debug=False):
        """
        Initialize a SeisBenchWyrm object

        :: INPUTS ::
        :param model: [seisbench.models.WaveformModel-like] a seisbench model
                        that has the WaveformModel parent class.
        :param weight_names: [str] or [list] model name(s) to pass to 
                              model.from_pretrained()
        :param device: [torch.device] device to run predictions on
        :param max_length: [float] maximum prediction buffer length in seconds
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
        # max_length compatability checks
        self.max_length = icc.bounded_floatlike(
            max_length,
            name='max_length',
            minimum=0,
            maximum=None,
            inclusive=False
        )

        # Initialize Realtime Instrument Stream in PredTrace mode
        # self.buffer = RtInstStream(trace_type=RtPredTrace, max_length=max_length)
        # self.queue = deque([])

        # Start layered dict tree {NSLI}{meta}{metadata}
        #                               {wgt1}{tensor}
        #                               {wgt2}{tensor}
        self.tree = {}

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
        # Iterate across weight_names
        for _n in self.weight_names:
            if self.debug:
                print('Starting {self.model_name}, {_n} weighted prediction')
            # Ensure correct model weights are loaded
            self.model = self.model.from_pretrained(_n)

            # if self.model.weights_docstring is None:
            #     self.model.from_pretrained(_n)
            # elif _n not in self.model.weights_docstring:
            #     self.model.from_pretrained(_n)
            raw_preds = self._run_prediction(window_concat_tensor)
            npy_preds = self._raw_preds2numpy(window_concat_tensor, raw_preds)
            if self.debug:
                print(f'Appending to tree ({time() - tick})')
            # Iterate across windows and append to tree
            for _i, _meta in enumerate(meta_list):
                self._append_to_tree(_meta, npy_preds[_i, :, :], _n)
        self._sort_tree_indices()
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
        # TODO: Need to change this from an iteration loop into an explicit
        if isinstance(raw_preds, tuple):
            raw_preds = torch.concat(raw_preds)
        
        if raw_preds.device.type != 'cpu':
            npy_preds = raw_preds.detach().cpu().numpy().reshape(window_concat_tensor.shape)
        else:
            npy_preds = raw_preds.detach().numpy().reshape(window_concat_tensor.shape)

        # for _i, _p in enumerate(raw_preds):
        #     if _p.device.type != 'cpu':
        #         npy_preds[_i, :, :] = _p.detach().cpu().numpy()
        #     else:
        #         npy_preds[_i, :, :] = _p.detach().numpy()
        return npy_preds
    

    def _append_to_tree(self, meta, pred, wgt_name):
        inst_code = meta['inst_code']
        idx = meta['index']
        # If completely new instrument branch
        if inst_code not in self.tree.keys():
            self.tree.update({inst_code: {'metadata': {idx: meta}, wgt_name: {idx: pred}}})
        # for preexisting instrument branch
        else:
            # for new weight_name limb
            if wgt_name not in self.tree[inst_code].keys():
                self.tree[inst_code].update({wgt_name: {idx: pred}})
            # for existing weight_name limb
            elif idx not in self.tree[inst_code][wgt_name].keys():
                self.tree[inst_code][wgt_name].update({idx: pred})
            # Raise error if trying to overwrite pre-existing entry
            else:
                if self.debug:
                    print(f'KeyError(Attempting to insert data in pre-existing prediction index {inst_code} {wgt_name} {idx})')
                    breakpoint()
                else:
                    self.tree[inst_code][wgt_name].update({idx: pred})
            # Do check if new metadata
            if idx not in self.tree[inst_code]['metadata'].keys():
                self.tree[inst_code]['metadata'].update({idx: meta})
        # # Cleanup
        # self._sort_tree_indices()
        return self

    def _sort_tree_indices(self):
        # Iterate across inst_code's
        for k1 in self.tree.keys():
            branch = self.tree[k1]
            # Iterate across metadata and model_weight's
            for k2 in branch.keys():
                # Sort by index
                self.tree[k1][k2] = dict(sorted(branch[k2].items()))
        return self

            
    def __repr__(self):
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
        return rstr

    def __str__(self):
        rstr = self.__repr__()
        return rstr
    


    #  npy_preds = np.full(
    #         shape = tensor.shape,
    #         fill_value=np.nan,
    #         dtype=np.float32
    #     )

    #     # Iterate across specified model_weight(s)
    #     for _n in self.weight_names:
    #         # Load model weights
    #         if self.model.weights_docstring is None:
    #             self.model.from_pretrained(_n)
    #         elif _n not in self.model.weights_docstring:
    #             self.model.from_pretrained(_n)
    #         # Ensure model is on specified device
    #         if self.model.device.type != self.device.type:
    #             self.model.to(self.device)
    #         ## !!! PREDICT !!! with check that data are on device ##
    #         if model_inputs.device.type == self.device.type:
    #             raw_preds = self.model(model_inputs)
    #         else:
    #             raw_preds = self.model(model_inputs.to(self.device))
            
    #         ## Detach predictions and split into tuples to pass onward
    #         npy_preds = np.full(
    #             shape=model_inputs.shape,
    #             fill_value=np.nan,
    #             dtype=np.float32
    #         )
    #         for _j, _p in enumerate(raw_preds):
    #             if _p.device.type != 'cpu':
    #                 npy_preds[:, _j, :] = _p.detach().cpu().numpy()
    #             else:
    #                 npy_preds[:, _j, :] = _p.detach().numpy()
    #         # Split out windows and re-associate metadata with some updates
    #         for _i, _pred in enumerate(npy_preds):
    #             ometa = meta_list[_i].copy()
    #             ometa.update({'weight_name': _n})
    #             ometa.update({'label_codes': self.label_codes})
    #             # Compose output dictionary
    #             okey = f"{ometa['inst_code']}|{ometa['model_name']}|{ometa['weight_name']}"
    #             out_branch = {
    #                 okey: {
    #                     'data': _pred,
    #                     'meta': ometa,
    #                     'index': ometa['index']
    #                 }
    #             }
    #             # Append to queue
    #             self.queue.appendleft(out_branch)