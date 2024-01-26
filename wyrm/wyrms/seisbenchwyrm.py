from collections import deque
from wyrm.wyrms.wyrm import Wyrm
# from wyrm.structures.rtinststream import RtInstStream
# from wyrm.structures.rtpredtrace import RtPredTrace
import wyrm.util.input_compatability_checks as icc
from wyrm.structures.window import InstWindow
import seisbench.models as sbm
import numpy as np
import torch

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
                raise RintimeError(f'devicetype {devicetype} is an invalid device string for PyTorch')
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
        self.queue = deque([])

    def pulse(self, x):
        """
        :: INPUTS ::
        :param x: [deque] of InstWindow objects

        :: OUTPUT ::
        :return y: [RtInstStream] access to RtInstStream composed
                of RtPredTrace objects.
        """
        if not isinstance(x, deque):
            raise TypeError('input x must be a deque')
        # Concatenate input tensors and metadata
        tensor_list = []
        meta_list = []
        for _i in range(self.max_pulse_size):
            if len(x) > 0:
                _x = x.pop()
                if not isinstance(_x, InstWindow):
                    x.appendleft(_x)
                tensor = _x.to_torch()
                meta = _x.get_metadata()
                tensor_list.append(tensor)
                meta_list.append(meta)
        
        model_inputs = torch.concat(tensor_list)

        # Iterate across specified model_weight(s)
        for _n in self.weight_names:
            # Load model weights
            if self.model.weights_docstring is None:
                self.model.from_pretrained(_n)
            elif _n not in self.model.weights_docstring:
                self.model.from_pretrained(_n)
            # Ensure model is on specified device
            if self.model.device.type != self.device.type:
                self.model.to(self.device)
            ## !!! PREDICT !!! with check that data are on device ##
            if model_inputs.device.type == self.device.type:
                raw_preds = self.model(model_inputs)
            else:
                raw_preds = self.model(model_inputs.to(self.device))
            
            ## Detach predictions and split into tuples to pass onward
            npy_preds = np.full(
                shape=model_inputs.shape,
                fill_value=np.nan,
                dtype=np.float32
            )
            for _j, _p in enumerate(raw_preds):
                if _p.device.type != 'cpu':
                    npy_preds[:, _j, :] = _p.detach().cpu().numpy()
                else:
                    npy_preds[:, _j, :] = _p.detach().numpy()
            # Split out windows and re-associate metadata with some updates
            for _i, _pred in enumerate(npy_preds):
                ometa = meta_list[_i].copy()
                ometa.update({'weight_name': _n})
                ometa.update({'label_codes': self.label_codes})
                # Compose output dictionary
                okey = f"{ometa['inst_code']}|{ometa['model_name']}|{ometa['weight_name']}"
                out_branch = {
                    okey: {
                        'data': _pred,
                        'meta': ometa,
                        'index': ometa['index']
                    }
                }
                # Append to queue
                self.queue.appendleft(out_branch)
            
    def __repr__(self):
        # rstr = super().__repr__()
        rstr = f'model: {self.model.name} | '
        rstr += f'shape: (1, {self.model.in_channels}, {self.model.in_samples}) ->'
        rstr += f'(1, {len(self.model.labels)}, {self.model.in_samples}) | '
        rstr += f'labels: {self.model.labels} (codes: {self.label_codes})\n'
        rstr += f'batch size: {self.max_pulse_size} | '
        rstr += f'queue length: {len(self.queue)}\n'
        rstr += f' model weight names:\n'
        for _n in self.weight_names:
            rstr += f'    {_n}\n'
        return rstr

    def __str__(self):
        rstr = self.__repr__()
        return rstr