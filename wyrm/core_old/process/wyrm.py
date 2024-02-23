import torch
import numpy as np
import seisbench.models as sbm
import wyrm.util.compatability as wcc
from collections import deque
from wyrm.core._base import Wyrm
from wyrm.core.buffer import TieredBuffer, PredictionBuffer
from wyrm.core.window import PredictionWindow


class MachineWyrm(Wyrm):

    def __init__(
            self,
            model,
            weight_names,
            devicetype='cpu',
            max_samples=12000,
            stack_method='max',
            max_pulse_size=1000,
            debug=False
    ):
        """
        Initialize a MachineWyrm object

        :: INPUS ::
        :param model: [seisbench.models.WaveformModel] model architecture defining
                        object based on the SeisBench WaveformModel class
        :param weight_names: [list-like] of [str]
                        one or more model weights that can be loaded for `model`
                        using the `model.from_pretrained()` method
        :param devicetype: [str] device type to run predictions on
                        see torch.device()
        :param max_samples: [int] maximum number of samples for each PredictionBuffer
                        at the terminations of this MachineWyrm's buffer attribute
                        see wyrm.core.buffer.prediction.PredictionBuffer(max_samples)
        :param stack_method: [str] stacking method to pass to terminating PredictionBuffer
                        objects. See wyrm.core.buffer.prediction.PredictionBuffer(stacking_method)
        :param max_pulse_size: [int] used as a maximum batch size here for windowed data 
                        accumulation prior to prediction.
        :param debug: [bool] run in debug mode?
        """
        super().__init(max_pulse_size=max_pulse_size, debug=debug)

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
        self.max_samples = wcc.bounded_floatlike(
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
            buff_class=PredictionBuffer,
            model=self.model,
            weight_names=self.weight_names,
            max_samples=self.max_samples,
            stack_method=self.stack_method
            )
    

    def pulse(self, x, **options):
        """
        Conduct a pulsed data accumulation from an input deque of PredictionWindow objects
        convert them into a torch.Tensor, and conduct predictions on each input data window
        with each weight specified in self.weight_Names

        :: INPUTS ::
        :param x: [deque] of [wyrm.core.window.predicition.PredictionWindow] objects
        :param options: [kwargs] key word arguments to pass to self.buffer.append()
                        see wyrm.core.buffer.prediction.PredictionBuffer.append()
                            -> include_blinding=False (default)
        :: OUTPUT ::
        :retury y: [wyrm.core.buffer.structure.TieredBuffer] terminating in 
                    [wyrm.core.buffer.prediction.PredicitonBuffer] objects with
                    tier keys set as TK0 = id, TK1 = weight_name.
                        Alias to access this MachineWyrm's self.buffer attribute
        """
        if not isinstance(x, deque):
            raise TypeError('input x must be type deque')

        qlen = len(x)
        batch_data = []
        batch_meta = []
        for _i in range(self.max_pulse_size):
            _x = x.pop()
            if not isinstance(_x, PredictionWindow):
                x.appendleft(_x)
            else:
                _tensor, _meta = _x.split_for_ml()
                batch_data.append(_tensor)
                batch_meta.append(_meta)
                # Clean up iterated copies
                del _tensor, _meta, _x
            # Early stopping clauses
            # if the input queue is empty or every queue element has been assessed
            if len(x) == 0 | _i + 1 == qlen:
                break
    
        batch_data = torch.concat(batch_data)

        for wname in self.weight_names:
            # Load model weights
            self.model = self.model.from_pretrained(wname)
            # Run Prediction
            batch_pred = self.run_prediction(batch_data)
            # Convert to numpy
            batch_pred_npy = self.pred2npy(batch_pred, batch_data)
            del batch_pred, batch_data

            # Split into individual windows and reconstitute PredictionWindows
            for _i, meta in enumerate(batch_meta):
                tk0 = meta['inst_code']
                meta.update({'labels': self.model.labels})
                tk1 = wname
                # Reconstitute PredictionWindows
                pwind = PredictionWindow(data=batch_pred_npy[_i,:,:], **meta)
                self.buffer.append(pwind.copy(), TK0=tk0, TK1=tk1, **options)
                # Cleanup
                del pwind, meta, tk0, tk1
            # Cleanup
            del batch_pred_npy
        
        y = self.buffer

        return y
    
    def run_prediction(self, batch_data):
        """
        Run a ML prediction on an input batch of windowed data using self.model on
        self.device. Provides checks that self.model and batch_data are appropriately
        formatted and staged on the same device

        :: INPUT ::
        :param batch_data: [numpy.ndarray] or [torch.Tensor] data array with scaling
                        appropriate to the input layer of self.model
        :: OUTPUT ::
        :return batch_preds: [torch.Tensor] or [tuple] thereof - prediction outputs
                        from self.model
        """
        if not isinstance(batch_data, (torch.Tensor, np.ndarray)):
            raise TypeError('batch_data must be type torch.Tensor or numpy.ndarray')
        elif isinstance(batch_data, np.ndarray):
            batch_data = torch.Tensor(batch_data)
        
        if self.model.device.type != self.device.type:
            self.model.to(self.device)
        
        if batch_data.device.type != self.device.type:
            batch_preds = self.model(batch_data.to(self.device))
        else:
            batch_preds = self.model(batch_data)
        
        return batch_preds
    
    def preds_torch2npy(self, nwind, preds):
        """
        Conduct minimal postprocessing to convert raw prediction outputs (preds)
        from a WaveformModel into a numpy.ndarray with the following axes
            0 - windows
            1 - labels
            2 - predicted values
        
        :: INPUTS ::
        :param nwind: [int] number of windows in preds
        :param preds: [torch.Tensor] or [tuple] thereof 
                    raw output from a WaveformModel object
        
        :: OUTPUT ::
        :return preds_npy: [numpy.ndarray] (nwind, nlabel, nvalues) array
                    containing predicted values from preds
        """
        out_shape = (nwind, len(self.model.labels), self.in_samples)
        # If output is a list-like of torch.Tensors (e.g., EQTransformer raw output)
        if isinstance(preds, (tuple, list)):
            if all(isinstance(_t, torch.Tensor) for _t in preds):
                preds = torch.concat(preds)
            else:
                raise TypeError('not all elements of preds is type torch.Tensor')
        # If reshaping is necessary
        if preds.shape != out_shape:
            preds = preds.reshape(out_shape)
        
        # Extract to numpy
        if preds.device.type != 'cpu':
            preds_npy = preds.detach().cpu().numpy()
        else:
            preds_npy = preds.detach().numpy()
        return preds_npy
        


            