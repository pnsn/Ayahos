import torch
import seisbench.models as sbm
from wyrm.wyrms.wyrm import Wyrm
import fnmatch
import numpy as np
from collections import deque
from wyrm.structures.window import MLInstWindow, LabeledArray

# class HardwareUnavailableError(Exception):
#     def __init__(self, msg):
#         self.msg = msg
#         super().__init__(self.msg)


class TorchNNWyrm(Wyrm):
    """
    Baseclass for generalized handling of PyTorch Neural Network
    models in the Wyrm environment. Provides a basic pulse method
    that iterates across an array-like input of objects.
    """

    def __init__(
        self,
        model,
        name=None,
        device="cpu",
        pred_dtype=np.float32,
        max_pulse_size=None,
        debug=False,
    ):
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)

        # Basic compatability check on model
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be type torch.nn.Module")
        else:
            self.model = model

        # Basic compatability check on name
        self.name = self.none_str(name, name="name")

        # Compatability and parsing checks on device
        if isinstance(device, (str, torch.device, type)):
            self._check_device_compatability(device)

        # Compatability check on pred_dtype
        if isinstance(pred_dtype, (np.int8, np.int16, np.int32, np.float32)):
            self.pred_dtype = pred_dtype
        else:
            raise TypeError(
                f"pred_dtype {pred_dtype} not supported - only np.int8, np.int16, np.int32, and np.float32"
            )
        # Initialize output queue
        self.queue = deque([])

    def pulse(self, x):
        """
        As a pulse,
        run prediction on x
        append the result to self.queue with appendleft()
        make queue available as y
        """
        _y = self._run_prediction(x)
        self.queue.appendleft(_y)
        y = self.queue
        return y

    def _run_prediction(self, x, dtype=np.float32):
        """
        Execute a PyTorch model prediction

        :: INPUT ::
        :param x: [torch.Tensor] input data tensor

        :: RETURN ::
        :param y: [torch.Tensor] output prediction tensor
        """
        self._sync_model_data_to_device(x)
        pred = self.model(x)
        y = self._detach_pred_from_device(pred)
        return y

    def _sync_model_data_to_device(self, x):
        """
        Ensure that self.model and data 'x' 
        are both sent to(self.device)

        :: INPUT ::
        :param x: [torch.Tensor] pytorch tensor object
                        OR
                  [numpy.ndarray] numpy.ndarray object to convert to
                                a torch.Tensor and then send to(self.device)
        
        :: OUTPUT ::
        :return x: [torch.Tensor] pytorch tensor object
        """
        if x.device != self.device:
            x = self._send_data_to_device(x)
        if self.model.device != self.device:
            self._send_model_to_device()
        return x

    def _check_device_compatability(self, device):
        if device is None:
            if torch.cpu.is_available():
                self.device = torch.device()
            else:
                raise OSError(
                    "Default to cpu device unsuccessful - cpu is unavailable...?"
                )
        elif isinstance(device, torch.device):
            if eval(f"torch.{device.type}.is_available()"):
                self.device = device
            else:
                raise ValueError(f"device type {device.type} is unavailable")
        elif isinstance(device, str):
            if eval(f"torch.{device.lower()}.is_available()"):
                self.device = torch.device(device.lower())
            else:
                raise ValueError(f"device type {device.lower()} is unavailable")
        else:
            raise TypeError("device must be type str or torch.device")

    def _send_model_to_device(self):
        """
        Send model to pre-defined device, if it isn't already there
        """
        if self.model.device != self.device:
            self.model.to(self.device)

    def _send_data_to_device(self, x):
        # If the data presented are not already in a torch.Tensor format
        # but appear that they can be, convert.
        if not isinstance(x, torch.Tensor) and isinstance(x, np.ndarray):
            x = torch.Tensor(x)
            x.to(self.device)
            return x
        # If the data are already a torch.Tensor, pass input to output
        elif isinstance(x, torch.Tensor):
            x.to(self.device)
            return x
        # For all other cases, raise TypeError
        else:
            raise TypeError
    
    def _detach_pred_from_device(self, pred):
        if pred.device != 'cpu':
            y = pred.detach().cpu()
        else:
            y = pred.detach()
        return y

    def __str__(self):
        rstr = f"{super().__str__(self)}\n"
        rstr += f"Model Name: {self.name} | Device {self.device.type}"
        return rstr

    def __repr__(self):
        rstr = self.__str__()
        return rstr

class SeisBenchWyrm(TorchNNWyrm):
    def __init__(
            self,
            model=sbm.EQTransformer().from_pretrained('pnw'),
            device='mps',
            max_batch_size=1000,
            max_pulse_size=None,
            debug=False):
        
        # Compatability check for model
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError('model must be a seisbench.model.WaveformModel')



class SeisBenchWyrm(TorchNNWyrm):
    """
    Wyrm for generalized hosting and operation of SeisBench
    formatted PyTorch models that accept windowed trace data
    as torch.Tensor objects.
    """

    def __init__(
        self,
        model="EQTransformer",
        wgt="stead",
        device="cpu",
        max_batch_size=1000.,
        max_pulse_size=None,
        debug=False,
    ):
        # Initialize TorchNNWyrm inheritance + some input compatability checks
        super().__init__(device=device, max_pulse_size=max_pulse_size, debug=debug)

        # Basic compatability checks on `model` and `wgt`
        if not isinstance(wgt, (str, type(None))):
            raise TypeError("wgt must be type str or None")

        if not isinstance(model, (sbm.WaveformModel, str, type(None))):
            raise TypeError(
                "model must be type seisbench.models.WaveformModel, str, or None"
            )

        # Compatability checks on `model` and `wgt`
        # If a pre-loaded seisbench model object
        if isinstance(model, sbm.WaveformModel):
            self.model = model
            self.name = self.model.name
            # Cross-check model and weight inputs
            self.load_weights(wgt)
        elif isinstance(model, str):
            self.name = self.validate_seisbench_model_name(model, arg_name='model')
            self.model = eval(f'sbm.{self.name}()')
            self.load_weights(wgt)
        # If None, create dummy.
        elif isinstance(model, type(None)):
            self.model = None
            self.name = None
            self.wgt_info = (None, None)
        else:
            raise TypeError(f'input "model" must be type seisbench.model.WaveformModel, str, or None')

    def __repr__(self):
        rstr = f"Device: {self.device} | "
        rstr += f"Model: {self.name} | "
        rstr += f"Wgt version: {self.wgt_info[0]}\n"
        rstr += f"{self.wgt_info[1]}"

        # rstr += f"Model Component Order: {self.model.component_order}\n"
        # rstr += f"Model Prediction Classes: {self.model.classes}\n"
        # rstr += f"Model Citation\n{self.model.citation}"
        return rstr

    def load_weights(self, *args, **kwargs):
        """
        Wrapper for SeisBench Model (sbm) pretrained weights loading
        i.e.
        model = sbm.<model_subclass>()
        model.from_pretrained(*args, **kwargs)

        see seisbench.models.<model_subclass>.from_pretrained documentation

        This method updates the self.model attribute and (re)populates
        the self.wgt_info attribute
        """
        self.model.from_pretrained(*args, **kwargs)
        self.wgt_info = (self.model.weights_version, self.model.weights_docstring)

    def from_pretrained(self, *args, wgt=None, **kwargs):
        if self.model is not None:
            # If weights are already loaded, make sure wgt_info is up-to-date
            if wgt.lower() in self.model.weights_docstring.lower():
                self.wgt_info = (
                    self.model.weights_version,
                    self.model.weights_docstring,
                )
            # If wgt name differs
            else:
                self.load_weights(wgt, *args, **kwargs)
                self.wgt_info = (
                    self.model.weights_version,
                    self.model.weights_docstring,
                )
        else:
            print("self.model is None - cannot load weights from_pretrained")

    def pulse(self, x):
        """
        Run a prediction on input data tensor.

        :: INPUT ::
        :param x: [torch.Tensor] or [numpy.ndarray]
                pre-processed data with appropriate dimensionality for specified model.
                e.g.,
                for PhaseNet: x.shape = (nwind, chans, time)
                                        (nwind, 3, 3000)
                for EQTransformer: x.shape = (time, chans, nwind)
                                             (6000, 3, nwind)
        :: OUTPUT ::
        :return y: [torch.Tensor] predicted values for specified model
            e.g.
            PhaseNet: y = (nwind, [P(tP(t)), P(tS(t)), P(Noise(t))], t)
            EQTransformer: y = (t, [P(Detection(t)), P(tP(t)), P(tS(t))], nwind)

        """

        tensor, metadata = self._assemble_batch_tensor(x)
        preds = self._run_prediction(tensor)
        labeled_preds = self._label_preds(preds, tensor, metadata)
        y = self._run_prediction(x)
        # Return raw output of prediction
        return y

    def _assemble_batch_tensor(self, x, target_type=WindowMsg, extraction_method='to_torch(order="ZNE")'):
        if not isinstance(x, deque):
            raise TypeError('input x must be type deque')
        x_len = len(x):
        if x_len > self.max_pulse_size:
            max_iter = self.max_pulse_size
        else:
            max_iter = x_len
        tt_holder = []; metadata = []
        for _ in range(max_iter):
            _x = x.pop()
            if not isinstance(_x, target_type):
                x.append_left(_x)
            elif _x.model_name != self.model_name:
                x.append_left(_x)
            else:
                tt, md = eval(f'_x.{extraction_method}')
                tt_holder.append(tt)
                metadata.append(md)
        tensor = torch.concat(tt_holder)
        return tensor, metadata
                
    def _label_preds(self, preds, metadata):
        if preds.shape[0] != len(metadata):
            raise IndexError('predictions should have axis 0 length == len(metadata)')
        for _i, _md in enumerate(metadata):
            _pc = preds[_i]
            for _j, _p in enumerate(_pc):
                lc = self.model.labels[_j]




    def _run_prediction(self, x):
        x = self._sync_model_data_to_device(x)
        preds = np.zeros(shape=x.shape, dtype=self.pred_dtype)
        y = self.model(x)
        for _i, _p in enumerate(preds):
            preds[:, _i, :] = self._detach_pred_from_device(_p)
        return preds
    


    def _run_labeled_prediction(self, x):
        data, metadata = x.to_torch()



# ###################################
# ### ML PROCESSING CHILD CLASSES ###
# ###################################

# class ContinuousMLWyrm(TorchWyrm):
#     """
#     ML Prediction module for models where the input (data) and output (pred) arrays
#     consist of windowed time-series with associated metadata
#     """
#     def __init__(self, model, device):#ml_input_shape, window_axis, sample_axis, channel_axis):
#         super().__init__(model, device)
#         self.windows =

#     def __repr__(self):
#         rstr = f'Input Dimensions: {self.ml_input_shape}\n'
#         rstr += f'Window Axis: {self.window_axis}\nSample Axis: {self.sample_axis}\nChannel Axis: {self.channel_axis}\n'
#         rstr += super().__repr__()

#         return rstr

#     def _parse_MLWMsg(self, mlwmsg):
#         _data = mlwmsg.data
#         # Handle 1C data
#         if _data.shape[0] == self.sample_axis:


# self.in_chan = model.in_channels
# self.in_samp = model.in_samples
# self.in_order = model.order

# self.labels = model.labels

# elif isinstance(model, torch.nn.Module):

# # Compatability checks on device
# self.device = device
