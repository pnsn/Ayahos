import torch
import seisbench.models as sbm
from wyrm.wyrms.wyrm import Wyrm


class TorchWyrm(Wyrm):
    """
    BaseClass for generalized handling of a SeisBench formatted
    PyTorch model for prediction purposes
    """

    def __init__(self, model, device):
        # Compatability checks
        if isinstance(model, sbm.WaveformModel):
            self.model = model
            self.name = model.name
            if self.name in ['PhaseNet', 'EQTransformer']:
                self.axis_order = {'window':0, 'channel': 1, 'sample': 2}
            else:
                self.axis_order = 'Unknown'
            self.in_chan = model.in_channels
            self.in_samp = model.in_samples
            self.in_order = model.order

            self.labels = model.labels
            self.
        elif isinstance(model, torch.nn.Module)
        self.device = device

    def __repr__(self):
        rstr = f"Device: {self.device}\n"
        rstr += f"Model Component Order: {self.model.component_order}\n"
        rstr += f"Model Prediction Classes: {self.model.classes}\n"
        rstr += f"Model Citation\n{self.model.citation}"
        return rstr

    def _send_model_to_device(self):
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

    def _run_prediction(self, x):
        """
        Execute a PyTorch model prediction

        :: INPUT ::
        :param x: [torch.Tensor] input data tensor

        :: RETURN ::
        :param y: [torch.Tensor] output prediction tensor
        """
        y = self.model(x)
        return y

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
        # Ensure model is on the specified device
        self._send_model_to_device()
        # Ensure data are in Tensor format and on the specified device
        self._send_data_to_device(x)
        # Run prediction
        y = self._run_prediction(x)
        # Return raw output of prediction
        return y


###################################
### ML PROCESSING CHILD CLASSES ###
###################################

class ContinuousMLWyrm(TorchWyrm):
    """
    ML Prediction module for models where the input (data) and output (pred) arrays
    consist of windowed time-series with associated metadata
    """
    def __init__(self, model, device):#ml_input_shape, window_axis, sample_axis, channel_axis):
        super().__init__(model, device)
        self.windows = 
    
    def __repr__(self):
        rstr = f'Input Dimensions: {self.ml_input_shape}\n'
        rstr += f'Window Axis: {self.window_axis}\nSample Axis: {self.sample_axis}\nChannel Axis: {self.channel_axis}\n'
        rstr += super().__repr__()

        return rstr
    
    def _parse_MLWMsg(self, mlwmsg):
        _data = mlwmsg.data
        # Handle 1C data
        if _data.shape[0] == self.sample_axis:
