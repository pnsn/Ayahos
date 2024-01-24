from collections import deque
from wyrm.wyrms.wyrm import Wyrm
from wyrm.structures.rtinststream import RtInstStream
from wyrm.structures.rtpredtrace import RtPredTrace
import wyrm.util.input_compatability_checks as icc
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
    def __init__(self, model, device, max_length=180., max_pulse_size=1000, debug=False):
        """
        Initialize a SeisBenchWyrm object

        :: INPUTS ::
        :param model: [seisbench.models.WaveformModel-like] a seisbench model
                        that has the WaveformModel parent class.
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
        # device compatability checks
        if not isinstance(device, torch.device):
            raise TypeError('device must be a torch.device object')
        else:
            if eval(f'torch.{device.type}.is_available()'):
                self.device = device
            else:
                raise ValueError(f'device type {device.type} is unavailable')
        # max_length compatability checks
        self.max_length = icc.bounded_floatlike(
            max_length,
            name='max_length',
            minimum=0,
            maximum=None,
            inclusive=False
        )

        # ################################################### #
        # Default/Derivative Attribute Initialization Section #
        # ################################################### #

        # Get convenience aliases for blinding and stacking values
        self._blinding = self.model._annotate_args['blinding'][-1]
        self._stacking = self.model._annotate_args['stacking'][-1]
        self._overlap = self.model._annotate_args['overlap'][-1]
        self._blind_mask = np.ones((1,self.model.in_samples))
        self._blind_mask[:self._blinding] = 0
        self._blind_mask[-self._blinding] = 0

        # Initialize Realtime Instrument Stream in PredTrace mode
        self.buffer = RtInstStream(trace_type=RtPredTrace, max_length=max_length)

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
        
        for _i in range(self.max_pulse_size):
            if len(x) > 0:
                _x = x.pop()
                if not isinstance(_x, InstWindow):
                    x.appendleft(_x)
                elif not self._validate_incoming_window(_x):
                    x.appendleft(_x)
                


    def preds_to_waves(self, preds, meta):
        """
        Given raw predictions from a seisbench WaveformModel-type model
        and input data metadata, convert all vectors into PyEW formatted
        wave messages

        :: INPUTS ::
        :param preds: [torch.Tensor] or [numpy.ndarray] array of continuous
                        predictions with SeisBench defined axes
                        i.e., (window #, channel #, data #)
        """
        # Basic type checks for preds
        if isinstance(preds, torch.Tensor):
            preds = self._detach_preds(preds)
        elif not isinstance(preds, np.ndarray):
            raise TypeError('preds must be type torch.Tensor or numpy.ndarray')
        # 


        if isinstance(meta, dict):
            meta = [meta]
        if not all(isinstance(_m, dict) for _m in meta):
            raise TypeError('not all entries in meta are type dict')
        
        expected_shape = (len(meta), self.model.in_channels, self.model.in_samples)
        if len(preds.shape) != 3:
            raise IndexError('preds does not have 3 dimensions')
        elif preds.shape != expected_shape:
            raise IndexError('preds ({preds.shape}) does not have the expected shape ({expected_shape})')
        if 
            

    def _preds_to_branch(self, preds):
        """
        Convert raw prediction outputs from a SeisBench WaveformModel-type
        into a branch (dictionary) structure keyed by the label code (first
        character, uppercase) to reflect the branch structure for RtInstStream
        objects.

        :: INPUT ::
        :param preds: [torch.Tensor]

        :: OUTPUT ::
        :return pred_branch: [dict] dictionary with (masked) numpy.ndarray
                    values and label code keys.
                    E.g., for EQTransformer
        """

        vdim_tuple = (1, self.model.in_samples)
        if isinstance(preds, tuple):
            if not all(isinstance(p, torch.Tensor) for p in preds):
                raise TypeError('all elements of preds must be type torch.Tensor')
        
        elif len(preds) != len(self.labels):
            emsg = f'mismatch in number of labels ({len(self.labels)}) '
            emsg += f'and number of predictions ({len(preds)})'
            raise IndexError(emsg)
        
        elif any(_p.shape != vdim_tuple for _p in preds):
            emsg = f'shape of prediction vectors mismatches those expected by this model'
            raise IndexError(emsg)

        pred_branch = {}
        for _i, _p in enumerate(preds):
            if _p.device.type != 'cpu':
                _o = _p.detach().cpu().numpy()
            else:
                _o = _p.detach().numpy()
            # Convert into a masked array if blinding is present
            if self._blinding_npts > 0:
                _o = np.ma.masked_array(
                    data=_o,
                    mask=np.zeros(_o.shape))
                _o.mask[:self._blinding[0]] = True
                _o.mask[-self._blinding[-1]:] = True

            pred_branch.update({self.labels[_i]: _o})

        return pred_branch
        



        
        if preds.device != 'cpu':
