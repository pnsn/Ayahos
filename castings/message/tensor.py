import torch
import seisbench.models as sbm
import numpy as np
from obspy import UTCDateTime
from wyrm import _BaseMsg

class TensorMsg(_BaseMsg):
    """
    Provides a slightly augmented PyTorch Tensor class 
    that includes instrument code (N.S.L.I), starttime,
    sampling rate, and _BaseMsg attributes
    """

    def __init__(
            self,
            data=[],
            model_name=None,
            nsli_code='..--.',
            starttime=UTCDateTime(0),
            sampling_rate=100.,
            mtype=None,
            mcode=None,
            **kwargs):
        
        # Initialize inherited features from torch.Tensor
        self.tensor = torch.Tensor(data, **kwargs)
        # Initialize inherited features from wyrm.message.base._BaseMsg
        super().__init__(mtype=mtype, mcode=mcode)

        # Compatability checks with model_name
        if isinstance(model_name, str):
            if model_name in dir(sbm):
                self.model_name = model_name
            else:
                print(f'custom model_name {model_name} not included in installed version of SeisBench')
        elif isinstance(model_name, type(None)):
            print('model_name=None is strictly a placeholder - check seisbench.models for standard model classes')
            self.model_code = model_name
        else:
            raise TypeError(f'model_name {model_name} must be type str or None.')
    
        # Compatability checks with nsli_code
        if not isinstance(nsli_code, str):
            raise TypeError('nsli_code must be type str')
        elif len(nsli_code.split('.')) != 4:
            raise SyntaxError('nsli_code must be a 4-element, "."-delimited string')
        else:
            self.nsli_code = nsli_code

        # Comptability checks for starttime
        try:
            self.starttime = UTCDateTime(starttime)
        except TypeError:
            raise TypeError('starttime must be UTCDateTime or a compatable input to UTCDateTime')


        # Comptatability checks for sampling_rate
        self.sampling_rate = self.bounded_floatlike(sampling_rate,
                                                    minimum=1.,
                                                    maximum=10000.)

    def __repr__(self):
        rstr = f'{super().__repr__()}\n'
        rstr += f'{self.nsli_code} | {self.starttime} | {self.sampling_rate} Hz\n'
        rstr += f'Current device {self.tensor.device}\n'
        rstr += f'{self.tensor}'
        return rstr
    
    def 