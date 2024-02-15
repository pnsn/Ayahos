"""
:module: wyrm.wyrms.composite
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This contains composite Wyrm class(es)
"""
from wyrm.wyrms.coordinate import TubeWyrm, CanWyrm
from wyrm.wyrms.window import WindowWyrm
from wyrm.wyrms.process import ProcWyrm
from wyrm.wyrms.predict import MachineWyrm
from wyrm.buffer.structures import TieredBuffer
from wyrm.buffer.trace import TraceBuff
import seisbench.models as sbm
from collections import deque

class MLTubeWyrm(TubeWyrm):
    """
    Provide the following structure
      vv User Input vv       _________________ TubeWyrm _________________________________
               /[C]{TraceBuff          {Windows}       {MLArray}           {MLArray}    /[W]{PredBuff
    (TB)[INST]{-[C]{TraceBuff->|WindWyrm|-deq->|ProcWyrm|-deq->|MachineWyrm|-(TB)[INST]{-[W]{PredBuff
               \[C]{TraceBuff          (default)(W1,W2,...WN)                           \[W]{PredBuff
                             ____________________________________________________________[W]{PredBuff

    Function representation MLTubeWyrm(TieredBuffer(TraceBuff)) = TieredBuffer(PredBuff)
    """
    def __init__(self,
                 wait_sec=0.,
                 devicetype='cpu',
                 model=sbm.EQTransformer(),
                 weight_names=['pnw','stead','instance'],
                 samprate=100,
                 max_samples=15000,
                 ww_kwargs={},
                 pw_kwargs={},
                 ml_kwargs={}):
        # Initialize Window-Generating WindWyrm
        windwyrm = WindowWyrm(
            model_name=model.name,
            target_sr=samprate,
            target_npts=model.in_samples,
            **ww_kwargs)
        
        # Initialize Preprocessing ProcWyrm
        procwyrm = ProcWyrm(**pw_kwargs)

        # Initialize ML Prediction MachineWyrm
        mlwyrm = MachineWyrm(
            model=model,
            devicetype=devicetype,
            max_samples=max_samples,
            weight_names=weight_names,
            **ml_kwargs)

        # Inherit from TubeWyrm
        super().__init__(wait_sec=wait_sec)
        # Construct wyrmqueue with append
        self.append(windwyrm)
        self.append(procwyrm)
        self.append(mlwyrm)
        
    def pulse(self, x):
        """
        Explicitly state use of TubeWyrm's pulse method
        """
        y = super().pulse(x)
        return y


def model_params_dict():
    eqt = {'model': sbm.EQTransformer(),
           'weight_names': ['pnw','stead','instance'],
           'devicetype': 'cpu',
           'samprate': 100,
           'max_samples': 15000,
           'ww_kwargs': {},
           'pw_kwargs': {},
           'ml_kwargs': {}}
    # Produce subsequent copies for different architectures
    pn = eqt.copy()
    # Run update on copy to 
    pn.update({'model': sbm.PhaseNet,
               'weight_names': ['stead','instance','diting']})
    out = {'EQTransformer': eqt, 'PhaseNet': pn}
    return out


class MultiModelWyrm(CanWyrm):
    
    def __init__(self,
                 wait_sec=0.,
                 model_params_dict=model_params_dict,
                 max_pulse_size=5,
                 debug=False):
        # Construct model-architecture-dependent MLTubes
        wyrm_dict = {}
        for _k, _v in model_params_dict.items():
            wyrm_dict.update(_k, MLTubeWyrm(wait_sec=wait_sec, debug=debug, **_v))
        # Initailize CanWyrm & inherit all features
        super.__init__(wyrm_dict=wyrm_dict, wait_sec=wait_sec, max_pulse_size=max_pulse_size, debug=debug)
        
    def pulse(self, x):
        """
        Run a pulse from CanWyrm with some additional checks on 
        input type

        :: INPUTS ::
        
        """
        # Run compatability checks on x
        if not isinstance(x, TieredBuffer):
            raise TypeError('x must be a wyrm.buffer.structures.TieredBuffer object')
        elif not isinstance(x._template, TraceBuff):
            raise TypeError('x must be a TieredBuffer terminating with wyrm.buffer.trace.TraceBuff objects')
        else:
            # Run pulse from CanWyrm
            y = super().pulse(x)
            return y
