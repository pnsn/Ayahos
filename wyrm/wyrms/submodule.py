"""
:module: wyrm.wyrms.composite
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This contains composite Wyrm class(es)
"""
from wyrm.wyrms.coordinate import TubeWyrm
from wyrm.wyrms.window import WindowWyrm
from wyrm.wyrms.process import ProcWyrm
from wyrm.wyrms.predict import MachineWyrm
import seisbench.models as sbm
from collections import deque

class MLTubeWyrm(TubeWyrm):
    """
    Provide the following structure
                ________________________ TubeWyrm _________________________________
               /{Wave                                                              /{Pred
    TieredBuff{-{Wave-> WindWyrm -deq-> ProcWyrm -deq-> MachineWyrm -TieredBuff-> {-{Pred
    (from disk)\{Wave                   (default)                                  \{Pred
                {___________________________________________________________________{Pred
    """
    def __init__(self,
                 wait_sec=0,
                 devicetype='cpu',
                 model=sbm.EQTransformer(),
                 weight_names=['pnw','stead','instance'],
                 samprate=100,
                 max_samples=15000,
                 ww_kwargs={},
                 pw_kwargs={},
                 ml_kwargs={}):
        # Initialize Wyrm Organelles
        windwyrm = WindowWyrm(
            model_name=model.name,
            target_sr=samprate,
            target_npts=model.in_samples,
            **ww_kwargs)
        
        procwyrm = ProcWyrm(**pw_kwargs)

        mlwyrm = MachineWyrm(
            model=model,
            devicetype=devicetype,
            max_samples=max_samples,
            weight_names=weight_names,
            **ml_kwargs)

        super().__init__(wyrm_queue=deque([windwyrm, procwyrm mlwyrm]),
                         wait_sec=wait_sec)

