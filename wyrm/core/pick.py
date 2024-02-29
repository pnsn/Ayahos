from wyrm.core.process import WindowWyrm
from wyrm.core.data import BufferTree, PredictionBuffer
from wyrm.util.stacking import semblance

class SemblanceWyrm(WindowWyrm):
    """
    This submodule applys the weighted semblance methods presented in Yuan et al. (2023)
    and references therein 

    Dependent Parameters
     - Model Architecture   (M)
     - Model Weights        (W)
     
    Semi-Independent Parameters
     - Instrument           (I)

    Independent Parameters
     - Label                (l)
     - Time                 (t)
    """

    def __init__(self,
                 merge_parameters=['M','W'],
                 label_aliases={'P': ['P'], 'S': ['S'], 'D': ['Detection']},
    ):


        self.buffer = wcd.