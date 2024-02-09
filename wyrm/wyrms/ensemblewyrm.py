from wyrm.util.stacking import semblance
from wyrm.wyrms.windowwyrm import WindowWyrm
import wyrm.util.input_compatability_checks as icc
import inspect

class SembWyrm(WindowWyrm):
    
    def __init__(self, inst_fn_str='*', model_fn_str='*', wgt_fn_str='*', label_dict={'P': ['P','p'], 'S': ['S', 's'], 'D': ['Detection','STALTA']}, samp_rate=100, **semblance_kwargs):
        """
        Pull windows from a TieredBuffTree at the end of a CanWyrm object
        and conduct semblance
        """
        # Inherit from WindowWyrm
        super().__init__()


        # Compat check for instrument name fnmatch.filter string
        if not isinstance(inst_fn_str, str):
            raise TypeError('inst_fn_str must be type str')
        else:
            self.ifs = inst_fn_str
        # Compat check for ML model name fnmatch.filter string
        if not isinstance(model_fn_str, str):
            raise TypeError('model_fn_str must be type str')
        else:
            self.mfs = model_fn_str
        # Compat check for pretrained model weight name fnmatch.filter string
        if not isinstance(wgt_fn_str, str):
            raise TypeError('wgt_fn_str must be type str')
        else:
            self.wfs = wgt_fn_str
        # Compat checks for sampling rate
        self.samp_rate = icc.bounded_floatlike(
            samp_rate,
            name='samp_rate',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # Compatability check for semblance kwarg gatherer
        sargs = inspect.getfullargspec(semblance).args
        emsg = ''
        for _k in semblance_kwargs.keys():
            if _k not in sargs:
                if emsg == '':
                    emsg += 'The following kwargs are not compabable with wyrm.util.stacking.semblance:'
                emsg += f'\n{_k}'
        if emsg == '':
            raise TypeError(emsg)
        else:
            self.skwargs = semblance_kwargs
        
        self.queue = deque([])

    def _gather