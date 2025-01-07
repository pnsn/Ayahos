
from collections import deque

import seisbench.models as sbm

from PULSE.mod.sequencing import SeqMod
from PULSE.mod.sampling import SampleMod
from PULSE.mod.windowing import WindMod
from PULSE.mod.processing import ProcMod
from PULSE.mod.predicting import SBMMod
from PULSE.mod.buffering import BufferMod
from PULSE.mod.triggering import TriggerMod

class SBM_PickingMod(SeqMod):
    """SeisBench Models Picking Module

    (DictStream)[FoldTrace] 
     | -- > Window -> PreProcess -> Predict -> Select -> Blind -> Buffer -> Trigger -> T2P


    :param SeqMod: _description_
    :type SeqMod: _type_
    """    
    def __init__(self, model, weight_names=['pnw'], labels='PS', trigger_level=0.3, buffer_length=300., maxlen=None, max_pulse_size=1, name=None):
        
        # Compatability check for model
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError
        elif model.name == 'WaveformModel':
            raise TypeError
        else:
            pass
        pt_names = model.list_pretrained()
        if isinstance(weight_names, str):
            weight_names = [weight_names]
        if not all(_e in pt_names for _e in weight_names):
            msg = 'The following weight_names are not available:'
            for _e in weight_names:
                if _e not in pt_names:
                    msg += f' {_e}'   
            raise ValueError(msg)
        

        # Compose Sequence using model to parameterize most things
        windmod = WindMod(
            name=model.name,
            target_npts=model.in_samples,
            target_sampling_rate=model.sampling_rate,
            overlap_npts=model._annotate_args['overlap'][1],
            primary_components=['Z','3'],
            primary_threshold=0.9,
            secondary_components=['NE','12'],
            secondary_threshold=0.8,
            windowing_mode='normal',
            blind_after_sampling=False,
            max_pulse_size=1,
            maxlen=None
        )

        preprocmod = ProcMod(pclass='PULSE.data.window.Window',
                             pmethod='preprocess',
                             pkwargs={'rule':'primary'},
                             mode='inplace')
        
        predictmod = SBMMod(model=model,
                            weight_names=weight_names,
                            batch_sizes=(1, model._annotate_args['batch_size'][1]),
                            device=model.device.type)
        
        selectmod = ProcMod(pclass='PULSE.data.dictstream.DictStream',
                            pmethod='select',
                            pkwargs={'component',f'[{labels}]'},
                            mode='output')
        
        copymod = ProcMod(pclass='PULSE.data.dictstream.DictStream',
                          pmethod='copy',
                          mode='output')
        
        blindmod = ProcMod(pclass='PULSE.data.dictstream.DictStream',
                           pmethod='blind',
                           mode='inplace',
                           pkwargs={'npts':model._annotate_args['blinding'][1]})
        
        stackmod = BufferMod(method=model._annotate_args['stacking'][1],
                             fill_value=0.,
                             maxlen=buffer_length)
        
        predsamplemod = SampleMod()

        triggermod = TriggerMod(thr_on=trigger_level)

        pickmod = ProcMod(pclass='PULSE.data.pick.Trigger',
                          pmethod='to_pick',
                          mode='output')

        sequence = [windmod,
                    preprocmod,
                    predictmod,
                    selectmod,
                    copymod,
                    blindmod,
                    stackmod,
                    triggermod,
                    pickmod]
        
        super().__init__(modules=sequence, maxlen=maxlen, max_pulse_size=max_pulse_size, name=model.name)

    def pulse(self, input: deque) -> deque:
        return super().pulse(input)
