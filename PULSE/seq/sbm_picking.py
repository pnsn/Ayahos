
from collections import deque

import seisbench.models as sbm

from PULSE.seq.sequence import Sequence
from PULSE.mod.sampling import SamplingMod, WindowingMod
from PULSE.mod.processing import ProcMod
from PULSE.mod.detecting import SBMMod
from PULSE.mod.buffering import BufferMod
from PULSE.mod.triggering import CRFTriggerMod

class SBM_Picking_Sequence(Sequence):
    """SeisBench Models Picking Sequence of PULSE Unit Modules

    (DictStream)[FoldTrace] 
     | -- > Window -> PreProcess -> Predict -> Select -> Blind -> Buffer -> Trigger -> T2P

    :param SeqMod: _description_
    :type SeqMod: _type_
    """    
    def __init__(
            self,
            model,
            weight_names=['pnw'],
            labels='PS',
            trigger_level=0.3,
            buffer_length=300.):
        

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

        # Generate Windows from an input DictStream
        windmod = WindowingMod().update_from_seisbench(model=model)

        # PreProcess Windows to synchronize sampling, resample, treat gaps, etc.
        preprocmod = ProcMod(pclass='PULSE.data.window.Window',
                             pmethod='preprocess',
                             pkwargs={'rule':'primary'},
                             mode='inplace')
        # Run ML prediction
        predictmod = SBMMod(model=model,
                            weight_names=weight_names,
                            batch_sizes=(1, model._annotate_args['batch_size'][1]),
                            device=model.device.type)
        # Subset prediction outputs to desired labels
        selectmod = ProcMod(pclass='PULSE.data.dictstream.DictStream',
                            pmethod='select',
                            pkwargs={'component',f'[{labels}]'},
                            mode='output')
        # Blind predictions
        blindmod = ProcMod(pclass='PULSE.data.dictstream.DictStream',
                           pmethod='blind',
                           mode='inplace',
                           pkwargs={'npts':model._annotate_args['blinding'][1]})
        # Buffer/stack predictions 
        stackmod = BufferMod(method=model._annotate_args['stacking'][1],
                             fill_value=0.,
                             maxlen=buffer_length)
        # Sample from prediction buffer with an enforced delay of 1 window length
        predsamplemod = SamplingMod(ref_val='P', blind_after_sampling=True)
        predsamplemod.update_from_seisbench(model=model, delay_scalar=1)
        # Generate Trigger objects from prediction peaks
        triggermod = CRFTriggerMod(thr_on=trigger_level)
        # Convert Trigger objects into obspy Pick objects
        pickmod = ProcMod(pclass='PULSE.data.pick.Trigger',
                          pmethod='to_pick',
                          mode='output')


    # def __init__(self, model, weight_names=['pnw'], labels='PS', trigger_level=0.3, buffer_length=300., maxlen=None, max_pulse_size=1, name=None):
        
        #
        

        

        ## STRING TOGETHER WORKFLOW
        sequence = [windmod,
                    preprocmod,
                    predictmod,
                    selectmod,
                    blindmod,
                    stackmod,
                    triggermod,
                    pickmod]
        ## INITIALIZE/INHERIT FROM SEQMOD
        super().__init__(modules=sequence, maxlen=maxlen, max_pulse_size=max_pulse_size, name=model.name)

    def pulse(self, input: deque) -> deque:
        return super().pulse(input)
