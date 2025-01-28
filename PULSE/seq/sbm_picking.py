
from collections import deque

import seisbench.models as sbm

from PULSE.data.dictstream import DictStream
from PULSE.mod.base import BaseMod
from PULSE.mod.sequencing import SeqMod
from PULSE.mod.sampling import SamplingMod, WindowingMod
from PULSE.mod.processing import ProcMod
from PULSE.mod.detecting import SBMMod
from PULSE.mod.buffering import BufferMod
from PULSE.mod.triggering import CRFTriggerMod


# class DevBreakpointMod(BaseMod):

#     def __init__(max_pulse_size=1e7,name='CheckNpts',)



class SBM_Picking_Sequence(SeqMod):
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
            buffer_length=300.,
            max_metadata_age=60.,
            max_pulse_size=1
            ):
        

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
        # Populate from defaults
        windmod = WindowingMod().update_from_seisbench(model=model)

        # PreProcess Windows to synchronize sampling, resample, treat gaps, etc.
        preprocmod = ProcMod(pclass='PULSE.data.window.Window',
                             pmethod='preprocess',
                             pkwargs={'trace_fill_rule':'primary'},
                             mode='inplace')
        # Run ML prediction
        predictmod = SBMMod(model=model,
                            weight_names=weight_names,
                            batch_sizes=(1, model._annotate_args['batch_size'][1]),
                            device=model.device.type)
        # Subset prediction outputs to desired labels
        selectmod = ProcMod(pclass='PULSE.data.dictstream.DictStream',
                            pmethod='select',
                            pkwargs={'component':f'[{labels}]'},
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
                          pmethod='to_pick_simple',
                          mode='output')

        ## STRING TOGETHER WORKFLOW
        modules = [windmod,
                    preprocmod,
                    predictmod,
                    selectmod,
                    blindmod,
                    stackmod,
                    predsamplemod,
                    triggermod,
                    pickmod]
        
        ## INITIALIZE/INHERIT FROM SEQMOD
        super().__init__(modules=modules, maxlen=max_metadata_age, max_pulse_size=max_pulse_size, name=model.name)

    def pulse(self, input: DictStream) -> deque:
        """Run one sequence of pulses for this :class:`~.SBM_Picking_Sequence` object

        :param input: a dictionary stream object containing waveform data
        :type input: PULSE.data.dictstream.DictStream
        :return: a collection of obspy picks from the specified ML detector/picker
        :rtype: collections.deque of :class:`~obspy.core.event.Pick` objects
        """        
        return super().pulse(input)
