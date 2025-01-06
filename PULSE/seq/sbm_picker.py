import seisbench.models as sbm

from obspy.core.event import Pick, WaveformStreamID

from PULSE.data.dictstream import DictStream
from PULSE.data.window import Window

from PULSE.mod.sequencing import SeqMod
from PULSE.mod.windowing import WindMod
from PULSE.mod.buffering import BufferMod
from PULSE.mod.processing import ProcMod

class SBMPicker(SeqMod):
    """
    Sequence taking prediction :class:`~.DictStream` objects from :class:`~.SBMMod`
    outputs and returning phase picks in :class:`~obspy.core.event.Pick` format

    Sequence
        - BufferMod
        - TriggerMod
        - 
    """
    def __init__(
            self,
            model,
            labels='PS',
            buffer_length=300.,
            stack_method=3,
            min_fold=1,
            trigger_level=0.3,
            
    )
        
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError('model must be type seisbench.models.WaveformModel')
        elif model.name == 'WaveformModel':
            raise TypeError('seisbench.models.WaveformModel is a template class. Non-operable')
        else:
            self.model = model

        





        self.populate_windowing()

        
        buffermod = BufferMod(
            method=stack_method,
            fill_value=None,
            maxlen=buffer_length,
            max_pulse_size=1000,
            name=f'{model.name}'
        )

        windmod = WindMod(name=f'{model.name}_Pick',
                          target_npts=model.in_samples,
                          target_sampling_rate=model.sampling_rate,
                          overlap_npts=model._annotate_args['overlap'][-1],
                          primary_threshold=0.01,
                          secondary_threshold=0.01,
                          )

        subsetmod = ProcMod(
            pclass=DictStream,
            pmethod='select',
            mode='output',
            pkwargs={'component':labels}

        )

        triggermod = ProcMod(
            pclass=DictStream,
            pmethod='trigger',
            pkwargs=
        )


    def __setattr__(self, key, value):
        if key == 'model':
            if not isinstance(value, sbm.WaveformModel):
                raise TypeError
            elif value.name not in ['EQTransformer','PhaseNet']:
                raise NotImplementedError
            else:
                super().__setattr__(key, value)
            self.sampling_rate = self.model.sampling_rate

            self.primary_component = self.model.labels[0]
            self.secondary_components = self.model.labels[1:]