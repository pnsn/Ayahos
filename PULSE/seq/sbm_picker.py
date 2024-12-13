import seisbench.models as sbm

from obspy.core.event import Pick, WaveformStreamID

from PULSE.data.dictstream import DictStream
from PULSE.data.window import Window

from PULSE.mod.sequence import SeqMod
from PULSE.mod.window import WindMod
from PULSE.mod.buffer import BufferMod
from PULSE.mod.process import ProcMod

class SBMPicker(SeqMod):
    """
    Sequence taking prediction :class:`~.DictStream` objects from :class:`~.SBMMod`
    outputs and returning phase picks in :class:`~obspy.core.event.Pick` format

    Sequence
        - ProcMod(DictStream.select) -- select labels to pick
        - BufferMod() -- buffer 
    """
    def __init__(
            self,
            labels='PS',
            buffer_length=300.,
            stack_method=3,
            min_fold=1,
            trigger_level=0.3,
            
    )
        
        buffermod = BufferMod(
            method=stack_method,
            fill_value=None,
            maxlen=buffer_length,
            max_pulse_size=1000,
            name='pred'
        )

        windmod = WindMod()

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