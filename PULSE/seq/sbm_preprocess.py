
import seisbench.models as sbm

from PULSE.mod.sequencer import SeqMod, Sequence
from PULSE.mod.processer import ProcMod
from PULSE.mod.windower import WindMod
from PULSE.data.window import Window





class SBM_PP_SeqMod(SeqMod):
    def __init__(
            self,
            model=sbm.EQTransformer,
            weight='pnw',
            max_pulse_size=1,
            maxlen=None,
            name=None):
        
        # Initailize SeisBench model with pretrained weights
        try:
            self.model = model.from_pretrained(weight)
        except ValueError:
            raise

        if name is None:
            name = f'{model.name}_{weight}'
        else:
            name = f'{name}_{model.name}_{weight}'

        super().__init__(max_pulse_size=max_pulse_size,
                         maxlen=None,
                         name=name)
        sequence = Sequence(
            WindMod()
        )

        

