import sys
from PULSE.module.predict import SeisBenchMod
from PULSE.module.sequence import SequenceMod
from PULSE.module.process import InPlaceMod
from PULSE.util.io import import_from_string

class SeisBenchSequenceMod(SequenceMod):
    
    def __init__(
            self,
            model_class_name,
            weight_names,
            devicetype='cpu',
            sampling_rate=100.,
            sample_overlap=1800,
            channel_fill_rule='cloneZ',
            max_pulse_size=1e6,
            max_output_size=1e7,
            report_period=False,
            meta_memory=60,
            **kwargs

    ):
        
        self.pmethods = ['treat_gaps','sync_to_reference','apply_fill_rule','normalize']

        if any(_k not in self.pmethods for _k in kwargs.keys()):
            for _k in kwargs.keys():
                if _k not in self.pmethods:            
                    self.Logger.critical(f'{_k} is not an approved kwarg')
            self.Logger.critical('exiting')
            sys.exit(1)
        sequence = {}

        seisbenchmod = SeisBenchMod(model_class_name,
                                    weight_names=weight_names,
                                    devicetype=devicetype,
                                    )

        for pmethod in self.pmethods:
            sequence.update({pmethod: })

        for _k, _v in kwargs.items():
