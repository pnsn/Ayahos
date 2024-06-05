import configparser, logging, sys
from ayahos import Ayahos
from ayahos.submodules import WaveInWyrm, SBMTubeWyrm, PickOutWyrm
import seisbench.models as sbm

Logger = logging.getLogger(__name__)

class AyahosPick(Ayahos):
    def __init__(self, config_file):
        init_kwargs = self.parse_config(config_file)
        try:
            super().__init__(self, **init_kwargs['Ayahos'])
        except:
            Logger.critical('Could not initialize super()')
            sys.exit(1)
        try:
            wiw = WaveInWyrm(self.module, **init_kwargs['WaveInWyrm'])
        except:
            Logger.critical('Could not initialize WaveInWyrm')
            sys.exit(1)
        try:
            stw = SBMTubeWyrm(**init_kwargs['SBMTubeWyrm'])
        except:
            Logger.critical('Could not initialize SBMTubeWyrm')
        try:
            pow = PickOutWyrm(self.module, **init_kwargs['PickOutWyrm - aborting'])
        except:
            Logger.critical('Could not initialize PickOutWyrm - aborting')
        Logger.info('Submodules successfully initialized - appending to AyahosPick.wyrm_dict')
        wyrm_dict = {
            'input':wiw, 'proc': stw, 'output': pow}
        self.update(wyrm_dict)

    def parse_config(self, config_file):
        # Parse configuration file from arguments
        self.config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        self.config.read(config_file)
        breakpoint()
        # Get kwargs for submodules
        init_kwargs = {'Ayahos': {},
                'WaveInWyrm':{},
                "SBMTubeWyrm":{},
                "PickOutWyrm":{}}
        if not all (_k in self.config._sections for _k in init_kwargs.keys()):
            missing_list = [_k for _k in init_kwargs.keys() if _k not in self.config._sections]
            Logger.critical(f'Missing sections from config_file: {missing_list}')
            sys.exit(1)
        for _k in init_kwargs.keys():
            conf_items = dict(self.config[_k].items())
            for _k2, _v in conf_items.items():
                init_kwargs[_k].update({_k2: eval(_v)})
        return init_kwargs