"""
TODO: Merge with Ayahos
"""

import configparser, logging, sys
from ayahos import Ayahos
# from ayahos.submodules import WaveInWyrm, SBMTubeWyrm, PickOutWyrm
# import seisbench.models as sbm

Logger = logging.getLogger(__name__)


class EWModuleConstructor(Ayahos):
    """
    A generalized module construction class that requires 
    """
    def __init__(self, config_file):
        # Get initialization kwargs
        self.initkw = self.parse_config(config_file)
        
        # Initialize Ayahos inheritance
        try:
            super().__init__(**self.initkw['Ayahos'])
        except:
            Logger.critical('failed to initialize Ayahos')
        # Initialize Wyrm_Dict Elements from Build
        for _ename, _esection in self.initkw['Build'].items():
            ekwargs = self.initkw[_esection].copy()
            # Ensure class is defined for each element
            if 'class' in ekwargs.keys():
                eclass = ekwargs.pop('class')
            else:
                Logger.critical(f'{_esection} in config file does not have an attribute "class"')
                sys.exit(1)
            # Update placeholder value with initalized AyahosEWModule
            if 'module' in ekwargs.keys():
                ekwargs.update({'module': self.module})
            # Attempt to construct the module element
            try:
                self.update({_ename: eclass(**ekwargs)})
            except:
                Logger.critical(f'failed to initialize {_ename} (type {eclass})')
                sys.exit(1)

    def parse_config(self, config_file):
        """Parse a configparser-formatted config_file for input parameter
        sets to submodules contained within this Ayahos module.

        :param config_file: (path) and name of config file to use
        :type config_file: str
        :return instructions: dictionary keyed to config file section headers with parsed values as key:value
            pairs in a subdictionary
        :rtype init_kwargs: dict of dicts
        """        
        # Set entended interpolation
        self.config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        # Read config file
        self.config.read(config_file)
        # Create holder for parsed config contents
        instructions = {}
        # Iterate across sections
        for section, entries in self.config._sections:
            # If this is a new section
            if section not in instructions.keys():
                instructions.update({section: {}})
            else:
                Logger.critical(f'section {section} already parsed - cannot have duplicate section names!')
                sys.exit(1)
            # Iterate across entries in this section
            for _k, _v in entries.items():
                # Ensure 'class' is imported into this scope
                if _k == 'class':
                    if _k not in dir():
                        Logger.critical(f'class {_k} not in scope')
                        sys.exit(1)
                # If module is specified, put a placeholder in (the module isn't initialized yet)
                if _k == 'module':
                    _val = "intentional placeholder"
                # If a boolean-like value is passed, use the getboolean special parser
                elif _v in ['True','False','yes','no']:
                    _val = self.config.getboolean(section, _k)
                # Otherwise use a general eval() statement to parse the entry's value
                else:
                    _val = eval(self.config.get(section, _k))
                # Update instructions
                instructions[section].update({_k: _val})
        # Return instructions
        return instructions
