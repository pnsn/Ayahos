"""
:module: wyrm.util.earthworm
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose: This module (under development) will host methods for loading information
        from the mapped earthworm config files (*.d and *.desc) to serve as cross-checks against PyEW and 
        Ayohos
"""
import argparse, configparser, logging

Logger = logging.getLogger('ayahos.util.earthworm.py')

def read_config_file(conffile):
    config = configparser.ConfigParser()
    config.read(conffile)
    return config

def parse_wyrm_config_section(config, section):
    required_fields = ['CLASS']
    if section not in config.keys():
        raise KeyError(f"{section} is not in config's keys")
    elif not all(x in config[section].keys() for x in required_fields):
        raise KeyError(f"not all required keys are present in this section")
    else:
        pass
        cfgsect = config[section]
# Check if class is imported
    if cfgsect['CLASS'] not in dir():
        Logger.critical(f'{cfgsect['CLASS']} not in current scope')
    else:
        pass

    kwargs = {}
    for _k, _v in cfgsect:
        # Pass on CLASS
        if _k.upper() == 'CLASS':
            iclass = _v
            continue
        # Parse float/int arguments
        elif _v.isnumeric():
            _v = float(_v)
            if _v == int(_v):
                _v = int(_v)
        # Parse dictionary like entries
        elif '{' in _v and '}' in _v:
            _v = eval(_v)
            for _k2, _v2 in _v.items():
                if _v2.isnumeric():
                    _v2 = float(_v2)
                    if _v2 == int(_v2):
                        _v2 = int(_v2)
        kwargs.update({_k:_v})
    instruction = {iclass: kwargs}
    return instruction
        
def exec_init_instructions(instructions):
    for _k, _v in instructions
        exec(f'{instructions}')

        
        
    