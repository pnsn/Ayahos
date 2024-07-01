"""
:module: PULSE.module.coordinate
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:


Classes
-------
:class:`~PULSE.module.coordinate.PulseMod_EW`
"""
import threading, logging, time, sys, configparser, inspect
from PULSE.util.pyew import is_wave_msg
from PULSE.module.sequence import SequenceMod, SequenceBuilderMod
from PULSE.module.transact import PyEWMod

Logger = logging.getLogger(__name__)


def _init_pyewmodule(modname, cfg_object):

    # Construct PyEW.EWmodule object
    ewmodinit = {
        'connections': eval(cfg_object.get(modname, 'connections')),
        'module_id': cfg_object.getint('Earthworm', 'MOD_ID'),
        'installation_id': cfg_object.getint('Earthworm', 'INST_ID'),
        'heartbeat_period': cfg_object.getfloat('Earthworm', 'HB'),
        'deep_debug': cfg_object.getboolean(modname, 'deep_debug')
        }

    module = PyEWMod(**ewmodinit)
    return module


class PulseMod_EW(SequenceBuilderMod):
    """
    The PULSE_EW class comprises an extended :class:`~PyEW.EWModule` object
    (:class:`~PULSE.module.ew_transact.PyEWMod`) and a sequence of PULSE modules
    that make an operational python module that communicates with the Earthworm message transport system
    
    This class inherits its sequencing methods and attributes from :class:`~PULSE.wyrms.sequencewyrm.sequenceWyrm`
    

    :param wait_sec: number of seconds to wait between pulses, defaults to 0
    :type wait_sec: int, optional
    :param default_ring_id: default ring ID to assign to PULSE module, defaults to 1000
    :type default_ring_id: int, optional
    :param module_id: module ID that Earthworm will see for this PULSE module, defaults to 193
    :type module_id: int, optional
    :param installation_id: installation ID, defaults to 255 - anonymous/nonexchanging installation
    :type installation_id: int, optional
    :param heartbeat_period: time in seconds between heartbeat message sends from this module to Earthworm, defaults to 15
    :type heartbeat_period: int, optional
    :param extra_connections: dictionary with {'NAME': RING_ID} formatting for additional Py<->EW connections
        to make in addition to the 'DEFAULT': default_ring_id connection
        defaults to {'WAVE': 1000, 'PICK': 1005}
    :type: dict, optional
    :param ewmodule_debug: should debugging level logging messages within the PULSE module object
        be included if logging level is set to DEBUG? Defaults to False.
        (acts as an extra nit-picky layer for debugging)
    :type ewmodule_debug: bool, optional
    """
    special_keys = SequenceBuilderMod.special_keys
    special_methods = SequenceBuilderMod.special_methods
    special_methods.update({'module': _init_pyewmodule})

    def __init__(self, config_file, starting_section='PulseMod_EW'):
        """Create a PULSE object
        Inherits the sequence attribute and pulse() method from sequenceWyrm

        :param wait_sec: number of seconds to wait between completion of one pulse sequence of
            the Wyrm-like objects in self.sequence and the next, defaults to 0
        :type wait_sec: float, optional
        :param default_ring_id: default ring ID to assign to EWModule, defaults to 1000
        :type default_ring_id: int, optional
        :param module_id: module ID that Earthworm will see for this EWModule, defaults to 200
        :type module_id: int, optional
        :param installation_id: installation ID, defaults to 255 - anonymous/nonexchanging installation
        :type installation_id: int, optional
        :param heartbeat_period: send heartbeat message to Earthworm every X seconds, defaults to 15
        :type heartbeat_period: int, optional
        :param sequence: dictionary of PULSE.core.wyrms-type objects that will be executed in a chain , defaults to {}
        :type sequence: dict, optional
            also see PULSE.core.wyrms.sequencewyrm.sequenceWyrm
        :param submodule_wait_sec: seconds to wait between execution of the **pulse** method of each
            Wyrm-like object
        """
        # Inherit and build module using SequenceBuilderMod inheritance
        # NOTE: uses polymorphic adaptation of :meth:`~PULSE.module.sequence.SequenceBuilderMod.parse_config_section`
        super().__init__(config_file=config_file,
                         starting_section=starting_section)

        # Construct PyEW.EWmodule object
        if 'Earthworm' not in self.cfg._sections.keys():
            self.Logger.critical('KeyError: "Earthworm" not included in sections')
            sys.exit(1)
        else:
            ewmodinit = {
                'connections': eval(self.cfg.get(self._mname, 'connections')),
                'module_id': self.cfg.getint('Earthworm', 'MOD_ID'),
                'installation_id': self.cfg.getint('Earthworm', 'INST_ID'),
                'heartbeat_period': self.cfg.getfloat('Earthworm', 'HB'),
                'deep_debug': self.cfg.getboolean(self._mname, 'deep_debug')
            }

            self.module = PyEWMod(**ewmodinit)



        # Initialize config parser
        self.cfg = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        # Read configuration file
        self.cfg.read(config_file)

        # Initialize Super for SequenceMod inheritance
        if 'PulseMod_EW' in self.cfg._sections.keys():
            PULSE_init = self.parse_config_section('PulseMod_EW')
            sequence_params = inspect.signature(SequenceMod).parameters
            super_init_kwargs = {}
            for _k, _v in PULSE_init.items():
                if _k in sequence_params.keys():
                    super_init_kwargs.update({_k: _v})
            super().__init__(**super_init_kwargs)
        # Trigger safety catch that something is missing
        else:
            Logger.critical(f'Cannot initialize {super().__name__(full=True)}')
        # Ensure minimum required fields are present for module initialization
        demerits = 0
        for _rs in ['Earthworm','PulseMod_EW','Sequence']:
            if _rs not in self.cfg._sections.keys():
                self.Logger.critical(f'section {_rs} missing from config file! Will not initialize PULSE')
                demerits += 1
        if demerits > 0:
            sys.exit(1)
        else:
            self.Logger.debug('config file has all required sections')

        # Initialize PULSEPyEWModule Object
        self.module = PyEWMod(
            connections = eval(self.cfg.get('PulseMod_EW','connections')),
            module_id = self.cfg.getint('Earthworm', 'MOD_ID'),
            installation_id = self.cfg.getint('Earthworm', 'INST_ID'),
            heartbeat_period = self.cfg.getfloat('Earthworm', 'HB'),
            deep_debug = self.cfg.getboolean('PulseMod_EW', 'deep_debug')
        )        
        # Create a thread for the module process
        try:
            self.module_thread = threading.Thread(target=self.run)
        except:
            self.Logger.critical('Failed to start thread')
            sys.exit(1)
        self.Logger.info('PyEWMod initialized')

        # Sequence submodules
        sequence = {}
        demerits = 0

        # Iterate across submodule names and section names
        for submod_name, submod_section in self.cfg['Sequence'].items():
            # Log if there are missing submodules
            if submod_section not in self.cfg._sections.keys():
                self.Logger.critical(f'submodule {submod_section} not defined in config_file. Will not compile!')
                demerits += 1
            # Construct if the submodule has a section
            else:
                # Parse the class name and __init__ kwargs
                submod_class, submod_init_kwargs = \
                    self.parse_config_section(submod_section)
                # Run import to local scope
                parts = submod_class.split('.')
                path = '.'.join(parts[:-1])
                clas = parts[-1]
                try:
                    exec(f'from {path} import {clas}')
                except ImportError:
                    self.Logger.critical(f'failed to import {submod_class}')
                    sys.exit(1)
                submod_object = eval(clas)(**submod_init_kwargs)
                # Attach object to sequence
                sequence.update({submod_name: submod_object})
                self.Logger.info(f'{submod_name} initialized')
        # If there are any things that failed to compile, exit
        if demerits > 0:
            sys.exit(1)
        
        # Update with non-empty sequence
        self.update(sequence)

        # Set runs flag to True
        self.runs = True
        self.Logger.critical('ALL OK - PULSE Initialized!')

    ###########################################
    ### MODULE CONFIGURATION PARSING METHOD ###
    ###########################################
        
    def parse_config_section(self, section):
        submod_init_kwargs = {}
        submod_class = None
        for _k, _v in self.cfg[section].items():
            # Handle special case where class is passed
            if _k == 'class':
                submod_class = _v
            # Handle special case where module is passed
            elif _k == 'module':
                _val = self.module
            # Handle case where the parameter value is bool-like    
            elif _v in ['True', 'False', 'yes', 'no']:
                _val = self.cfg.getboolean(section, _k)
            # For everything else, use eval statements
            else:
                _val = eval(self.cfg.get(section, _k))
            
            if _k != 'class':
                submod_init_kwargs.update({_k: _val})
        
        if submod_class is None:
            return submod_init_kwargs
        else:
            return submod_class, submod_init_kwargs

    ######################################
    ### MODULE OPERATION CLASS METHODS ###
    ######################################

    def start(self):
        """
        Start Module Command
        runs ```self._thread.start()```
        """
        if len(self.sequence) == 0:
            self.Logger.warning('No *Mod-type sub-/base-modules contained in this PULSE Module')
        self.module_thread.start()

    def stop(self):
        """
        Stop Module Command
        sets self.runs = False
        """
        self.runs = False

    def run(self, input=None):
        """
        Run the PyEW.EWModule housed by this PULSE with the option
        of an initial input that is shown to the first wyrm in 
        this PULSE' sequence attribute.

        :param input: input for the pulse() method of the first Mod in PULSE.sequence, default None
        :type input: varies, optional
        """
        self.Logger.critical("Starting Module Operation")     
        print('Im doing science')   
        while self.runs:
            if self.module.mod_sta() is False:
                break
            time.sleep(0.001)
            if self.module.debug:
                self.Logger.debug('running PULSE pulse')
            # Run pulse inherited from sequenceWyrm
            _ = super().pulse(input)
        # Note shutdown in logging
        self.Logger.critical("Shutting Down Module") 
        # Gracefully shut down
        self.module.goodbye()
    

    def pulse(self):
        """
        Overwrites the inherited :meth:`~PULSE.module.coordinate.SequenceBuildMod.pulse` method
        that broadcasts a pair of logging errors pointing to use PULSE.run()
        as the operational 
        """        
        self.Logger.error("pulse() method disabled for PULSE.core.PULSE.PULSE")
        self.Logger.error("Use PULSE.run() to start module operation")
        return None, None
