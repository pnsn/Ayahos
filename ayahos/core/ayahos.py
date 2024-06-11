"""
Module for handling Ayahos :class: `~ayahos.core.ayahos.Ayahos` objects

:author: Nathan T. Stevens
:org: Pacific Northwest Seismic Network
:email: ntsteven (at) uw.edu
:license: AGPL-3.0
"""
import threading, logging, time, sys, configparser, inspect
from ayahos.wyrms.tubewyrm import TubeWyrm
from ayahos.core.ayahosewmodule import AyahosEWModule

# def add_earthworm_to_path(Earthworm_Root='/usr/local/earthworm'):
#     ew_home = os.getenv('EW_HOME')
#     if ew_home:
#         Logger.debug(f'EW_HOME is already exported as {ew_home}')
#     else:
#         os.system(f'export EW_HOME={Earthworm_Root}')

Logger = logging.getLogger(__name__)
###################################################################################
# AYAHOS CLASS DEFINITION #########################################################
###################################################################################
class Ayahos(TubeWyrm):
    """
    The Ayahos class comprises an extended :class:`~PyEW.EWModule` object
    (:class:`~ayahos.core.ayahosewmodule.AyahosEWModule) and a sequence of 
    :class:`~ayahos.wyrms.wyrm.Wyrm` sub-/base-modules that make an operational
    python module that communicates with the Earthworm message transport system
    
    This class inherits its sequencing methods and attributes from :class:`~ayahos.wyrms.tubewyrm.TubeWyrm`
    

    :param wait_sec: number of seconds to wait between pulses, defaults to 0
    :type wait_sec: int, optional
    :param default_ring_id: default ring ID to assign to AyahosEWModule, defaults to 1000
    :type default_ring_id: int, optional
    :param module_id: module ID that Earthworm will see for this AyahosEWModule, defaults to 193
    :type module_id: int, optional
    :param installation_id: installation ID, defaults to 255 - anonymous/nonexchanging installation
    :type installation_id: int, optional
    :param heartbeat_period: time in seconds between heartbeat message sends from this module to Earthworm, defaults to 15
    :type heartbeat_period: int, optional
    :param extra_connections: dictionary with {'NAME': RING_ID} formatting for additional Py<->EW connections
        to make in addition to the 'DEFAULT': default_ring_id connection
        defaults to {'WAVE': 1000, 'PICK': 1005}
    :type: dict, optional
    :param ewmodule_debug: should debugging level logging messages within the AyahosEWModule object
        be included if logging level is set to DEBUG? Defaults to False.
        (acts as an extra nit-picky layer for debugging)
    :type ewmodule_debug: bool, optional
    """

    def __init__(self, config_file):
        """Create a Ayahos object
        Inherits the wyrm_dict attribute and pulse() method from TubeWyrm

        :param wait_sec: number of seconds to wait between completion of one pulse sequence of
            the Wyrm-like objects in self.wyrm_dict and the next, defaults to 0
        :type wait_sec: float, optional
        :param default_ring_id: default ring ID to assign to EWModule, defaults to 1000
        :type default_ring_id: int, optional
        :param module_id: module ID that Earthworm will see for this EWModule, defaults to 200
        :type module_id: int, optional
        :param installation_id: installation ID, defaults to 255 - anonymous/nonexchanging installation
        :type installation_id: int, optional
        :param heartbeat_period: send heartbeat message to Earthworm every X seconds, defaults to 15
        :type heartbeat_period: int, optional
        :param wyrm_dict: dictionary of ayahos.core.wyrms-type objects that will be executed in a chain , defaults to {}
        :type wyrm_dict: dict, optional
            also see ayahos.core.wyrms.tubewyrm.TubeWyrm
        :param submodule_wait_sec: seconds to wait between execution of the **pulse** method of each
            Wyrm-like object
        """
        # Initialize config parser
        self.cfg = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        # Read configuration file
        self.cfg.read(config_file)

        # Ensure minimum required fields are present for module initialization
        demerits = 0
        for _rs in ['Earthworm','Ayahos','Connections','Build']:
            if _rs not in self.cfg._sections.keys():
                Logger.critical(f'section {_rs} missing from config file! Will not initialize Ayahos')
                demerits += 1
        if demerits > 0:
            sys.exit(1)
        else:
            Logger.debug('config file has all required sections')

        # Get connections
        connections = self.parse_config_section('Connections')

        # Initialize AyahosPyEWModule Object
        self.module = AyahosEWModule(
            connections = connections,
            module_id = self.cfg.getint('Earthworm', 'MOD_ID'),
            installation_id = self.cfg.getint('Earthworm', 'INST_ID'),
            heartbeat_period = self.cfg.getfloat('Earthworm', 'HB'),
            deep_debug = self.cfg.getboolean('Ayahos', 'deep_debug')
        )        
        # Create a thread for the module process
        try:
            self.module_thread = threading.Thread(target=self.run)
        except:
            Logger.critical('Failed to start thread')
            sys.exit(1)
        Logger.info('AyahosEWModule initialized')

        # Build submodules
        wyrm_dict = {}
        demerits = 0

        # Iterate across submodule names and section names
        for submod_name, submod_section in self.cfg['Build'].items():
            # Log if there are missing submodules
            if submod_section not in self.cfg._sections.keys():
                Logger.critical(f'submodule {submod_section} not defined in config_file. Will not compile!')
                demerits += 1
            # Construct if the submodule has a section
            else:
                submod_class, submod_init_kwargs = \
                    self.parse_config_section(submod_section)
                try:
                    exec(f'from ayahos.wyrms import {submod_class}')
                except ModuleNotFoundError:
                    try:
                        exec(f'from ayahos.submodule import {submod_class}')
                    except ModuleNotFoundError:
                        Logger.critical(f'Cannot import class {submod_class} from ayahos.wyrms or ayahos.submodule shortcuts')
                breakpoint()
                submod_object = eval(submod_class)(**submod_init_kwargs)
                wyrm_dict.update({submod_name: submod_object})
                Logger.info(f'{submod_class} object initialized')
        # If there are any things that failed to compile, exit
        if demerits > 0:
            sys.exit(1)

        
        ayahos_init = self.parse_config_section('Ayahos')
        tube_params = inspect.signature(TubeWyrm).parameters
        tube_init = {'wyrm_dict': wyrm_dict}
        for _k, _v in ayahos_init.items():
            if _k in tube_params.keys():
                tube_init.update({_k: _v})
        breakpoint()

        # Initialize tubewyrm inheritance
        super().__init__(**tube_init)

        # Set runs flag to True
        self.runs = True
        Logger.critical('ALL OK - Ayahos Initialized!')

    ###########################################
    ### MODULE CONFIGURATION PARSING METHOD ###
    ###########################################
        
    def parse_config_section(self, section):
        submod_init_kwargs = {}
        submod_class = None
        for _k, _v in self.cfg[section].items():
            # Handle special case where class is passed
            if _k == 'class':
                # Ensure wyrm is imported
                # TODO: See if this works...
                exec(f'from ayahos.wyrms import {_v}')
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
        if len(self.wyrm_dict) == 0:
            Logger.warning('No Wyrm-type sub-/base-modules contained in this Ayahos Module')
        self.module_thread.start()

    def stop(self):
        """
        Stop Module Command
        sets self.runs = False
        """
        self.runs = False

    def run(self, input=None):
        """
        Run the PyEW.EWModule housed by this Ayahos with the option
        of an initial input that is shown to the first wyrm in 
        this Ayahos' wyrm_dict attribute.

        :param input: input for the pulse() method of the first wyrm in Ayahos.wyrm_dict, default None
        :type input: varies, optional
        """
        Logger.critical("Starting Module Operation")     
        print('Im doing science')   
        while self.runs:
            if self.module.mod_sta() is False:
                break
            time.sleep(0.001)
            if self.module.debug:
                Logger.debug('running main pulse')
            # Run pulse for 
            _ = super().pulse(input)
            super().update_summary_metrics()
            if self.time_to_report_summary():
                self.transmit_summary()
        # Note shutdown in logging
        Logger.critical("Shutting Down Module") 
        # Gracefully shut down
        self.module.goodbye()
    

    def pulse(self):
        """
        Overwrites the inherited :meth:`~ayahos.wyrms.tubewyrm.Tubewyrm.pulse` method
        that broadcasts a pair of logging errors pointing to use Ayahos.run()
        as the operational 
        """        
        Logger.error("pulse() method disabled for ayahos.core.ayahos.Ayahos")
        Logger.error("Use Ayahos.run() to start module operation")
        return None, None
    



    ### REPORTING METHODS ###

    def time_to_report_summary(self):
        if self.last_report_time is None:
            self.last_report_time = time.time()
            answer = True
        else:
            now = time.time()
            dt = now - self.last_report_time
            if dt >= self.reporting_interval:
                answer = True
                self.last_report_time = now
            else:
                answer = False
        return answer
    
    def transmit_summary(self):
        for _k in self.wyrm_dict.keys():
            summary_line = self.summary[_k]
            Logging.info('Summary Report')
            Logging.info()

    ########################
    # CONSTRUCTION METHODS #
    # def init_from_config(self, config_file):
    #     raise NotImplementedError("Work in progress. GOAL: populate whole modules from a single configparser.ConfigParser compliant config file")

    # # def run(self):
    #     """
    #     Module Execution Command
    #     """
    #     while self.runs:
    #         if self.module.mod_sta() is False:
    #             break
    #         time.sleep(self.wait_sec)
    #         # Run Pulse Command inherited from TubeWyrm
    #         self.pulse()  # conn_in = conn_in, conn_out = conn_out)
    #     # Polite shut-down of module
    #     self.module.goodbye()
    #     print("Exiting Ayahos Instance")



       ##########################################
    # MODULE INITIALIZATION HELPER FUNCTIONS #
    ##########################################
        
    # def _initialize_module(self, user_check=False):
    #     """private method: _initialize_module
    #     Wraps ```PyEW.EWModule.__init__(**self.module_init_kwargs)```
    #     to initialize the self.module object contained in this Ayahos

    #     :param user_check: should the pre-initialization user input check occur? Defaults to True
    #     :type user_check: bool, optional
    #     :raises RuntimeError: Raisese error if the EWModule is already running
    #     """        
    #     if user_check:
    #         cstr = "About to initialize the following PyEW.EWModule\n"
    #         for _k, _v in self.module_init_kwargs.items():
    #             cstr += f'{_k}: {_v}\n'
    #         cstr += "\n Do you want to continue? [(y)/n]"
    #         ans = input(cstr)
    #         if ans.lower().startswith("y") or ans == "":
    #             user_continue = True
    #         elif ans.lower().startswith("n"):
    #             user_continue = False
    #         else:
    #             Logger.critical("Invalid input -> exiting")
    #             sys.exit(1)
    #     else:
    #         user_continue = True
    #     if user_continue:
    #         # Initialize PyEarthworm Module
    #         if not self.module:
    #             try:
    #                 self.module = PyEW.EWModule(**self.module_init_kwargs)
    #             except RuntimeError:
    #                 Logger.error("There is already a EWModule running!")
    #         elif isinstance(self.module, PyEW.EWModule):
    #             Logger.error("Module already assigned to self.module")
    #         else:
    #             Logger.critical(
    #                 f"module is type {type(self.module)} - incompatable!!!"
    #             )
    #             sys.exit(1)
    #         self.add_connection('DEFAULT', self.module_init_kwargs['def_ring'])
    #     else:
    #         Logger.critical("User canceled module initialization -> exiting politely")
    #         sys.exit(0)

    # def add_connection(self, name, ring_id):
    #     """add a connection to the self.module (EWModule) object 
    #     attached to Ayahos and update information in the self.connections attribute

    #     :param name: human-readable name to use as a key in self.connections
    #     :type name: str or int
    #     :param ring_id: earthworm ring ID (value falls in the range [0, 9999])
    #     :type ring_id: int
    #     :return connections: a view of the connections attribute of this Ayahos object
    #     :rtype connections: dict
    #     """
    #     # === RUN COMPATABILITY CHECKS ON INPUT VARIABLES === #
    #     if not isinstance(name, (str, int)):
    #         raise TypeError(f'name must be type str or int, not {type(name)}')
    #     elif name in self.connections.keys():
    #         raise KeyError(f'name {name} is already claimed as a key for a connection in this module')
        
    #     # Enforce integer RING_ID type
    #     if not isinstance(ring_id, int):
    #         raise TypeError
    #     elif ring_id < 0:
    #         raise ValueError
    #     elif ring_id > 10000:
    #         raise ValueError
    #     else:
    #         pass
    #     self.module.add_ring(ring_id)
    #     idx = len(self.connections)
    #     self.connections.update({name: (idx, ring_id)})
    #     return self.connections