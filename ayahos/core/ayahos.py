"""
Module for handling Ayahos :class: `~ayahos.core.ayahos.Ayahos` objects

:author: Nathan T. Stevens
:org: Pacific Northwest Seismic Network
:email: ntsteven (at) uw.edu
:license: AGPL-3.0
"""
import threading, logging, time, os, sys
import PyEW
from ayahos.wyrms.tubewyrm import TubeWyrm

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
    The Ayahos class encapsulates a PyEW.EWModule object and provides the `run`,
    `start` and `stop` class methods required for running a continuous
    instance of a PyEW.EWModule interface between a running instance
    of Earthworm.

    This class inherits from wyrm.core.wyrms.tubewyrm.TubeWyrm to house 
    and orchestrate sequenced operations of wyrm submodules
    """

    def __init__(
        self,
        ew_env_file=None,
        wait_sec=0,
        default_ring_id=1000,
        module_id=200,
        installation_id=255,
        heartbeat_period=15,
        module_debug = False,
        conn_dict = {},
        wyrm_dict={}
    ):
        """Create a Ayahos object
        Inherits the wyrm_dict attribute and pulse() method from TubeWyrm

        :param ew_env_file: filepath for the desired EW environment to source, defaults to None
        :type ew_env_file: str or None, optional
        :param wait_sec: number of seconds to wait between pulses, defaults to 0
        :type wait_sec: int, optional
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
        """
        # Initialize TubeWyrm inheritance
        super().__init__(
            wyrm_dict=wyrm_dict,
            wait_sec=wait_sec,
            max_pulse_size=1)

        # Check that earthworm is in path before initializing PyEW modules
        if ew_env_file is None:
            # Check if EW_HOME is in $PATH
            try:
                os.environ['EW_HOME']
            # If not, exit on code 1
            except KeyError:
                Logger.critical('Environmental varible $EW_HOME not mapped - cannot proceed')
                sys.exit(1)
        else:
            os.system(f'source {ew_env_file}')
            try:
                os.environ['EW_HOME']
            # If not, exit on code 1
            except KeyError:
                Logger.critical(f'Environmental varible $EW_HOME not mapped with environment {ew_env_file}')
                sys.exit(1)      

        # Compatability check for default_ring_id          
        if isinstance(default_ring_id, int):
            if 0 <= default_ring_id < 1e5:
                self._default_ring_id = default_ring_id
            else:
                raise ValueError(f'default_ring_id must be an integer in [0, 10000). {default_ring_id} out of bounds')
        else:
            raise TypeError(f'default_ring_id must be type int, not {type(default_ring_id)}')
        
        # Compatability check for module_id
        if isinstance(module_id, int):
            if 0 <=module_id < 1e5:
                pass
            else:
                raise ValueError(f'module_id must be an integer in [0, 10000). {module_id} out of bounds')
        else:
            raise TypeError(f'module_id must be type int, not {type(module_id)}')
        
        # Compatability check for default_ring_id
        if isinstance(default_ring_id, int):
            if 0 <=default_ring_id < 1e5:
                pass
            else:
                raise ValueError(f'default_ring_id must be an integer in [0, 10000). {default_ring_id} out of bounds')
        else:
            raise TypeError(f'default_ring_id must be type int, not {type(default_ring_id)}')
        
        # Create connection holder
        self.connections = {}

        # Assemble module initialization kwargs
        self.module_init_kwargs = {
            'def_ring': default_ring_id,
            'mod_id': module_id,
            'inst_id' :installation_id,
            'hb_time': heartbeat_period,
            'db': module_debug}
        
        # Placeholder for module
        self.module = False
        # Initialize processing thread
        self._thread = threading.Thread(target=self.run)

        # Initialize EWModule object
        self._initialize_module(user_check = True)

        # Initialize module connections from conn_dict
        if isinstance(conn_dict, dict):
            if all(isinstance(_v, int) for _v in conn_dict.values()):
                for _k, _v in conn_dict.items():
                    if _k not in self.connections.keys():
                        self.add_connection(_k, _v)
                    else:
                        Logger.critical(f'Ring Name {_k} already assigned')
                        sys.exit(1)
        # Allow the module to run when self.run is next called
        self.runs = True


    ##########################################
    # MODULE INITIALIZATION HELPER FUNCTIONS #
    ##########################################
        
    def _initialize_module(self, user_check=False):
        """private method: _initialize_module
        Wraps ```PyEW.EWModule.__init__(**self.module_init_kwargs)```
        to initialize the self.module object contained in this Ayahos

        :param user_check: _description_, defaults to True
        :type user_check: bool, optional
        :raises RuntimeError: _description_
        """        
        if user_check:
            cstr = "About to initialize the following PyEW.EWModule\n"
            for _k, _v in self.module_init_kwargs.items():
                cstr += f'{_k}: {_v}\n'
            cstr += "\n Do you want to continue? [(y)/n]"
            ans = input(cstr)
            if ans.lower().startswith("y") or ans == "":
                user_continue = True
            elif ans.lower().startswith("n"):
                user_continue = False
            else:
                Logger.critical("Invalid input -> exiting")
                sys.exit(1)
        else:
            user_continue = True
        if user_continue:
            # Initialize PyEarthworm Module
            if not self.module:
                try:
                    self.module = PyEW.EWModule(**self.module_init_kwargs)
                except RuntimeError:
                    Logger.error("There is already a EWModule running!")
            elif isinstance(self.module, PyEW.EWModule):
                Logger.error("Module already assigned to self.module")
            else:
                Logger.critical(
                    f"module is type {type(self.module)} - incompatable!!!"
                )
                sys.exit(1)
            self.add_connection('DEFAULT', self.module_init_kwargs['def_ring'])
        else:
            Logger.critical("User canceled module initialization -> exiting politely")
            sys.exit(0)

    def add_connection(self, name, ring_id):
        """add a connection to the self.module (EWModule) object 
        attached to Ayahos and update information in the self.connections attribute

        :param name: human-readable name to use as a key in self.connections
        :type name: any, recommend str or int
        :param ring_id: earthworm ring ID (value falls in the range [0, 9999])
        :type ring_id: int
        """
        # === RUN COMPATABILITY CHECKS ON INPUT VARIABLES === #
        # Enforce integer RING_ID type
        if not isinstance(ring_id, int):
            raise TypeError
        elif ring_id < 0:
            raise ValueError
        elif ring_id > 10000:
            raise ValueError
        else:
            pass
        self.module.add_ring(ring_id)
        idx = len(self.connections)
        self.connections.update({name: (idx, ring_id)})


    ######################################
    ### MODULE OPERATION CLASS METHODS ###
    ######################################

    def start(self):
        """
        Start Module Command
        runs ```self._thread.start()```
        """
        self._thread.start()

    def stop(self):
        """
        Stop Module Command
        sets self.runs = False
        """
        self.runs = False

    def unit_process(self, x):
        """
        unit_process for Ayahos inherited from TubeWyrm.unit_process()

        1) wait for self.wait_sec
        2) execute y = TubeWyrm(...).pulse(x)

        also see ayahos.core.wyrms.tubewyrm.TubeWyrm

        :param x: input collection of objects for first wyrm_ in self.wyrm_dict, defaults to None
        :type x: Varies, optional
        """
        # Sleep for wait_sec
        time.sleep(self.wait_sec)
        # Then run TubeWyrm unit_process
        status = super().pulse(x)
        return status

    def run(self, input=None):
        """
        Run the PyEW.EWModule housed by this Ayahos with the option
        of an initial input

        :param input: standard input for the pulse() method of the first wyrm in Ayahos.wyrm_dict, default None
        :type input: varies, optional
        """
        Logger.critical("Starting Module Operation")        
        while self.runs:
            Logger.debug('running main pulse')
            output, nproc = super().pulse(input)
            if self.module.mod_sta() is False:
                break
        # Gracefully shut down
        self.module.goodbye()
        # Note shutdown in logging
        Logger.critical("Shutting Down Module")     

    def pulse(self):
        Logger.error("pulse() method disabled for ayahos.core.ayahos.Ayahos")
        Logger.error("Use Ayahos.run() to start module operation")

    ########################
    # CONSTRUCTION METHODS #
    def init_from_config(self, config_file):
        raise NotImplementedError("Work in progress. GOAL: populate whole modules from a single configparser.ConfigParser compliant config file")

    # def run(self):
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