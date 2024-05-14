import threading, logging, time, os, sys
import PyEW
from ayahos.core.wyrms.tubewyrm import TubeWyrm
import pandas as pd

Logger = logging.getLogger(__name__)

# def add_earthworm_to_path(Earthworm_Root='/usr/local/earthworm'):
#     ew_home = os.getenv('EW_HOME')
#     if ew_home:
#         Logger.debug(f'EW_HOME is already exported as {ew_home}')
#     else:
#         os.system(f'export EW_HOME={Earthworm_Root}')


###################################################################################
# HEART WYRM CLASS DEFINITION #####################################################
###################################################################################
class HeartWyrm(TubeWyrm):
    """
    This class encapsulates a PyEW.EWModule object and provides the `run`,
    `start` and `stop` class methods required for running a continuous
    instance of a PyEW.EWModule interface between a running instance
    of Earthworm.

    This class inherits the pulse method from wyrm.core.base_wyrms.TubeWyrm
    and adds the functionality of setting a "wait_sec" that acts as a pause
    between instances of triggering pulses to allow data accumulation.
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
        """Create a HeartWyrm object
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
        """        """
        ChildClass of wyrm.core.base_wyrms.TubeWyrm

        Initialize a HeartWyrm object that contains the parameters neededto
        initialize an EWModule object assocaited with a running instance of
        Earthworm.

        The __init__ method populates the attributes necessary to initialize
        the EWModule object with a subsequent heartwyrm.initialize_module().

        :: INPUTS ::
        :param wait_sec: [float] wait time in seconds between pulses
        :param DR_ID: [int-like] Identifier for default reference memory ring
        :param MOD_ID: [int-like] Module ID for this instace of Wyrms
        :param INST_ID: [int-like] Installation ID (Institution ID)
        :param HB_PERIOD: [float-like] Heartbeat reporting period in seconds
        :param debug: [BOOL] Run module in debug mode?
        :param wyrm_dict: [list-like] iterable set of *wyrm objects
                            with sequentially compatable *wyrm.pulse(x)

        :: PUBLIC ATTRIBUTES ::
        :attrib module: False or [PyEW.EWModule] - Holds Module Object
        :attrib wait_sec: [float] rate in seconds to wait between pulses
        :attrib connections: [pandas.DataFrame]
                            with columns 'Name' and 'Ring_ID' that provides
                            richer indexing and display of connections
                            made to the EWModule via EWModule.add_ring(RING_ID)
                            Updated using heartwyrm.add_connection(RING_ID)
        :attrib wyrm_dict: [list] list of *wyrm objects
                            Inherited from TubeWyrm

        :: PRIVATE ATTRIBUTES ::
        :attrib _default_ring_id: [int] Saved DR_ID input
        :attrib _module_id: [int] Saved MOD_ID input
        :attrib _installation_id: [int] Saved INST_ID input
        :attrib _HBP: [float] Saved HB_PERIOD input
        :attrib _debug: [bool] Saved debug input
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
            os.eval(f'source {ew_env_file}')
            try:
                os.environ['EW_HOME']
            # If not, exit on code 1
            except KeyError:
                Logger.critical(f'Environmental varible $EW_HOME not mapped with environment {ew_home}')
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
        
    def _initialize_module(self, user_check=True):
        """private method: _initialize_module
        Wraps ```PyEW.EWModule.__init__(**self.module_init_kwargs)```
        to initialize the self.module object contained in this HeartWyrm

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
                    Logger.error("HeartWyrm: There is already a EWModule running!")
            elif isinstance(self.module, PyEW.EWModule):
                Logger.error("HeartWyrm: Module already assigned to self.module")
            else:
                Logger.critical(
                    f"HeartWyrm.module is type {type(self.module)}\
                     incompatable!!!"
                )
                sys.exit(1)
            self.add_connection('DEFAULT', self.module_init_kwargs['def_ring'])
        else:
            Logger.critical("User canceled module initialization -> exiting politely")
            sys.exit(0)

    def add_connection(self, name, ring_id):
        """add a connection to the self.module (EWModule) object attached to this HeartWyrm
        and update information in the self.connections attribute

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
        self.module.add_connection(ring_id)
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

    def _early_stopping(self):
        """_early_stopping for ayahos.core.wyrms.heartwyrm.HeartWyrm

        Return status of self.module.mod_sta()
            True - still running
            False - shutdown signal

        :return: self.module.mod_sta
        :rtype: bool
        """        
        return self.module.mod_sta()

    def _core_process(self, x=None):
        """
        _core_process for HeartWyrm

        1) wait for self.wait_sec
        2) execute y = TubeWyrm(...).pulse(x)

        also see ayahos.core.wyrms.tubewyrm.TubeWyrm

        :param x: input collection of objects for first wyrm_ in self.wyrm_dict, defaults to None
        :type x: Varies, optional
        :return: output of last wyrm_ in self.wyrm_dict
        :rtype: Varies
        """        
        # Sleep for wait_sec
        time.sleep(self.wait_sec)
        # Then run TubeWyrm pulse
        y = super().pulse(x)
        return y

    def run(self):
        """
        Run the PyEW.EWModule housed by this HeartWyrm
        """
        Logger.critical("Starting Module Operation")        
        self.start()
        while self.runs:
            # Break while loop if _early_stopping triggers at start of iteration
            if self._early_stopping() is False:
                break
            self._core_process()
            # Use stop to shut down while loop if _early_stopping triggers at end of iteration
            if self._early_stopping() is False:
                self.stop()
        # Gracefully shut down
        self.module.goodbye()
        # Note shutdown in logging
        Logger.critical("Shutting Down Module")     

    def pulse(self):
        Logger.error("pulse() method disabled for ayahos.core.wyrms.heartwyrm.Heartwyrm")
        Logger.error("Use HeartWyrm.run() to start module operation")

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
    #     print("Exiting HeartWyrm Instance")