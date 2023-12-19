"""
:module: wyrm.core.sequential
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains class definitions for Wyrms that execute
    sequences of pulse(x) commands for queues of other Wyrm class
    objects.

:class synopses & family tree:
Wyrm - ð
=--> TubeWyrm - pulse(x)
    |           executes pulse(x) commands in sequence passing the 
    |           output of *wyrm.pulse(x) to the input of the next
    |           i.e., 
    |                y = wyrm3.pulse(wyrm2.pulse(wyrm1.pulse(x)))
    |           Symbolic Representation: < >
    |
    =--> CanWyrm - pulse(x)
    |           execute pulse(x) commands in sequence, reading from a
    |           shared input x and concatenating each output y from
    |           the sequence
    |           i.e., 
    |                y = deque([wyrm3.pulse(x), wyrm2.pulse(x), wyrm1.pulse(x)])
    |           Symbolic Representation: | |
    |
    =--> HeartWyrm - pulse(x)
                child-class of TubeWyrm that adds attributes for housing
                the PyEW.EWModule object that orchestrates connections between
                Python and Earthworm 
                 
                Adds the necessary class-methods for continuous operation:
                     start(), stop(), and run()
                Symbolic Representation: { }

:attribution:
    This module builds on the PyEarthworm (C) 2018 F. Hernandez interface
    between an Earthworm Message Transport system and Python distributed
    under an AGPL-3.0 license.

"""
import PyEW
from pandas import DataFrame
from time import time
from collections import deque
from threading import Thread
from wyrm.core.wyrm import Wyrm


class TubeWyrm(Wyrm):
    """
    Base Class facilitating chained execution of pulse(x) class methods
    for a sequence wyrm objects, with each wyrm.pulse(x) taking the prior
    member's pulse(x) output as its input. 
    This `wyrm_queue` is a double ended queue (collections.deque), 
    which provides easier append/pop syntax for editing the wyrm_queue.

    Convenience methods for appending and popping entries from the processing
    queue are provided
    """

    def __init__(self, wyrm_queue=deque([])):
        """
        Create a tubewyrm object
        :: INPUT ::
        :param wyrm_list: [deque] or [list]
                            double ended queue of Wyrm objects
                            if list is provided, will be automatically
                            converted into a deque

        :: OUTPUT ::
        Initialized TubeWyrm object
        """
        # Run compatability checks on wyrm_list
        # If given a single Wyrm, wrap it in a deque
        if isinstance(wyrm_queue, Wyrm):
            self.wyrm_queue = deque([wyrm_queue])
        # If given a list of candidate wyrms, ensure they are all of Wyrm class
        elif isinstance(wyrm_queue, (list, deque)):
            if any(not isinstance(_wyrm, Wyrm) for _wyrm in wyrm_queue):
                print('Not all entries of wyrm_queue are type <class Wyrm>')
                raise TypeError
            # If all members are Wyrms, write to attribute
            else:
                self.wyrm_queue = wyrm_queue
            # Final check that the wyrm_queue is a deque
            if isinstance(wyrm_queue, list):
                self.wyrm_queue = deque(wyrm_queue)
                
        # In any other case:
        else:
            print('Provided wyrm_list was not a list or a Wyrm')
            raise TypeError

    def __repr__(self):
        rstr = '--- Tube ---\n'
        for _i, _wyrm in enumerate(self.wyrm_queue):
            if _i == 0:
                rstr += '(head) '
            else:
                rstr += '       '
            rstr += f'{type(_wyrm)}'
            if _i == len(self.wyrm_queue) - 1:
                rstr += ' (tail)'
            rstr += '\n'

    def append(self, object, end='right'):
        """
        Convenience method for left/right append
        to wyrm_queue

        :: INPUTS ::
        :param object: [Wyrm] candidate wyrm object
        :param end: [str] append side 'left' or 'right'

        :: OUTPUT ::
        None
        """
        if isinstance(object, Wyrm):
            if end.lower() in ['right','r']:
                self.wyrm_list.append(object)
            elif end.lower() in ['left', 'l']
                self.wyrm_queue.appendleft(object)
        
        if isinstance(object, (list, deque)):
            if all(isinstance(_x, Wyrm) for _x in object):
                if end.lower() in ['right', 'r']:
                    self.wyrm_list += deque(object)
                elif end.lower() in ['left', 'l']:
                    self.wyrm_list = deque(object) + self.wyrm_list

        
    def pop(self, end='right'):
        """
        Convenience method for left/right pop
        from wyrm_queue

        :: INPUT ::
        :param end: [str] 'left' or 'right'
        
        :: OUTPUT ::
        :param x: [Wyrm] popped Wyrm object from
                wyrm_queue
        """
        if end.lower() in ['right', 'r']:
            x = self.wyrm_list.pop()
        elif end.lower() in ['left','l']:
            x = self.wyrm_list.popleft()
        return x
    
    def pulse(self, x):
        """
        Initiate a chained pulse for elements of 
        wyrm_queue. 

        E.g., 
        tubewyrm.wyrm_queue = [<wyrm1>, <wyrm2>, <wyrm3>]
        y = tubewyrm.pulse(x) 
            is equivalent to 
        y = wyrm3.pulse(wyrm2.pulse(wyrm1.pulse(x)))

        :: INPUT ::
        :param x: Input `x` for the first Wyrm object in wyrm_queue
        
        :: OUTPUT ::
        :param y: Output `y` from the last Wyrm object in wyrm_queue 
        """
        for _wyrm in self.wyrm_list:
            x = _wyrm.pulse(x)
        y = x
        return y


class CanWyrm(TubeWyrm):
    """
    Child class of TubeWyrm.
    It's pulse(x) method runs the queue of *wyrm_n.pulse(x)'s
    sourcing inputs from a common input `x` and creating a queue
    of each wyrm_n.pulse(x)'s output `y_n`.

    NOTE: This class runs items in serial, but with some modification
    this would be a good candidate class for orchestraing multiprocessing.
    """

    # Inherits __init__ from TubeWyrm
    def __init__(self, wyrm_queue=deque([]), output_method=deque):
        super().__init__(wyrm_queue=wyrm_queue)
        self.output_format = output_method

    def __repr__(self):
        rstr = '~~~ CanWyrm ~~~'
        rstr += super().__repr__()
        return rstr

    def pulse(self, x):
        """
        Iterate across wyrms in wyrm_queue that all feed
        from the same input variable x and gather each
        iteration's output y in a deque assembled with an
        appendleft() at the end of each iteration

        :: INPUT ::
        :param x: [variable] Single variable that every
                    Wyrm (or the head-wyrm in a tubewyrm)
                    in the wyrm_queue can accept as a 
                    pulse(x) input.
        
        :: OUTPUT ::
        :return y: [deque] or [self.output_method]
                    Serially assembled outputs of each Wyrm
                    (or the tail-wyrm in a tubewyrm)
        """
        y = self.output_format()
        for _wyrm in self.wyrm_queue:
            _y = _wyrm.pulse(x)
            _y = self.output_format(_y)
            y.appendleft(_y)
        return y
    

class HeartWyrm(TubeWyrm):
    """
    This class encapsulates a PyEW.EWModule object and provides the `run`,
    `start` and `stop` class methods required for running a continuous
    instance of a PyEW.EWModule interface between a running instance
    of Earthworm.

    This class inherits the pulse method from wyrm.core.base_wyrms.TubeWyrm
    and adds the functionality of setting a "pulse_rate" that acts as a pause
    between instances of triggering pulses to allow data accumulation.
    """

    def __init__(
        self, pulse_rate, DR_ID, MOD_ID, INST_ID, HB_PERIOD, debug=False, wyrm_list=[]
    ):
        """
        ChildClass of wyrm.core.base_wyrms.TubeWyrm

        Initialize a HeartWyrm object that contains the parameters neededto
        initialize an EWModule object assocaited with a running instance of
        Earthworm.

        The __init__ method populates the attributes necessary to initialize
        the EWModule object with a subsequent heartwyrm.initialize_module().

        :: INPUTS ::
        :param pulse_rate: [float] rate in seconds to wait between pulses
        :param DR_ID: [int-like] Identifier for default reference memory ring
        :param MOD_ID: [int-like] Module ID for this instace of Wyrms
        :param INST_ID: [int-like] Installation ID (Institution ID)
        :param HB_PERIOD: [float-like] Heartbeat reporting period in seconds
        :param debug: [BOOL] Run module in debug mode?
        :param wyrm_list: [list-like] iterable set of *wyrm objects
                            with sequentially compatable *wyrm.pulse(x)

        :: PUBLIC ATTRIBUTES ::
        :attrib module: False or [PyEW.EWModule] - Holds Module Object
        :attrib pulse_rate: [float] rate in seconds to wait between pulses
        :attrib connections: [pandas.DataFrame]
                            with columns 'Name' and 'Ring_ID' that provides
                            richer indexing and display of connections
                            made to the EWModule via EWModule.add_ring(RING_ID)
                            Updated using heartwyrm.add_connection(RING_ID)
        :attrib wyrm_list: [list] list of *wyrm objects
                            Inherited from TubeWyrm
        
        :: PRIVATE ATTRIBUTES ::
        :attrib _default_ring_id: [int] Saved DR_ID input
        :attrib _module_id: [int] Saved MOD_ID input
        :attrib _installation_id: [int] Saved INST_ID input
        :attrib _HBP: [float] Saved HB_PERIOD input
        :attrib _debug: [bool] Saved debug input
        """
        super().__init__(wyrm_list)
        # Public Attributes
        self.module = False
        self.pulse_rate = pulse_rate
        self.conn_info = DataFrame(columns=["Name", "RING_ID"])

        # Private Attributes
        self._default_ring_id = int(DR_ID)
        self._module_id = int(MOD_ID)
        self._installation_id = int(INST_ID)
        self._HBP = HB_PERIOD
        self._debug = debug

        # Module run attributes
        # Threading - TODO - need to understand this better
        self._thread = Thread(target=self.run)
        self.runs = True

    def __repr__(self):
        # Start with TubeWyrm __repr__ 
        rstr = super().__repr__()
        # List Pulse Rate
        rstr += f"Pulse Rate: {self.pulse_rate:.4f} sec\n"
        # List Module Status and Parameters
        if isinstance(self.module, PyEW.EWModule):
            rstr += "Module: Initialized\n"
        else:
            rstr += "Module: NOT Initialized\n"
        rstr += f'MOD: {self._module_id}'
        rstr += f'DR: {self._default_ring_id}\n'
        rstr += f'INST: {self._installation_id}\n'
        rstr += f'HB: {self._HBP} sec\n'
        rstr == f'debug: {self._debug}\n'
        # List Connections
        rstr += "---- Connections ----\n"
        rstr += f"{self.conn_info}\n"
        rstr += "-------- END --------\n"
        return rstr

    def initialize_module(self, user_check=True):
        if user_check:
            cstr = 'About to initialize the following PyEW.EWModule\n'
            cstr += f'Default ring ID: {self._default_ring_id:d}\n'
            cstr += f'Moudle ID: {self._module_id:d}\n'
            cstr += f'Inst. ID: {self._installation_id:d}\n'
            cstr += f'Heartbeat: {self._HBP:.1f} sec\n'
            cstr += f'Debug?: {self._debug}\n'
            cstr += '\n Do you want to continue? [(y)/n]'
            ans = input(cstr)
            if ans.lower().startswith('y') or ans == '':
                user_continue = True
            elif ans.lower().startswith('n'):
                user_continue = False
            else:
                print('Invalid input -> exiting')
                exit()
        else:
            user_continue = True
        if user_continue:
            # Initialize PyEarthworm Module
            if not self.module:
                try:
                    self.module = PyEW.EWModule(
                        self._default_ring_id,
                        self._module_id,
                        self._installation_id,
                        self._HBP,
                        debug=self._debug,
                    )
                except RuntimeError:
                    print("HeartWyrm: There is already a EWModule running!")
            elif isinstance(self.module, PyEW.EWModule):
                print("HeartWyrm: Module already initialized")
            else:
                print(
                    f"HeartWyrm.module is type {type(self.module)}\
                    -- incompatable!!!"
                )
                raise RuntimeError
        else:
            print('Canceling module initialization -> exiting')
            exit()

    def add_connection(self, RING_ID, RING_Name):
        """
        Add a connection between target ring and the initialized self.module
        and update the conn_info DataFrame.

        Method includes safety catches
        """
        # === RUN COMPATABILITY CHECKS ON INPUT VARIABLES === #
        # Enforce integer RING_ID type
        try:
            RING_ID = int(RING_ID)
        except ValueError:
            print(
                f"Invalid RING_ID type (wants int-like) -\
                   input type is {type(RING_ID)}"
            )

        # Warn on non-standard RING_Name types and convert to String
        if not isinstance(RING_Name, (int, float, str)):
            print(
                f"Warning, RING_Name is not type (int, float, str) -\
                   input type is {type(RING_Name)}"
            )
            print("Converting RING_Name to <type str>")
            RING_Name = str(RING_Name)

        # --- End Input Compatability Checks --- #

        # === RUN CHECKS ON MODULE === #
        # If the module is not already initialized, try to initialize module
        if not self.module:
            self.initialize_module()
        # If the module is already initialized, pass
        elif isinstance(self.module, PyEW.EWModule):
            pass
        # Otherwise, raise TypeError with message
        else:
            print(f"Module type {type(self.module)} is incompatable!")
            raise TypeError
        # --- End Checks on Module --- #

        # === MAIN BLOCK === #
        # Final safety check that self.module is an EWModule object
        if isinstance(self.module, PyEW.EWModule):
            # If there isn't already an established connection to a given ring
            if not any(self.conn_info.RING_ID == RING_ID):
                # create new connection
                self.module.add_connection(RING_ID)

                # If this is the first connection logged, populate conn_info
                if len(self.conn_info) == 0:
                    self.conn_info = pd.DataFrame(
                        {"RING_Name": RING_Name, "RING_ID": RING_ID}, index=[0]
                    )

                # If this is not the first connection, append to conn_info
                elif len(self.conn_info) > 0:
                    new_conn_info = pd.DataFrame(
                        {"RING_Name": RING_Name, "RING_ID": RING_ID},
                        index=[self.conn_info.index[-1] + 1],
                    )
                    self.conn_info = pd.concat(
                        [self.conn_info, new_conn_info], axis=0, ignore_index=False
                    )

            # If connection exists, notify and provide the connection IDX in the notification
            else:
                idx = self.conn_info[self.conn_info.RING_ID == RING_ID].index[0]
                print(
                    f"IDX {idx:d} -- connection to RING_ID {RING_ID:d}\
                       already established"
                )

        # This shouldn't happen, but if somehow we get here...
        else:
            print(f"Module type {type(self.module)} is incompatable!")
            raise RuntimeError

        
    ######################################
    ### MODULE OPERATION CLASS METHODS ###
    ######################################
        
    def start(self):
        """
        Start Module Command
        """
        self._thread.start()

    def stop(self):
        """
        Stop Module Command
        """
        self.runs = False

    def run(self):
        """
        Module Execution Command
        """
        while self.runs:
            if self.module.mod_sta() is False:
                break
            time.sleep(self.pulse_rate)
            # Run Pulse Command inherited from TubeWyrm
            self.pulse()#conn_in = conn_in, conn_out = conn_out)
        # Polite shut-down of module
        self.module.goodbye()
        print("Exiting HeartWyrm Instance")
