"""
:module: wyrm.core.coordinate
:auth: Nathan T Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains class defining submodules for coordinating sets
    of wyrms to form parts of fully operational Wyrm modules, including:

    TubeWyrm - a submodule class for running a sequence of wyrms' pulse()
              methods in series starting with a single input to the first
              wyrm in the series and providing a single output from the
              last wyrm in the series

              y = wyrmN.pulse(...pulse(wyrm1.pulse(wyrm0.pulse(x))))
    
    CanWyrm - a submodule class for running a sequence of wyrms' pulse()
                methods in parallel starting with a single input and returning
                a dictionary of each wyrm's output

                y = {id0: wyrm0.pulse(x),
                     id1: wyrm1.pulse(x),
                     ...
                     idN: wyrmN.pulse(x)}
    
    HeartWyrm - a submodule class that houses the core instance of PyEW.EWModule
                that allows operation of a python package connected to the
                Earthworm Message Transport System of a running Earthworm instance.
                It is a child-class of TubeWyrm that wraps a RingToRing style, 
                or similar, modular workflow.

    BufferWyrm - a submodule class for coordinating the merging of a deque of
                MLTrace objects into a DictStream of MLTraceBuffer objects
    
    CloneWyrm - a submodule class for producing cloned deques of objects in
                a pulsed manner.
"""
import time, threading, logging, copy
#import PyEW
import numpy as np
import pandas as pd
from collections import deque
from wyrm.core._base import Wyrm
from wyrm.util.compatability import bounded_intlike, bounded_floatlike
from wyrm.data.mltrace import MLTrace
from wyrm.data.mltracebuffer import MLTraceBuffer
from wyrm.data.dictstream import DictStream

class TubeWyrm(Wyrm):
    """
    Wyrm child-class facilitating chained execution of pulse(x) class methods
    for a sequence wyrm objects, with each wyrm.pulse(x) taking the prior
    member's pulse(x) output as its input.

    The pulse method operates as follows
    for wyrm_dict = {key0: <wyrm0>, key2: <wyrm2> , key1: <wyrm1>}
        tubewyrm.pulse(x) = wyrm1.pulse(wyrm2.pulse(wyrm0.pulse(x)))

        Note that the dictionary ordering dictates the execution order!
    """

    def __init__(self, wyrm_dict={}, wait_sec=0.0, max_pulse_size=1, debug=False):
        """
        Create a tubewyrm object
        :: INPUT ::
        :param wyrm_dict: [dict], [list], or [deque] of [Wyrm-type] objects
                        that are executed in their provided order. dict-type
                        entries allow for naming of component wyrms for 
                        user-friendly assessment. list, deque, and Wyrm-type
                        inputs use a 0-indexed naming system.
        :param wait_sec: [float] seconds to wait between execution of each
                        Wyrm in wyrm_dict
        :param max_pulse_size: [int] number of times to run the sequence of pulses
                        Generally, this should be 1.
        :param debug: [bool] run in debug mode?

        :: OUTPUT ::
        Initialized TubeWyrm object
        """
        # Inherit from Wyrm
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)

        # wyrm_dict compat. checks
        if isinstance(wyrm_dict, Wyrm):
            self.wyrm_dict = {'_tmp': wyrm_dict}
        elif isinstance(wyrm_dict, dict):
            if all(isinstance(_w, Wyrm) for _w in wyrm_dict.values()):
                self.wyrm_dict = wyrm_dict
            else:
                raise TypeError('All elements in wyrm_dict must be type Wyrm')
        # Handle case where a list or deque are provided
        elif isinstance(wyrm_dict, (list, deque)):
            if all(isinstance(_w, Wyrm) for _w in wyrm_dict):
                # Default keys are the sequence index of a given wyrm
                self.wyrm_dict = {_k: _v for _k, _v in enumerate(wyrm_dict)}
            else:
                raise TypeError('All elements in wyrm_dict must be type Wyrm')
        else:
            raise TypeError('wyrm_dict must be a single Wyrm-type object or a list or dictionary thereof')
        
        # wait_sec compat. checks
        self.wait_sec = bounded_floatlike(
            wait_sec,
            name='wait_sec',
            minimum=0,
            maximum=None,
            inclusive=True
        )
        # Create a list representation of keys
        self.names = list(wyrm_dict.keys())

        # Enforce debug setting on all subsequent wyrms
        for _w in self.wyrm_dict.values():
            _w.debug = self.debug

    def update(self, new_dict):
        """
        Apply update to wyrm_dict using the 
        dict.update(new_dict) builtin_function_or_method
        and then update relevant attributes of this TubeWyrm

        This will update existing keyed entries and append new
        at the end of self.wyrm_dict (same behavior as dict.update)

        :: INPUT ::
        :param new_dict: [dict] of [Wyrm] objects

        :: OUTPUT ::
        :return self: [TubeWyrm] enables cascading
        """
        if not isinstance(new_dict, dict):
            raise TypeError('new_dict must be type dict')
        elif not all(isinstance(_w, Wyrm) for _w in new_dict.values()):
            raise TypeError('new_dict can only have values of type Wyrm')
        else:
            pass
        # Run updates on wyrm_dict, enforce debug, and update names list
        self.wyrm_dict.update(new_dict)
        for _w in self.wyrm_dict.values():
            _w.debug = self.debug
        self.names = list(self.wyrm_dict.keys())
        return self
    
    def remove(self, key):
        """
        Convenience wrapper of the dict.pop() method
        to remove an element from self.wyrm_dict and
        associated attributes

        :: INPUT ::
        :param key: [object] valid key in self.wyrm_dict.keys()

        :: RETURN ::
        :return popped_item: [tuple] popped (key, value)
        """
        if key not in self.wyrm_dict.keys():
            raise KeyError(f'key {key} is not in self.wyrm_dict.keys()')
        
        val = self.wyrm_dict.pop(key)
        self.names = list(self.wyrm_dict.keys())
        return (key, val)

    def reorder(self, reorder_list):
        """
        Reorder the current contents of wyrm_dict using either
        an ordered list of wyrm_dict

        :: INPUT ::
        :reorder_list: [list] unique list of keys from self.wyrm
        """
        # Ensure reorder_list is a list
        if not isinstance(reorder_list, list):
            raise TypeError('reorder_list must be type list')

        # Ensure reorder_list is a unique set
        tmp_in = []
        for _e in reorder_list:
            if _e not in tmp_in:
                tmp_in.append(_e)
        if tmp_in != reorder_list:
            raise ValueError('reorder_list has repeat entries - all entries must be unique')

        # Conduct reordering if checks are passed
        # Handle input (re)ordered wyrm_dict key list
        if all(_e in self.wyrm_dict.keys() for _e in reorder_list):
            tmp = {_e: self.wyrm_dict[_e] for _e in reorder_list}
        # Handle input (re)ordered index list
        elif all(_e in np.arange(0, len(reorder_list)) for _e in reorder_list):
            tmp_keys = list(self.wyrm_dict.keys())
            tmp = {_k: self.wyrm_dict[_k] for _k in tmp_keys}

        # Run updates
        self.wyrm_dict = tmp
        self.names = list(tmp.keys())
        return self


    def __repr__(self, extended=False):
        rstr = super().__str__()
        rstr = "(wait: {self.wait_sec} sec)\n"
        for _i, (_k, _v) in enumerate(self.wyrm_dict.items()):
            # Provide index number
            rstr += f'({_i:<2}) '
            # Provide labeling of order
            if _i == 0:
                rstr += "(head) "
            elif _i == len(self.wyrm_dict) - 1:
                rstr += "(tail) "
            else:
                rstr += "  ||   "
            rstr += f"{_k} | "
            if extended:
                rstr += f'{_v.__str__()}\n'
            else:
                rstr += f'{type(_v)}\n'
        return rstr


    def pulse(self, x):
        """
        Initiate a chained pulse for elements of wyrm_queue.

        E.g.,
        tubewyrm.wyrm_dict = {name0:<wyrm0>,
                              name1:<wyrm1>,
                              name2:<wyrm3>}
        y = tubewyrm.pulse(x)
            is equivalent to
        y = wyrmw.pulse(wyrmq.pulse(wyrm0.pulse(x)))

        Between each successive wyrm in the wyrm_dict there
        is a pause of self.wait_sec seconds.

        :: INPUT ::
        :param x: Input `x` for the first Wyrm object in wyrm_dict

        :: OUTPUT ::
        :param y: Output `y` from the last Wyrm object in wyrm_dict
        """
        if self.debug:
            start = time.time()
        for _i in range(self.max_pulse_size):
            if self.debug:
                print(f'TubeWyrm pulse {_i} - {time.time() - start}')
            for _j, _wyrm in enumerate(self.wyrm_dict.values()):
                if self.debug:
                    print(f'...{_wyrm} pulse firing - {time.time() - start}')
                # For first stage of pulse, pass output to `y`
                if _j == 0:
                    y = _wyrm.pulse(x)
                # For all subsequent pulses, update `y`
                else:
                    y = _wyrm.pulse(y)
                # if not last step, wait specified wait_sec
                if _j + 1 < len(self.wyrm_dict):
                    time.sleep(self.wait_sec)
        return y


class CanWyrm(TubeWyrm):
    """
    Child class of TubeWyrm.
    It's pulse(x) method runs the dict of *wyrm_n.pulse(x)'s
    sourcing inputs from a common input `x` and creating a dict
    of each wyrm_n.pulse(x)'s output `y_n`.

    NOTE: This class runs items in serial, but with some modification
    this would be a good candidate class for orchestraing multiprocessing.
    """

    def __init__(self,
                 wyrm_dict={},
                 wait_sec=0.,
                 max_pulse_size=1,
                 debug=False):
        """
        
        """
        # Handle some basic indexing/formatting
        if not isinstance(wyrm_dict, dict):
            if isinstance(wyrm_dict, (list, tuple)):
                wyrm_dict = dict(zip(range(len(wyrm_dict)), wyrm_dict))
            elif isinstance(wyrm_dict, Wyrm):
                wyrm_dict = {0: wyrm_dict}
            else:
                raise TypeError('wyrm_dict must be type dict, list, tuple, or Wyrm')
        
        # Initialize from Wyrm inheritance
        super().__init__(wyrm_dict=wyrm_dict, wait_sec=wait_sec, debug=debug, max_pulse_size=max_pulse_size)

        self.dict = {_k: None for _k in self.names}


    def pulse(self, x, **options):
        """
        Triggers the wyrm.pulse(x) method for each wyrm in wyrm_dict, sharing
        the same inputs, and writing outputs to self.dict[wyrmname] via the __iadd__
        method. I.e.,

            self.dict[wyrmname] += wyrm.pulse(x, **options)

        :: INPUTS ::
        :param x: [object] common input for all wyrms in wyrm_dict
        :param options: [kwargs] key word arguments passed to each wyrm's
                    wyrm.pulse(x, **options) method

        :: OUTPUT ::
        :return y: [dict] access to self.dict
        """
        for _i in range(self.max_pulse_size):
            for _k, _v in self.wyrm_dict.items():
                _y = _v.pulse(x, **options)
                # If this wyrm output has not been mapped to self.dict
                if self.dict[_k] is None:
                    self.dict.update({_k: _y})
            if self.debug:
                print(f'CanWyrm pulse {_i + 1}')
                for _l, _w in self.dict.items():
                    print(f'    {_l} - {len(_w)}')
        y = self.dict
        return y



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
        wait_sec,
        DR_ID,
        MOD_ID,
        INST_ID,
        HB_PERIOD,
        wyrm_list={},
        debug=False
    ):
        """
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
        :param wyrm_list: [list-like] iterable set of *wyrm objects
                            with sequentially compatable *wyrm.pulse(x)

        :: PUBLIC ATTRIBUTES ::
        :attrib module: False or [PyEW.EWModule] - Holds Module Object
        :attrib wait_sec: [float] rate in seconds to wait between pulses
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
        super().__init__(
            wyrm_list=wyrm_list,
            wait_sec=wait_sec,
            max_pulse_size=None,
            debug=debug)
        
        # Public Attributes
        self.module = False
        self.conn_info = pd.DataFrame(columns=["Name", "RING_ID"])

        # Private Attributes
        self._default_ring_id = self._bounded_intlike_check(DR_ID, name='DR_ID', minimum=0, maximum=9999)
        self._module_id = self._bounded_intlike_check(MOD_ID, name='MOD_ID', minimum=0, maximum=255)
        self._installation_id = self._bounded_intlike_check(INST_ID, name='INST_ID', minimum=0, maximum=255)
        self._HBP = self._bounded_floatlike_check(HB_PERIOD, name='HB_PERIOD', minimum=1)

        # Module run attributes
        # Threading - TODO - need to understand this better
        self._thread = threading.Thread(target=self.run)
        self.runs = True

    def __repr__(self):
        # Start with TubeWyrm __repr__
        rstr = f"{super().__repr__()}\n"
        # List Pulse Rate
        rstr += f"Pulse Wait Time: {self.wait_sec:.4f} sec\n"
        # List Module Status and Parameters
        if isinstance(self.module, PyEW.EWModule):
            rstr += "Module: Initialized\n"
        else:
            rstr += "Module: NOT Initialized\n"
        rstr += f"MOD: {self._module_id}"
        rstr += f"DR: {self._default_ring_id}\n"
        rstr += f"INST: {self._installation_id}\n"
        rstr += f"HB: {self._HBP} sec\n"
        # List Connections
        rstr += "---- Connections ----\n"
        rstr += f"{self.conn_info}\n"
        rstr += "-------- END --------\n"
        return rstr

    def initialize_module(self, user_check=True):
        if user_check:
            cstr = "About to initialize the following PyEW.EWModule\n"
            cstr += f"Default ring ID: {self._default_ring_id:d}\n"
            cstr += f"Module ID: {self._module_id:d}\n"
            cstr += f"Inst. ID: {self._installation_id:d}\n"
            cstr += f"Heartbeat: {self._HBP:.1f} sec\n"
            cstr += f"Debug?: {self.debug}\n"
            cstr += "\n Do you want to continue? [(y)/n]"
            ans = input(cstr)
            if ans.lower().startswith("y") or ans == "":
                user_continue = True
            elif ans.lower().startswith("n"):
                user_continue = False
            else:
                print("Invalid input -> exiting")
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
                        debug=self.debug,
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
            print("Canceling module initialization -> exiting")
            exit()

    def add_connection(self, RING_ID, RING_Name):
        """
        Add a connection between target ring and the initialized self.module
        and update the conn_info DataFrame.

        Method includes safety catches
        """
        # === RUN COMPATABILITY CHECKS ON INPUT VARIABLES === #
        # Enforce integer RING_ID type
        RING_ID = self._bounded_intlike_check(RING_ID,name='RING_ID', minimum=0, maximum=9999)
        
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
            time.sleep(self.wait_sec)
            # Run Pulse Command inherited from TubeWyrm
            self.pulse()  # conn_in = conn_in, conn_out = conn_out)
        # Polite shut-down of module
        self.module.goodbye()
        print("Exiting HeartWyrm Instance")


class BufferWyrm(Wyrm):
    """
    Class for buffering/stacking MLTrace objects into a DictStream of MLTraceBuffer objects with
    self-contained settings and sanity checks for changing the MLTraceBuffer.__add__ method options.
    """
    def __init__(self, max_length=300., add_method=3, restrict_past_append=True, blinding=(0,0), max_pulse_size=10000, debug=False, **add_kwargs):
        """
        Initialize a BufferWyrm object

        :: INPUTS ::
        :param max_length: [float] maximum MLTraceBuffer length in seconds
        :param add_style: [str] style of handling overlapping samples in a given MLTraceBuffer
                            Supported:
                                'merge' - follows the obspy.core.trace.Trace.__add__(method=1, interpolation_samples=-1)
                                            style of behaviors, linearly interpolating between end-point samples in
                                            a given overlap.
                                        NOTE: Generally used for stitching together waveform data
                                'stack' - stacks overlapping samples either using the 'max' or 'avg' approach typically
                                            used for overlapping predicted values from machine learning models
        :param restrict_past_add: [bool] - enforce restrictions on appending data that temporally preceed data contained
                            within MLTraceBuffer objects?  also see wyrm.data.mltrace.MLTraceBuffer.__add__
        :param max_pulse_size: [int] maximum number of items in `x` to assess in a single call of BufferWyrm.pulse(x)
        :param debug: [bool] - run in debug mode?
        :param **add_kwargs: [kwargs] key-word argument collector for passing to MLTraceBuffer.__add__() to alter
                        default parameter value(s) for all calls of MLTraceBuffer.__add__ conducted by 
                        __init__ or pulse()
        """
        self.max_length = bounded_floatlike(
            max_length,
            name='max_length',
            minimum = 0,
            maximum=None,
            inclusive=False
            )

        super().__init__(max_pulse_size=max_pulse_size, debug=debug)

        if add_method in [0,'dis','discard',
                          1,'int','interpolate',
                          2,'max','maximum',
                          3,'avg','average']:
            self.add_method = add_method
        else:
            raise ValueError(f'add_method {add_method} not supported. See wyrm.data.mltrace.MLTrace.__add__()')
        
        if not isinstance(restrict_past_append, bool):
            raise TypeError('restrict_past_append must be type bool')
        else:
            self.restrict_past_append = restrict_past_append

        if not isinstance(blinding, (type(None), bool, tuple)):
            raise TypeError
        else:
            self.blinding = blinding

   
        
            

        # _stack_kwargs = {'blinding_samples': int,'method': str,'fill_value': (type(None), int, float),'sanity_checks': bool}
        # _merge_kwargs = {'interpolation_samples': int,'method': str,'fill_value': (type(None), int, float),'sanity_checks': bool}
        # _kwarg_checks = {'stack': _stack_kwargs, 'merge': _merge_kwargs}

        # for _k, _v in add_kwargs.items():
        #     if _k not in _kwarg_checks[self.add_style].keys():
        #         raise KeyError(f'kwarg {_k} is not supported for add_style "{self.add_style}"')
        #     elif not isinstance(_v, _kwarg_checks[self.add_style][_k]):
        #         raise TypeError(f'kwarg {_k} type "{type(_v)}" not supported for add_style "{self.add_style}')
        self.add_kwargs = add_kwargs
            
        if not isinstance(restrict_add_past, bool):
            raise TypeError
        else:
            self.restrict_add_past = restrict_add_past
        # Initialize Buffer Containing Object
        self.buffer = DictStream()

    def pulse(self, x):
        """
        Conduct a pulse on input deque of MLTrace objects (x)
        """
        if not isinstance(x, deque):
            raise TypeError
        
        qlen = len(x)
        for _i in range(self.max_pulse_size):
            if qlen == 0:
                break
            elif _i + 1 > qlen:
                break
            else:
                _x = x.popleft()
            
            if not isinstance(_x, MLTrace):
                x.append(_x)
            else:
                _id = _x.id
                if _id not in self.buffer.labels():
                    new_buffer_item = MLTraceBuffer(max_length=self.max_length,
                                               add_style=self.add_style,
                                               restrict_add_past=self.restrict_add_past)
                    new_buffer_item.__add__(_x, **self.add_kwargs)
                    self.buffer.append(new_buffer_item, **self.add_kwargs)
                else:
                    self.buffer.append(_x,
                                       restrict_add_past=self.restrict_add_past,
                                        **self.add_kwargs)
                    
    def __repr__(self, extended=False):
        rstr = f'Add Style: {self.add_style}'
        return rstr
    
class CloneWyrm(Wyrm):
    """
    Submodule class that provides a pulsed method for producing (multiple) copies
    of an arbitrary set of items in an input deque into a dictionary of output deques
    """
    def __init__(self, queue_names=['A','B'], max_pulse_size=1000000, debug=False):
        """
        Initialize a CloneWyrm object
        :: INPUTS ::
        :param queue_names: [list-like] of values to assign as keys (names) to 
                            output deques held in self.queues
        :param max_pulse_size: [int] maximum number of elements to pull from 
                            an input deque in a single pulse
        :param debug: [bool] - should this be run in debug mode?
        """
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)
        self.nqueues = len(queue_names)
        self.queues = {}
        for _k in queue_names:
            self.queues.update({_k: deque()})
    
    def pulse(self, x):
        """
        Run a pulse on deque x where items are popleft'd out of `x`,
        deepcopy'd for each deque in self.queues and appended to 
        each deque in self.deques. The original object popped off
        of `x` is then deleted from memory as a clean-up step.

        Stopping occurs if `x` is empty (len = 0) or self.max_pulse_size
        items are cloned from `x` 

        :: INPUT ::
        :param x: [collections.deque] double ended queue containing N objects

        :: OUTPUT ::
        :return y: [dict] of [collections.deque] clones of contents popped from `x`
                with dictionary keys corresponding to queue_names elements.
        """
        if not isinstance(x, deque):
            raise TypeError('x must be type deque')
        for _ in range(self.max_pulse_size):
            if len(x) == 0:
                break
            _x = x.popleft()
            for _q in self.queues.values():
                _q.append(copy.deepcopy(_x))
            del _x
        y = self.queues
        return y