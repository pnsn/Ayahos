"""
:module: PULSE.mod.base
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This contains the definition for the :class:`~PULSE.mod.base.BaseMod`
    class that serves as a template for all other :mod:`~PULSE.mod`
    classes in PULSE. This class defines the :meth:`~PULSE.mod.base.BaseMod.pulse` 
    method and its sub-methods that inheriting :mod:`~PULSE.mod` classes modify
    to change the functionality of the :meth:`~.BaseMod.pulse` method

    Sub-methods have a POLYMORPHIC label attached to their docstrings to indicate
    they are targets for mutable behaviors and to track the last module in which
    they were modified.

"""

from copy import deepcopy
from collections import deque
import logging, sys, os
from obspy.core.utcdatetime import UTCDateTime
from PULSE.data.header import ModStats

# Logger at module level
Logger = logging.getLogger(__name__)



class BaseMod(object):
    """Base class template all other :mod:`~PULSE.mod` classes that defines the polymorphic :meth:`~PULSE.mod.base.BaseMod.pulse`

    :param max_pulse_size: maximum number of iterations to run for each call of :meth:`~PULSE.mod.base.BaseMod.pulse`, defaults to 1
    :type max_pulse_size: int, optional
    :param maxlen: maximum number of items to keep in this object's *output* attribute, defaults to None.
        This is passed as the **collections.deque.maxlen** attribute of **BaseMod.output**
    :type maxlen: int, optional

    :var output: a :class:`~collections.deque` with **maxlen** set by :param:`~maxlen` that holds outputs from :meth:`~PULSE.mod.base.BaseMod.pulse` calls
    :var stats: a :class:`~PULSE.mod.base.PulseStats` object that holds metadata from the last call of :meth:`~PULSE.mod.base.BaseMod.pulse`
    :var max_pulse_size: maximum number of iterations for single call of :meth:`~PULSE.mod.base.BaseMod.pulse` 
    :var maxlen: maximum size of the **BaseMod.output** attribute. This is enforced with the :class:`~collections.deque` **maxlen** behavior. 
    :var Logger: a :class:`~logging.Logger` object named to match this class (BaseMod)

    """

    def __init__(self, max_pulse_size=1, maxlen=None, name=None):
        """Initialize a :class:`~PULSE.module.base.BaseMod object`

        :param max_pulse_size: maximum number of iterations to run for each call of :meth:`~PULSE.module.base.BaseMod.pulse`, defaults to 10
        :type max_pulse_size: int, optional
        :param maxlen: maximum number of items to keep in this BaseMod.output attribute, defaults to 100.
            This is passed to the **collections.deque.maxlen** attribute of **BaseMod.output**
        :type maxlen: int, optional
        """
        # Initialize Stats
        self.stats = ModStats()

        # Compatability check for `max_pulse_size`
        if isinstance(max_pulse_size, (int, float)):
            if max_pulse_size >= 1:
                self.stats.mps = int(max_pulse_size)
            else:
                raise ValueError('max_pulse_size must be g.e. 1 ')
        else:
            raise TypeError('max_pulse_size must be int-like')
        
        # Compatability check for `name`
        self.setname(name)

        # Set up logging at the module object level
        self.Logger = logging.getLogger(f'{self.name}')
        # Flag input type
        self._input_types = [deque]
        # Initialize output
        self.output = deque(maxlen=maxlen)
        self.stats.maxlen = maxlen
        # Initialize flag for continuing pulse iterations
        self._continue_pulsing = True
    
    ####################
    ## DUNDER METHODS ##
    ####################
    def __repr__(self, full=False):
        """
        Return a string report essential user data for this BaseMod object
        """
        rstr = self.stats.__str__()
        if full:
            rstr += f'\n{self.output.__str__()}'
        return rstr

    def __str__(self):
        """Return self.__class__ string for this BaseMod object"""
        rstr = self.__class__ 
        return rstr
    
    #####################
    ## UTILITY METHODS ##
    #####################
    def setname(self, name=None):
        """Set the name of this module 
        
        Name must include the class name, or **name** will
        be appended to the class name as ClassName_name.
        e.g., name = test --> BaseMod_test

        name=None uses the class name (e.g., BaseMod)


        :param name: Name of this module, defaults to None
        :type name: str or NoneType, optional.
        """        
        if name is None:
            self.stats.name = self.__class__.__name__
        elif isinstance(name, str):
            if self.__class__.__name__ not in name:
                self.stats.name = f'{self.__class__.__name__}_{name}'
            else:
                self.stats.name = name
        else:
            raise TypeError('name must be type None or str')
        
        self.name = self.stats.name

    def copy(self, newname=False):
        """Return a deepcopy of this object with the 
        option to give it a new name using :meth:`~.BaseMod.setname`
        
        :param newname: new name for the coppied Mod,
            default is None
        :type newname: str, bool, or NoneType, optional
        
        :returns:
         - **newmod** (*same as self*) -- new copy of mod object
        """
        newmod = deepcopy(self)
        if not newname:
            return newmod
        else:
            newmod.setname(newname)
            return newmod

    def import_class(self, class_path_str):
        """Use the full extension ID of a class object to import that class within
        the local scope of this class-method and return the class object for use
        elsewhere

        e.g. class_path_str = 'obspy.core.trace.Trace'
          runs exec('from obspy.core.trace import Trace')
          and returns obj = Trace

        :param class_path_str: class extension ID
        :type class_path_str: str
        :return: class defining object
        :rtype: type
        """        
        if not isinstance(class_path_str, str):
            raise TypeError('class_path_str must be type str')
        elif '.' not in class_path_str:
            raise ValueError('class_path_str is not .-delimited. Does not look like a class __name__')
        
        parts = class_path_str.split('.')
        path = '.'.join(parts[:-1])
        clas = parts[-1]
        try:
            exec(f'from {path} import {clas}')
            obj = eval(f'{clas}')
            return obj
        except ImportError:
            self.Logger.critical(f'ImportError: failed to import {class_path_str}. Exiting on CANTCREAT ({os.EX_CANTCREAT})')
            sys.exit(1)

    ###################
    ## PULSE METHODS ##
    ###################
    def check_input(self, input: deque) -> None:
        # Conduct type-check on input
        if not any(isinstance(input, _t) for _t in self._input_types):
            self.Logger.critical(f'TypeError: input ({type(input)}) is not type collections.deque. Exiting')
            sys.exit(os.EX_DATAERR)

    def measure_input(self, input: deque) -> int:
        """Measure the size of an input to the pulse method
        for this :class:`~.BaseMod`-type object

        POLYMORPHIC: last update with :class:`~.BaseMod`

        :param input: input collection of unit inputs
        :type input: collections.deque
        :return:
            - **measure** (*int*) -- length of **input**
        """        
        return len(input)
    
    def measure_output(self) -> int:
        """Measure the size of the output attribute of
        this :class:`~.BaseMod`-type object

        POLYMORPHIC: last update with :class:`~.BaseMod`

        :return:
            - **measure** (*int*) -- length of **output**
        """  
        return len(self.output)

    def pulse_startup(self, input: deque) -> None:
        """Run startup checks and metadata capture
         at the outset of a call of :meth:`~.pulse`
        
        :param input: collection of input objects
        :type input: deque
        """        
        self.stats.starttime = UTCDateTime.now()
        self.stats.in0 = self.measure_input(input)
        self.stats.out0 = self.measure_output()
        self._continue_pulsing = True
    
    def pulse_shutdown(self, input: deque, niter: int, exit_type: str) -> None:
        """Run shutdown checks and metadata capture
        at the conclusion of a call of :meth:`~.BaseMod.pulse`

        :param input: collection of input objects
        :type input: deque
        :param niter: current iteration number (0-indexed)
        :type niter: int
        :param exit_type: reason for conclusion
        :type exit_type: str
            Supported exit_type values & meanings:
                - 'nodata' -- pulse received a non-NoneType input with 0 length
                - 'early-get' -- pulse iterations stopped early at the `get_unit_input` method
                - 'early-run' -- pulse iterations stopped early at the `run_unit_process` method
                - 'early-put' -- pulse iterations stopped early at the `put_unit_output` method
                - 'max' -- pulse concluded at maximum iterations
        """        
        self.stats.endtime = UTCDateTime.now()
        self.stats.in1 = self.measure_input(input)
        self.stats.out1 = self.measure_output()
        if exit_type == 'nodata':
            self.stats.niter = 0
        elif exit_type == 'max':
            self.stats.niter = self.stats.mps
        elif 'early' in exit_type:
            if exit_type == 'early-put':
                self.stats.niter = niter + 1
            else:
                self.stats.niter = niter
        else:
            self.Logger.critical(f'exit_type "{exit_type}" not supported. Exiting')
            sys.exit(os.EX_DATAERR)
        self.stats.stop = exit_type

    def get_unit_input(self, input: deque) -> object:
        """Extract a unit process input object from input

        POLYMORPHIC: last update with :class:`~.BaseMod`

        Here,
            pop an object off input
            Early stopping if input is an empty deque

        :param input: collection of input objects
        :type input: deque
        :return:
         - **unit_input** (*object* or *NoneType*) -- unit input object. Empty input returns None
        """        
        try:
            unit_output = input.pop()
        except IndexError:
            self._continue_pulsing = False
            unit_output = None
        except AttributeError:
            self.Logger.critical(f'AttributeError: input of type {type(input)} does not have method "pop". Exiting')
            sys.exit(os.EX_USAGE)
        return unit_output

    def run_unit_process(self, unit_input: object) -> object:
        """Run the unit process on a unit input object

        POLYMORPHIC: last update with :class:`~.BaseMod`

        Here, 
            unit_output = unit_input
            No early stopping clauses

        :param unit_input: unit input object
        :type unit_input: object
        :return: 
         - **unit_output** (*object*) -- the same object as unit_input
        """        
        unit_output = unit_input
        return unit_output
    
    def put_unit_output(self, unit_output: object) -> None:
        """Store the unit output object in this module's **output**
        attribute.

        POLYMORPHIC: last update with :class:`~.BaseMod`

        Here,
            appendleft unit_output to the **output** attribute
            No early stopping clauses

        :param unit_output: _description_
        :type unit_output: object
        """        
        self.output.appendleft(unit_output)

    #######################
    ## CORE PULSE METHOD ##
    #######################
    def pulse(self, input):
        """Core method for

        :param input: _description_
        :type input: _type_
        :return: _description_
        :rtype: _type_
        """
        # Run initial check on input type/properties
        self.check_input(input)
        # Run startup checks & capture start stats
        self.pulse_startup(input)

        ## DETERMINE NUMBER OF ITERATIONS & ALLOWANCE FOR EARLY STOPPING
        # Zero length input - stop before iterations
        if self.stats.in0 == 0:
            self.pulse_shutdown(input,
                                niter=None,
                                exit_type='nodata')
            return self.output
        else:
            pass
        # Run Iterations
        for _n in range(self.stats.mps):
            # Extract an unit input object (and handle early stopping)
            unit_input = self.get_unit_input(input)
            if not self._continue_pulsing:
                self.pulse_shutdown(input,
                                    niter=_n,
                                    exit_type='early-get')
                return self.output
            # Process the unit input object (and handle early stopping)
            unit_output = self.run_unit_process(unit_input)
            if not self._continue_pulsing:
                self.pulse_shutdown(input,
                                    niter=_n,
                                    exit_type='early-run')
                return self.output
            # Store the unit output object (and handle early stopping)
            self.put_unit_output(unit_output)
            if not self._continue_pulsing:
                self.pulse_shutdown(input,
                                    niter=_n + 1,
                                    exit_type='early-put')
                return self.output
        # If iterations conclude, 
        self.pulse_shutdown(input,
                            niter=self.stats.mps,
                            exit_type='max')
        return self.output


    # def pulse(self, input):
    #     """CORE POLYMORPHIC METHOD
        
    #     Specifics for :class:`~PULSE.mod.base.BaseMod`

    #     :param input: collection of objects that will have elements removed using :meth:`~collections.deque.popleft` 
    #         and appended to **BaseMod.output** using :meth:`~collections.deque.append`
    #     :type input: collections.deque
    #     :return:
    #         - **output** (*collections.deque*) -- view of this BaseMod object's **BaseMod.deque** attribute
        
    #     .. rubric:: Pulse Template Explanation

    #     This definition of :meth:`~PULSE.mod.base.BaseMod.pulse` is used by all
    #     :mod:`~PULSE.mod` classes an that executes a number of unit-tasks
    #     depending on the behaviors of sub-routine methods specific to each of these clases.
    #     This method also updates values in the **stats** attribute with metadata from the most
    #     recent call of this method.

    #     This method is inherited by all :mod:`~PULSE.mod` classes and has the following structure:
        
    #     A) Update **stats** 'starttime', 'in0', and 'out0' values

    #     B) Iteration Loop (up to **max_pulse_size** iterations)

    #         1) :meth:`~PULSE.mod.base.BaseMod.get_unit_input` - extract an object from **input** that will be used for the next step
    #         2) :meth:`~PULSE.mod.base.BaseMod.run_unit_process` - run the unit process for this class.
    #         3) :meth:`~PULSE.mod.base.BaseMod.store_unit_output` - merge the output from the previous step into the *output* attribute of this object
        
    #     C) Update **stats** 'stop', 'niter', 'endtime', 'in1', and 'out1' values

    #     """ 
    #     # Capture pulse stats at beginning of call and measure/check input
    #     self.stats.starttime = UTCDateTime.now()
    #     self.stats.in0 = self.measure_input(input)
    #     self.stats.out0 = self.measure_output()
        

    #     if self.stats.in0 == 0:
    #         self.stats.niter = 0
    #         self.stats.stop = 'empty_input'
        
    #     # Convert flag to proceed with pulse
    #     self._continue_pulsing = True
    #     # Set initial value of niter to 0
    #     self.stats.niter = 0
    #     _early = False

    #     for _n in range(self.max_pulse_size):
    #         # get single object for unit_process
    #         unit_input = self.get_unit_input(input)
    #         # If _continue_pulsing flag is flipped to False by get_unit_input
    #         if not self._continue_pulsing:
    #             # Log stop type
    #             self.stats.stop = 'head'
    #             _early = True
    #             # Log current iteration index
    #             self.stats.niter = _n
    #             # Break for-loop
    #             break
    #         # If _continue_pulsing flag remains True, proceed with unit process
    #         else:
    #             unit_output = self.run_unit_process(unit_input)
    #             # Capture output
    #             self.store_unit_output(unit_output)
            
    #         # If _conclude_pulsing flag is flipped to False by store_unit_output
    #         if self._continue_pulsing:
    #             pass
    #         else:
    #             # Break iteration late & update 'stop' reason
    #             self.stats.stop = 'tail'
    #             self.stats.niter = _n + 1
    #             _early = True
    #             break
    #     # Max iteration capture
    #     if not _early:
    #         self.stats.stop = 'max'
    #         self.stats.niter = self.max_pulse_size
        
    #     # Capture pulse stats at conclusion of call
    #     self.stats.in1 = self.measure_input(input)
    #     self.stats.out1 = self.measure_output()
    #     self.stats.endtime = UTCDateTime.now()

    #     return self.output




    # UPDATED PULSE





    # def measure_input(self, input):
    #     """
    #     POLYMORPHIC METHOD

    #     Last updated with :class:`~PULSE.module.base.BaseMod`

    #     Checks if **input** is iterable, and if so, this method returns the length of **input**

    #     :param input: input deque of objects to process
    #     :type input: collections.deque
    #     :return:
    #      - **input_size** (*int* or *NoneType*) - length of the input or None
        
    #     If **input** is not iterable or NoneType, logs CRITICAL and exits program on code *os.EX_DATAERR*.
    #     """
    #     if isinstance(input, deque):
    #         input_size = len(input)
    #         return input_size
    #     else:
    #         self.Logger.critical(f'input is not type deque. Quitting on code DATAERR ({os.EX_DATAERR})')
    #         sys.exit(os.EX_DATAERR)
    
    # def measure_output(self):
    #     """POLYMORPHIC METHOD

    #     Last updated with :class`~PULSE.module.base.BaseMod`

    #     Measure the length of the **BaseMod.output** attribute

    #     :return:
    #      - **output_size** (*int*) -- length of **self.output**
    #     """        
    #     return len(self.output)
    
    # def get_unit_input(self, input):
    #     """POLYMORPHIC METHOD

    #     Last updated with :class:`~PULSE.module.base.BaseMod`

    #     Get the input object for :meth:`~PULSE.module.base.BaseMod.run_unit_process` from
    #     the `input` provided to :meth:`~PULSE.module.base.BaseMod.pulse`

    #     :param input: deque of objects
    #     :type input: collections.deque
    #     :return:
    #      - **unit_input** (*object* or *bool*) -- object removed from **input** using :meth:`~collections.deque.popleft`.
    #             if input is empty, this method returns `False`, triggering early iteration stopping in :meth:`~PULSE.mod.base.BaseMod.pulse`
    #     """ 
    #     # Check input type and exit if it is incorrect type
    #     if not isinstance(input, deque):
    #         self.Logger.critical(f'input is not type deque. Quitting on DATAERR ({os.EX_DATAERR})')
    #         sys.exit(os.EX_DATAERR)
    #     # Proceed if input is correct type
    #     else:
    #         pass
    #     # If there are elements to pull from input, pull one    
    #     if self.measure_input(input) > 0:
    #         unit_input = input.popleft()
    #     # If the deque is empty, flag early stopping and return None
    #     else:
    #         unit_input = None
    #         self._continue_pulsing = False
    #     return unit_input
        
    # def run_unit_process(self, unit_input):
    #     """POLYMORPHIC METHOD

    #     Last updated with :class: `~PULSE.module.base.BaseMod`

    #     Returns the input object as output

    #     :param unit_input: any unit input object, except False, which is reserved for early stopping
    #     :type unit_input: object
    #     :return:
    #      - **unit_output** (*object*)
    #     """
    #     unit_output = unit_input
    #     return unit_output        
    
    # def store_unit_output(self, unit_output):
    #     """POLYMORPHIC METHOD

    #     Last updated with :class: `~PULSE.module.base.BaseMod`

    #     Attach unit_output to self.output using :meth:`~collections.deque.append`

    #     :param unit_output: unit output from :meth:`~PULSE.mod.base.BaseMod.run_unit_process`
    #     :type unit_output: object
    #     """        
    #     self.output.append(unit_output)
    #     # NOTE: This is another place where self._continue_pulsing can be updated for early stopping type 1
        
 


    #TODO: Make averaging __add__ for mltrace?

    # def capture_pulse_iteration_metadata(self, pulse_starttime, input_size, niter, early_stop_code):
    #     """
    #     POLYMORPHIC METHOD
    #     Last updated with :class:`~PULSE.module.base.BaseMod`

    #     Captures metadata from the most recent pulse and updates
    #     the representative reporting string attribute *self.report*

    #     :param pulse_starttime: time the pulse was triggered
    #     :type pulse_starttime: pandas.core.timestamp.Timestamp
    #     :param input_size: size of the input at the start of a call of pulse()
    #     :type input_size: int
    #     :param niter: number of unit processes completed in this pulse
    #     :type niter: int
    #     :param early_stop_code: integer standin for True (1) and False (0) if the pulse hit an early termination clause
    #     :type early_stop_code: int
    #     """        
    #     # Calculate/capture new metadata values for this pulse
    #     timestamp = pd.Timestamp.now().timestamp()
    #     runtime = (timestamp - pulse_starttime)
    #     output_size = len(self.output)
    #     df_line = pd.DataFrame(dict(zip(self._keys_meta,
    #                                    [timestamp, niter, runtime, input_size, output_size, early_stop_code])),
    #                            index=[0])
    #     # Append line to dataframe
    #     if len(self._metadata) == 0:
    #         self._metadata = df_line
    #     else:
            
    #         self._metadata = pd.concat([self._metadata, df_line], axis=0, ignore_index=True)

    #     # Trim outdated metadata
    #     oldest_time = timestamp - self.meta_memory
    #     self._metadata = self._metadata[self._metadata.time >= oldest_time]

    # def _update_report(self):
    #     """
    #     Update the **BaseMod.report** attribute with a synopsis of saved metadata
    #     """        
    #     # Update Report       
    #     df_last = self._metadata[self._metadata.time == self._metadata.time.max()]
    #     df_last.index = ['last']
    #     self.report = df_last 
    #     self.report = pd.concat([self.report, self._metadata.agg(['mean','std','min','max'])], axis=0, ignore_index=False)
    #     self.report.time = self.report.time.apply(lambda x: pd.Timestamp(x, unit='s'))

    # def _update_pulse_rate(self):
    #     """Update the estimate of average pulse rate from logged metadata and update the **BaseMod._pulse_rate** attribute
    #     """
    #     # Calculate pulse rate
    #     nd = len(self._metadata)
    #     if nd > 1:
    #         dt = (self._metadata.time.max() - self._metadata.time.min())
    #         if dt > 0:
    #             self._pulse_rate = nd/dt

    # def _generate_report_string(self):
    #     """Generate a string representation of metadata logging for output to the command line
    #     comprising a formatted header and the current **BaseMod.report** attribute elements

    #     :return: report string
    #     :rtype: str
    #     """        
    #     header = f'~~|^v~~~ {pd.Timestamp.now()} | {self.__name__()} ~~|^v~~~\n'
    #     header += f'pulse rate: {self._pulse_rate:.2e} Hz | max pulse size: {self.max_pulse_size}\n'
    #     header += f'sample period: {self.meta_memory} sec | max output size: {self.maxlen}'
    #     return f'{header}\n{self.report}\n'
    



    # def raise_log(self, etype, emsg='', level='critical', exit_code=1):
    #     """Convenience wrapper for writing *Error messages to logging.
    #     If logging `level` == 'critical' program exits using :meth:`~sys.exit`
    #     with the specified exit code

    #     :param etype: Error Type to write to message
    #     :type etype: 
    #     :param emsg: _description_, defaults to ''
    #     :type emsg: str, optional
    #     :param level: _description_, defaults to 'critical'
    #     :type level: str, optional
    #     :param exit_code: _description_, defaults to 1
    #     :type exit_code: int, optional
    #     :raises ValueError: _description_
    #     """
    #     if isinstance(etype, str):
    #         pass
    #     else:
    #         raise TypeError('etype must be type str')
    #     if level.lower() == 'debug':
    #         self.Logger.debug(f'{etype}: {emsg}')
    #     elif level.lower() == 'error':
    #         self.Logger.error(f'{etype}: {emsg}')
    #     elif level.lower() == 'warning':
    #         self.Logger.warning(f'{etype}: {emsg}')
    #     elif level.lower() == 'critical':
    #         self.Logger.critical(f'{etype}: {emsg}')
    #         sys.exit(exit_code)
    #     else:
    #         raise ValueError(f'level "{level}" not supported.')


        #     #  Check if early stopping should occur at the end of this iteration
        #     if self.should_next_iteration_run(unit_output):
        #         pass
        #     else:
        #         self._update_metadata(pulse_starttime, input_size, niter, 1)
        #         break
        # # Get alias of self.output as output
        # output = self.output
        # self.cli_reporter.update(
        # # TODO: make the reporting a separate 

        # # If the 
        # if _n + 1 == self.max_pulse_size:
        #     self._update_metadata(pulse_starttime, input_size, niter, 0)

        # self._update_report()
        # # self._update_pulse_rate()

        # # Check if reporting is turned on
        # if self.report_period:
        #     # If so, get the now-time
        #     nowtime = pd.Timestamp.now().timestamp()
        #     # If the elapsed time since the last report transmission is exceeded
        #     if self.report_period <= nowtime - self._last_report_time:
        #         # Transmit report to logging
        #         self.Logger.info(f'\n\n{self._generate_report_string()}\n')
        #         # Update the last report time
        #         self._last_report_time = nowtime
    
    # # Compatability checks for `meta_memory`
        # if isinstance(meta_memory, (int, float)):
        #     if meta_memory > 0:
        #         if meta_memory <= 24*3600:
        #             self.meta_memory = meta_memory
        #         else:
        #             raise NotImplementedError('pulse metadata logging currently does not support memory periods of more than 24 hours')
        #     else:
        #         raise ValueError
        # else:
        #     raise TypeError
        
        # # Compatability checks for `report_period`
        # if report_period is False:
        #     self.report_period = False
        # elif isinstance(report_period, (int, float)):
        #     if 0 < report_period <= self.meta_memory:
        #         self.report_period = report_period
        #     else:
        #         self.Logger.critical(f'{self.__class__} | report_period greater than memory period!')
        #         sys.exit(1)
        # else:
        #     raise TypeError
    
       
        # # Metadata keys for reporting
        # self._keys_meta = ['time', '# proc', 'runtime', 'isize', 'osize', 'early_stop']
        # # Metadata holder for reporting
        # self._metadata = pd.DataFrame(columns=self._keys_meta)
        # # Initialize report attribute
        # self.report = pd.DataFrame()
        # # Initialize last report timestamp
        # self._last_report_time = pd.Timestamp.now().timestamp()
        # # Initalize placeholder for _pulse_rate
        # self._pulse_rate = nan