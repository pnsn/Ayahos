"""
:module: PULSE.module.base
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This contains the definition for the :class:`~PULSE.module.base.BaseMod` class that serves as a template for all
    other :mod:`~PULSE.module` classes in PULSE. It defines the :meth:`~PULSE.module.base.BaseMod.pulse` polymorphic method,
    its sub-methods, and a set of controls associated with 

"""

from numpy import nan
from copy import deepcopy
from collections import deque
import logging, sys, os
from obspy.core.utcdatetime import UTCDateTime
from PULSE.data.header import PulseStats

# Logger at module level
Logger = logging.getLogger(__name__)


# @add_class_name_to_docstring
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

    def __init__(self, max_pulse_size=1, maxlen=None, name_suffix=None):
        """Initialize a :class:`~PULSE.module.base.BaseMod object`

        :param max_pulse_size: maximum number of iterations to run for each call of :meth:`~PULSE.module.base.BaseMod.pulse`, defaults to 10
        :type max_pulse_size: int, optional
        :param maxlen: maximum number of items to keep in this BaseMod.output attribute, defaults to 100.
            This is passed to the **collections.deque.maxlen** attribute of **BaseMod.output**
        :type maxlen: int, optional
        """

        # Compatability check for `max_pulse_size`
        if isinstance(max_pulse_size, (int, float)):
            if 1 <= max_pulse_size:
                self.max_pulse_size = int(max_pulse_size)
            else:
                raise ValueError('max_pulse_size must be g.e. 1 ')
        else:
            raise TypeError('max_pulse_size must be positive int-like')
        
        if name_suffix is None:
            self._suffix = ''
        elif isinstance(name_suffix, (str,int)):
            self._suffix = f'_{name_suffix}'

        # Set up logging at the module object level
        self.Logger = logging.getLogger(f'{self.__name__(full=False)}')
        # Initialize output
        self.output = deque(maxlen=maxlen)
        self.maxlen = maxlen
        # Initialize pulse metadata holder
        self.stats = PulseStats()
        self.stats.modname = self.__name__(full=False)
        # Initialize flag for continuing pulse iterations
        self._continue_pulsing = True

    def __name__(self, full=False):
        """Return the camel-case name of this class with/without the submodule extension

        :param full: use the full class path? Defaults to False
        :type full: bool, optional
        :return: class name
        :rtype: str
        """
        if full:
            return self.__class__.__str__(self)
        else:
            return self.__class__.__name__ + self._suffix
    
    def __repr__(self, full=False):
        """
        Return a string report essential user data for this BaseMod object
        """
        rstr = f'Name: {self.__name__(full=False)}\n'
        rstr += f"Output Size: {self.stats.out1}/{self.maxlen}\n"
        rstr += f"Max Iterations: {self.max_pulse_size}\n"
        if full:
            rstr += " - Last Pulse Stats - \n"
            rstr += self.stats.__str__()
        else:
            rstr += f'Last Pulse Rate: {self.stats["pulse rate"]:.2f} Hz (No. Iterations: {self.stats.niter})'
        
        return rstr

    def __str__(self):
        """Return self.__class__ string for this BaseMod object"""
        rstr = self.__class__ 
        return rstr
    
    def copy(self):
        """Return a deepcopy of this object"""
        return deepcopy(self)

    def pulse(self, input):
        """TEMPLATE METHOD
        
        Specifics for :class:`~PULSE.mod.base.BaseMod`

        :param input: collection of objects that will have elements removed using :meth:`~collections.deque.popleft` 
            and appended to **BaseMod.output** using :meth:`~collections.deque.append`
        :type input: collections.deque
        :return:
            - **output** (*collections.deque*) -- view of this BaseMod object's **BaseMod.deque** attribute
        
        TEMPLATE EXPLANATION

        Template method for all :mod:`~PULSE.mod` classes that executes a number of unit-tasks
        depending on the behaviors of sub-routine methods specific to each of these clases.
        This method also updates values in the **stats** attribute with metadata from the most
        recent call of this method.

        This method is inherited by all :mod:`~PULSE.mod` classes and has the following structure:
        
        A) Update **stats** 'starttime', 'in0', and 'out0' values

        B) Iteration Loop (up to **max_pulse_size** iterations)

            1) :meth:`~PULSE.mod.base.BaseMod.get_unit_input` - extract an object from **input** that will be used for the next step
            2) :meth:`~PULSE.mod.base.BaseMod.run_unit_process` - run the unit process for this class.
            3) :meth:`~PULSE.mod.base.BaseMod.store_unit_output` - merge the output from the previous step into the *output* attribute of this object
        
        C) Update **stats** 'stop', 'niter', 'endtime', 'in1', and 'out1' values

        """ 
        # Capture pulse stats at beginning of call and measure/check input
        self.stats.starttime = UTCDateTime.now()
        self.stats.in0 = self.check_input(input)
        self.stats.out0 = self.measure_output()
        self._continue_pulsing = True
        for _n in range(self.max_pulse_size):
            # get single object for unit_process
            unit_input = self.get_unit_input(input)
            # Run unit process if unit_input is not False
            if self._continue_pulsing:
                # Execute unit process
                unit_output = self.run_unit_process(unit_input)
            # Otherwise break for-loop & flag as early stopping
            else:
                self.stats.stop = 'early0'
                break

            # Capture output
            self.store_unit_output(unit_output)
            # Check if pulse should conclude early
            if self._continue_pulsing:
                pass
            else:
                # Break iteration late & update 'stop' reason
                self.stats.stop = 'early1'
                break
        # If completed due to max_pulse_size iterations
        if _n + 1 == self.max_pulse_size:
            self.stats.stop = 'max'
        # Capture pulse stats at conclusion of call
        self.stats.in1 = self.check_input(input)
        self.stats.out1 = self.measure_output()
        try:
            self.stats.niter = _n + 1
        except NameError:
            self.stats.niter = 0
        self.stats.endtime = UTCDateTime.now()

        return self.output

    def check_input(self, input):
        """
        POLYMORPHIC METHOD

        Last updated with :class:`~PULSE.module.base.BaseMod`

        Checks if **input** is iterable, and if so, this method returns the length of **input**

        :param input: input deque of objects to process
        :type input: collections.deque
        :return:
         - **input_size** (*int* or *NoneType*) - length of the input or None
        
        If **input** is not iterable or NoneType, logs CRITICAL and exits program on code *os.EX_DATAERR*.
        """
        if isinstance(input, deque):
            input_size = len(input)
            return input_size
        else:
            self.Logger.critical(f'input is not type deque. Quitting on code DATAERR ({os.EX_DATAERR})')
            sys.exit(os.EX_DATAERR)
    
    def measure_output(self):
        """POLYMORPHIC METHOD

        Last updated with :class`~PULSE.module.base.BaseMod`

        Measure the length of the **BaseMod.output** attribute

        :return:
         - **output_size** (*int*) -- length of **self.output**
        """        
        return len(self.output)
    
    def get_unit_input(self, input):
        """POLYMORPHIC METHOD

        Last updated with :class:`~PULSE.module.base.BaseMod`

        Get the input object for :meth:`~PULSE.module.base.BaseMod.run_unit_process` from
        the `input` provided to :meth:`~PULSE.module.base.BaseMod.pulse`

        :param input: deque of objects
        :type input: collections.deque
        :return:
         - **unit_input** (*object* or *bool*) -- object removed from **input** using :meth:`~collections.deque.popleft`.
                if input is empty, this method returns `False`, triggering early iteration stopping in :meth:`~PULSE.mod.base.BaseMod.pulse`
        """ 
        if self.ckeck_input(input) > 0:
            unit_input = input.popleft()
        else:
            unit_input = None
            self._continue_pulsing = False
        return unit_input
        
    def run_unit_process(self, unit_input):
        """POLYMORPHIC METHOD

        Last updated with :class: `~PULSE.module.base.BaseMod`

        Returns the input object as output

        :param unit_input: any unit input object, except False, which is reserved for early stopping
        :type unit_input: object
        :return:
         - **unit_output** (*object*)
        """
        unit_output = unit_input
        return unit_output        
    
    def store_unit_output(self, unit_output):
        """POLYMORPHIC METHOD

        Last updated with :class: `~PULSE.module.base.BaseMod`

        Attach unit_output to self.output using :meth:`~collections.deque.append`

        :param unit_output: unit output from :meth:`~PULSE.mod.base.BaseMod.run_unit_process`
        :type unit_output: object
        """        
        self.output.append(unit_output)
        # NOTE: This is another place where self._continue_pulsing can be updated for early stopping type 1
        
 
    #TODO: Make averaging 

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
    

    # def import_class(self, class_path_str):
    #     """Use the full extension ID of a class object to import that class within
    #     the local scope of this class-method and return the class object for use
    #     elsewhere

    #     e.g. class_path_str = 'obspy.core.trace.Trace'
    #       runs exec('from obspy.core.trace import Trace')
    #       and returns obj = Trace

    #     :param class_path_str: class extension ID
    #     :type class_path_str: str
    #     :return: class defining object
    #     :rtype: type
    #     """        
    #     if not isinstance(class_path_str, str):
    #         raise TypeError('class_path_str must be type str')
    #     elif '.' not in class_path_str:
    #         raise ValueError('class_path_str is not .-delimited. Does not look like a class __name__')
        
    #     parts = class_path_str.split('.')
    #     path = '.'.join(parts[:-1])
    #     clas = parts[-1]
    #     try:
    #         exec(f'from {path} import {clas}')
    #         obj = eval(f'{clas}')
    #         return obj
    #     except ImportError:
    #         self.Logger.critical(f'failed to import {class_path_str}')
    #         sys.exit(1)

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