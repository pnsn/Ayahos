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
import pandas as pd
from copy import deepcopy
from collections import deque
import logging, sys, datetime, os

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
    :var output: object that holds outputs from :meth:`~PULSE.mod.base.BaseMod.run_unit_process`. A :class:`~collections.deque` object.

        also see :meth:`~PULSE.mod.base.BaseMod.store_unit_output`
    :var max_pulse_size: maximum number of iterations for single call of :meth:`~PULSE.mod.base.BaseMod.pulse` 
    :var maxlen: maximum size of the **BaseMod.output** attribute. This is enforced with the :class:`~collections.deque` **maxlen** behavior. 
    :var Logger: a :class:`~logging.Logger` object named to match this class (BaseMod)
    :var _last_pulse_stats: a :class:`~dict` object summarizing stats from the last call of :meth:`~PULSE.mod.base.BaseMod.pulse`
        **start** (*datetime.datetime*) -- start time of last pulse
        **end** (*datetime.datetime*) - -end time of last pulse
        **exit** (*str*) -- string describing the nature of the :meth:`~PULSE.mod.base.BaseMod.pulse` conclusion
            'early' - triggered early stopping due to :meth:`~PULSE.mod.base.BaseMod.should_this_iteration_run`
            'late' - triggered early stopping due to :meth:`~PULSE.mod.base.BaseMod.should_next_iteration_run`
            'max' - if iterations reached **BaseMod.max_pulse_size**
        **count** (*int*) -- number of iterations completed in the call
        **in0** (*int*) -- measure of the input to pulse at the start of the call
        **in1** (*int*) -- measure of the input to pulse at the end of the call
        **out0** (*int*) -- measure of **BaseMod.output** at the start of the call
        **out1** (*int*) -- measure of **BaseMod.output** at the end of the call
        All values start as None
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
        self.Logger = logging.getLogger(f'{self.__name__()}')
        # Initialize output
        self.output = deque(maxlen=maxlen)
        self.maxlen = maxlen
        # Initialize pulse metadata
        self._last_pulse_stats = {'start': None,
                                  'end': None,
                                  'stop': None,
                                  'count': None,
                                  'in0': None,
                                  'in1': None,
                                  'out0': None,
                                  'out1': None}

        


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
    
    def __repr__(self):
        """
        Return a string report essential user data for this BaseMod object
        """
        rstr = f'Module Name: {self.__name__(full=False)}\n'
        rstr += f"Max Pulse Size: {self.max_pulse_size}\n"
        rstr += f"Max Output Size: {self.maxlen}\n"
        rstr += " - Last Pulse Stats - \n"
        for _k, _v in self._last_pulse_stats.items():
            rstr += f'{_k}: {_v}\n'
            # \nOutput: {len(self.output)} (type {type(self.output)})"
        return rstr

    def __str__(self):
        """Return self.__class__ string for this BaseMod object"""
        rstr = self.__class__ 
        return rstr
    
    def copy(self):
        """Return a deepcopy of this object"""
        return deepcopy(self)

    def pulse(self, input):
        """Abstracted method for :mod:`~PULSE.mod` classes that executes one or more unit-tasks
        depending on the behaviors of sub-routine methods specific to each of these clases.

        :param input: class-specific input
        :type input: class-specific. See the **measure_input** method for this class
        
        :return:
          - **output** (*class-specific*) -- aliased access to this object's **output** attribute.

        This method is inherited by all :mod:`~PULSE.mod` classes and has the following structure
        
        1) :meth:`~PULSE.mod.base.BaseMod.measure_input` - measure the input to determine the maximum number of iterations to run, up to *max_pulse_size*.
        ITERATION LOOP
            2) :meth:`~PULSE.mod.base.BaseMod.should_this_iteration_run` - determine if this iteration should be run, or if iterations should be stopped early.
            3) :meth:`~PULSE.mod.base.BaseMod.get_unit_input` - extract an object from **input** that will be used for the next step
            4) :meth:`~PULSE.mod.base.BaseMod.run_unit_process` - run the unit process for this class.
            5) :meth:`~PULSE.mod.base.BaseMod.store_unit_output` - merge the output from the previous step into the *output* attribute of this object
            6) :meth:`~PULSE.mod.base.BaseMod.should_next_iteration_run` - determine if the next iteration should be run.

        The start-time and end-time of this call of pulse, along with the number of completed iterations, are stored in private attributes
        """ 
        # Capture pulse stats at beginning of call and measure/check input
        self._last_pulse_stats.update({'start': datetime.datetime.now(),
                                       'in0': self.measure_input(input),
                                       'out0': self.measure_output()})
        # Start process counter
        nproc = 0
        for _n in range(self.max_pulse_size):
            # Check if this iteration should run
            if self.should_this_iteration_run(_n):
                pass
            else:
                self._last_pulse_stats.update({'stop': 'early'})
                break
            # get single object for unit_process
            unit_input = self.get_unit_input(input)
            # Execute unit process
            unit_output = self.run_unit_process(unit_input)
            # Increase process counter
            nproc += 1
            # Capture output
            self.store_unit_output(unit_output)
            # Check if pulse should conclude early
            if self.should_next_iteration_run(unit_output):
                pass
            else:
                # Break iteration late & update 'stop' reason
                self._last_pulse_stats.upadte({'stop': 'late'})
                break
        # If completed due to max_pulse_size iterations
        if _n + 1 == self.max_pulse_size:
            self._last_pulse_stats.update({'stop': 'max'})
        # Capture pulse stats at conclusion of call
        self._last_pulse_stats.update({'in1': self.measure_input(input),
                                       'out1': self.measure_output(),
                                       'count': nproc,
                                       'end': datetime.datetime.now()})
        return self.output

    def measure_input(self, input):
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

        Returns output of len(self.output)

        :return:
         - **output_size** (*int*) -- length of **self.output**
        """        
        return len(self.output)

    def should_this_iteration_run(self, iter_number):
        """POLYMORPHIC METHOD  

        last updated with :class:`~PULSE.mod.base.BaseMod`

        Should this iteration inside :meth:`~PULSE.mod.base.BaseMod.pulse` be run?
        
        Criteria:
         - the max_pulse_inputs

        :param iter_number: iteration number
        :type iter_number: int
        :return status: continue to next iteration?
        :rtype status: bool
        """
        input_size = self._last_pulse_stats['in0']
        # if input is non-empty
        if len(input) > 0:
            # and iter_number +1 is l.e. length of input
            if iter_number + 1 <= input_size:
                # Signal to run this 
                status = True
        else:
            status = False
        return status
    
    def get_unit_input(self, input):
        """
        POLYMORPHIC METHOD
        Last updated with :class:`~PULSE.module.base.BaseMod`

        Get the input object for :meth:`~PULSE.module.base.BaseMod.run_unit_process` from
        the `input` provided to :meth:`~PULSE.module.base.BaseMod.pulse`

        :param input: deque of objects to pass to **run_unit_process**
        :type input: collections.deque
        :return unit_input: unit_input popleft'd from input
        :rtype unit_input: any
        """        
        if isinstance(input, deque):
            unit_input = input.popleft()
            return unit_input
        else:
            self.Logger.error(f'input object was incorrect type')
            sys.exit(1)        

    def run_unit_process(self, unit_input):
        """
        POLYMORPHIC METHOD
        Last updated with :class: `~PULSE.module.base.BaseMod`

        return unit_output = unit_input

        :param obj: input object
        :type obj: any
        :return unit_output: output object
        :rtype unit_output: any
        """
        unit_output = unit_input
        return unit_output        
    
    def store_unit_output(self, unit_output):
        """
        POLYMORPHIC METHOD
        Last updated with :class: `~PULSE.module.base.BaseMod`

        Append unit_output to self.output

        run Mod().output.append(unit_output)

        :param unit_output: standard output object from unit_process
        :type unit_output: any
        """        
        self.output.append(unit_output)
        
    def should_next_iteration_run(self, unit_output):
        """POLYMORPHIC METHOD

        Last updated with :class:`~PULSE.module.base.BaseMod`

        Check if the next iteration should be run based on unit_output

        Returns status = True unconditionally

        :param unit_output: output run_unit_process
        :type unit_output: 
        :return:
         - **status** (*bool*) -- should the next iteration be run? Always returns True.
         
        """
        if isinstance(unit_output, object):
            status = True
        else:
            self.Logger.critical('Somehow produced a non-object unit_output... quitting.')
            sys.exit(1)
        return status
    


    # def capture_pulse_iteration_metadata(self, pulse_starttime, input_size, nproc, early_stop_code):
    #     """
    #     POLYMORPHIC METHOD
    #     Last updated with :class:`~PULSE.module.base.BaseMod`

    #     Captures metadata from the most recent pulse and updates
    #     the representative reporting string attribute *self.report*

    #     :param pulse_starttime: time the pulse was triggered
    #     :type pulse_starttime: pandas.core.timestamp.Timestamp
    #     :param input_size: size of the input at the start of a call of pulse()
    #     :type input_size: int
    #     :param nproc: number of unit processes completed in this pulse
    #     :type nproc: int
    #     :param early_stop_code: integer standin for True (1) and False (0) if the pulse hit an early termination clause
    #     :type early_stop_code: int
    #     """        
    #     # Calculate/capture new metadata values for this pulse
    #     timestamp = pd.Timestamp.now().timestamp()
    #     runtime = (timestamp - pulse_starttime)
    #     output_size = len(self.output)
    #     df_line = pd.DataFrame(dict(zip(self._keys_meta,
    #                                    [timestamp, nproc, runtime, input_size, output_size, early_stop_code])),
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
        #         self._update_metadata(pulse_starttime, input_size, nproc, 1)
        #         break
        # # Get alias of self.output as output
        # output = self.output
        # self.cli_reporter.update(
        # # TODO: make the reporting a separate 

        # # If the 
        # if _n + 1 == self.max_pulse_size:
        #     self._update_metadata(pulse_starttime, input_size, nproc, 0)

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