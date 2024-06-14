"""
:module: camper.module._base
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This contains the definition for the _BaseMod class that serves serves as a template
    (parent class) for all other modules.

Classes
-------
:class:`~camper.module._base._BaseMod`
"""

from numpy import nan
import pandas as pd
from copy import deepcopy
from collections import deque
import logging, sys

# self.Logger = logging.getLogger(__name__)

def add_class_name_to_docstring(cls):
    for name, method in vars(cls).items():
        if callable(method) and hasattr(method, "__doc__"):
            method.__doc__ = method.__doc__.format(class_name_lower=cls.__name__.lower(),
                                                   class_name_camel=cls.__name__,
                                                   class_name_upper=cls.__name__.upper())
    return cls



Logger = logging.getLogger(__name__)

# @add_class_name_to_docstring
class _BaseMod(object):
    """Fundamental base class template all other *Mod classes with the following
    template methods

    Mod().pulse() - performs multiple iterations of the following polymorphic subroutines
        Mod()._should_this_iteration_run() - check if this iteration in pulse should be run
        Mod()._unit_input_from_input() - get the object input for _unit_process()
        Mod()._unit_process() - execute a single operation on the input object
        Mod()._capture_unit_output() - do something to save the output of _unit_process
        Mod()._should_next_iteration_run() - decide if the next iteration should be run

    Mod.output - collector of outputs from core_process(). Should be modified along with
                Mod().core_process() to meet your needs.

    :param max_pulse_size: maximum number of iterations to run for Mod.pulse(), defaults to 10
    :type max_pulse_size: int, optional
    
    """

    def __init__(
        self,
        max_pulse_size=10,
        meta_memory=3600,
        report_period=False,
        max_output_size=1e6):
        """Initialize a _BaseMod object

        :param max_pulse_size: maximum number of iterations to run for each call of :meth:`~camper.module._base._BaseMod.pulse`, defaults to 10
        :type max_pulse_size: int, optional
        :param meta_memory: maximum age of metadata logging in seconds, defaults to 3600.
        :type meta_memory: float, optional
        :param report_period: send report to logging every `report_period` seconds, defaults to None
                    Positive values turn reporting on
                    None value turns reporting off
        :type report_period: float or NoneType, optional
        """

        # Compatability check for max_pulse_size
        if isinstance(max_pulse_size, (int, float)):
            if 1 <= max_pulse_size:
                self.max_pulse_size = int(max_pulse_size)
            else:
                raise ValueError('max_pulse_size must be g.e. 1 ')
        else:
            raise TypeError('max_pulse_size must be positive int-like')
        
        # Initialize output
        self.output = deque()

        # Metadata keys for reporting
        self._keys_meta = ['time', '# proc', 'runtime', 'isize', 'osize', 'early_stop']

        # Metadata holder for reporting
        self._metadata = pd.DataFrame(columns=self._keys_meta)
        self.report = pd.DataFrame()
        if isinstance(meta_memory, (int, float)):
            if meta_memory > 0:
                if meta_memory <= 24*3600:
                    self.meta_memory = meta_memory
                else:
                    raise NotImplementedError('pulse metadata logging currently does not support memory periods of more than 24 hours')
            else:
                raise ValueError
        else:
            raise TypeError
        
        if report_period is False:
            self.report_period = False
        elif isinstance(report_period, (int, float)):
            if 0 < report_period <= self.meta_memory:
                self.report_period = report_period
            else:
                self.Logger.critical(f'{self.__class__} | report_period greater than memory period!')
                sys.exit(1)
        else:
            raise TypeError
        
        if max_output_size is None:
            self.max_output_size = 1e12
            self.Logger.critical(f'setting non-limited output size for {self.__class__.__name__} object to {self.max_output_size}')
        elif isinstance(max_output_size, (int,float)):
            if 0 < max_output_size <= 1e12:
                self.max_output_size = max_output_size
        else:
            raise TypeError

        self._last_report_time = pd.Timestamp.now().timestamp()
        self._pulse_rate = nan
        self.Logger = logging.getLogger(f'{self.__name__()}')


    def __name__(self, full=False):
        """Return the camel-case name of this class with or without the submodule extension
        :param full: use the full class path? Defaults to False
        :type full: bool, optional
        :return: class name
        :rtype: str
        """
        if full:
            return self.__class__.__str__(self)
        else:
            return self.__class__.__name__
    
    def __repr__(self):
        """
        Provide a string representation string of essential user data for this *Mod object
        """
        rstr = f'{self.__class__}\n'
        rstr += f"Max Pulse Size: {self.max_pulse_size}\nOutput: {len(self.output)} (type {type(self.output)})"
        return rstr

    def __str__(self):
        """
        Return self.__class__ for this *Mod
        """
        rstr = self.__class__ 
        #(max_pulse_size={self.max_pulse_size})'
        return rstr
    
    def copy(self):
        """
        Return a deepcopy of this *Mod object
        """
        return deepcopy(self)

    def pulse(self, input):
        """
        TEMPLATE METHOD
        Last updated with :class:`~camper.module._base._BaseMod`

        Run up to max_pulse_size iterations of _unit_process()

        NOTE: Houses the following polymorphic methods that can be modified for decendent Mod-like classes
            _should_this_iteration_run: check if iteration should continue (early stopping opportunity)
            _unit_input_from_input: get input object for _unit_process
            _unit_process: run core process
            _capture_unit_output: attach _unit_process output to self.output
            _should_next_iteration_run: check if the next iteration should occur (early stopping opportunity)
            _update_metadata: update metadata dataframe with a summary of pulse activity

        :param input: standard input
        :type input: collections.deque
            see camper.core.module._base._BaseMod._unit_input_from_input()
        :return output: aliased access to this object's **output** attribute
        :rtype output: typically collections.deque or camper.core.dictstream.DictStream
        """ 
        input_size = self._measure_input_size(input)
        nproc = 0
        pulse_starttime = pd.Timestamp.now().timestamp()
        for _n in range(self.max_pulse_size):
            # Check if this iteration should proceed
            if self._should_this_iteration_run(input, input_size, _n):
                pass
            else:
                self._update_metadata(pulse_starttime, input_size, nproc, 1)
                break
            # get single object for unit_process
            unit_input = self._unit_input_from_input(input)
            # Execute unit process
            unit_output = self._unit_process(unit_input)
            # Increase process counter
            nproc += 1
            # Capture output
            self._capture_unit_output(unit_output)
            #  Check if early stopping should occur at the end of this iteration
            if self._should_next_iteration_run(unit_output):
                pass
            else:
                self._update_metadata(pulse_starttime, input_size, nproc, 1)
                break
        # if not self.mute_pulse_logging:
        #     self.Logger.info(f'{self.__name__} {nproc} processes run (MAX: {self.max_pulse_size})')
        # Get alias of self.output as output
        output = self.output
        # If the 
        if _n + 1 == self.max_pulse_size:
            self._update_metadata(pulse_starttime, input_size, nproc, 0)

        self._update_report()
        self._update_pulse_rate()

        # Check if reporting is turned on
        if self.report_period:
            # If so, get the now-time
            nowtime = pd.Timestamp.now().timestamp()
            # If the elapsed time since the last report transmission is exceeded
            if self.report_period <= nowtime - self._last_report_time:
                # Transmit report to logging
                self.Logger.info(f'\n\n{self._generate_report_string()}\n')
                # Update the last report time
                self._last_report_time = nowtime

        return output

    def _measure_input_size(self, input):
        """
        POLYMORPHIC
        Last updated with :class:`~camper.module._base._BaseMod`

        take a reference measurement for the input before starting
        iterations within the pulse() method.

        This version measures the length of input.

        :param input: standard input
        :type input: varies, deque here
        :return input_size: representative measure of input
        :rtype: int-like
        """        
        if input is None:
            input_size = self.max_pulse_size
        else:
            input_size = len(input)
        return input_size

    def _should_this_iteration_run(self, input, input_size, iter_number):
        """
        POLYMORPHIC - last updated with :class:`~camper.module._base._BaseMod`

        Should this iteration in :meth:`~camper.module._base._BaseMod.pulse()` be run?
        
        Criteria:
         - input is type collections.deque
         - len(input) > 0
         - iter_number + 1 <= len(input)
        
        :param input: input data object collection
        :type input: collections.deque
        :param iter_number: iteration number
        :type iter_number: int
        :return status: continue to next iteration?
        :rtype status: bool
        """
        status = False
        # if input is deque
        if isinstance(input, deque):
            # and input is non-empty
            if len(input) > 0:
                # and iter_number +1 is l.e. length of input
                if iter_number + 1 <= input_size:
                    status = True
        return status
    
    def _unit_input_from_input(self, input):
        """
        POLYMORPHIC
        Last updated with :class: `~camper.module._base._BaseMod`

        Get the input object for this Mod's _unit_process

        :param input: standard input object
        :type input: collections.deque
        :return unit_input: unit_input popleft'd from input
        :rtype unit_input: any
        
        :raises TypeError: if input is not expected type
        
        """        
        if isinstance(input, deque):
            unit_input = input.popleft()
            return unit_input
        else:
            self.Logger.error(f'input object was incorrect type')
            sys.exit(1)
            raise TypeError
        

    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last updated with :class: `~camper.module._base._BaseMod`

        return unit_output = unit_input

        :param obj: input object
        :type obj: any
        :return unit_output: output object
        :rtype unit_output: any
        """
        unit_output = unit_input
        return unit_output        
    
    def _capture_unit_output(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class: `~camper.module._base._BaseMod`

        Append unit_output to self.output

        run Mod().output.append(unit_output)

        :param unit_output: standard output object from unit_process
        :type unit_output: any
        :return: None
        :rtype: None
        """        
        self.output.append(unit_output)
        extra = len(self.output) - self.max_output_size
        # if extra > 0:
        #     self.Logger.info(f'{self.__class__.__name__} object reached max_output_size. Deleting {extra} oldest values')

        while len(self.output) > self.max_output_size:
            self.output.popleft()
        
    def _should_next_iteration_run(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class: `~camper.module._base._BaseMod`

        check if the next iteration should be run based on unit_output

        Returns status = True unconditionally

        :param unit_output: output _unit_process
        :type unit_output: object
        :return status: should the next iteration be run?
        :rtype: bool
        """
        status = True
        return status

    def _update_metadata(self, pulse_starttime, input_size, nproc, early_stop_code):
        """
        POLYMORPHIC
        Last updated with :class:`~camper.module._base._BaseMod`

        Captures metadata from the most recent pulse and updates
        the representative reporting string attribute *self.report*

        :param pulse_starttime: time the pulse was triggered
        :type pulse_starttime: pandas.core.timestamp.Timestamp
        :param input_size: size of the input at the start of a call of pulse()
        :type input_size: int
        :param nproc: number of unit processes completed in this pulse
        :type nproc: int
        :param early_stop_code: integer standin for True (1) and False (0) if the pulse hit an early termination clause
        :type early_stop_code: int
        """        
        # Calculate/capture new metadata values for this pulse
        timestamp = pd.Timestamp.now().timestamp()
        runtime = (timestamp - pulse_starttime)
        output_size = len(self.output)
        S_line = pd.DataFrame(dict(zip(self._keys_meta,
                                       [timestamp, nproc, runtime, input_size, output_size, early_stop_code])),
                               index=[0])
        # Append line to dataframe
        self._metadata = self._metadata._append(S_line, ignore_index=True)

        # Trim outdated metadata
        oldest_time = timestamp - self.meta_memory
        self._metadata = self._metadata[self._metadata.time >= oldest_time]

    def _update_report(self):
        # Update Report       
        df_last = self._metadata[self._metadata.time == self._metadata.time.max()]
        df_last.index = ['last']
        self.report = df_last 
        self.report = pd.concat([self.report, self._metadata.agg(['mean','std','min','max'])], axis=0, ignore_index=False)
        self.report.time = self.report.time.apply(lambda x: pd.Timestamp(x, unit='s'))
    def _update_pulse_rate(self):
        # Calculate pulse rate
        nd = len(self._metadata)
        if nd > 1:
            dt = (self._metadata.time.max() - self._metadata.time.min())
            if dt > 0:
                self._pulse_rate = nd/dt

    def _generate_report_string(self):
        header = f'~~^v~~~ {pd.Timestamp.now()} | {self.__name__()} ~~^v~~~\n'
        header += f'pulse rate: {self._pulse_rate:.2e} Hz | max pulse size: {self.max_pulse_size}\n'
        header += f'sample period: {self.meta_memory} sec | max output size: {self.max_output_size}'
        return f'{header}\n{self.report}\n'
    

    def import_class(self, class_path_str, **kwargs):
        parts = class_path_str.split('.')
        path = '.'.join(parts[:-1])
        clas = parts[-1]
        try:
            exec(f'from {path} import {clas}')
            obj = eval(f'{clas}')
            return obj
        except ImportError:
            self.Logger.critical(f'failed to import {class_path_str}')
            sys.exit(1)
