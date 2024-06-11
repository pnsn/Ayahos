"""
:submodule: wyrm.unit.wyrm
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module hosts the fundamental class definition for all
    other *Wyrm classes -- "Wyrm" -- and serves as a template
    for the minimum required methods of each successor class. 
"""
import pandas as pd
from copy import deepcopy
from collections import deque
import logging, sys

# Logger = logging.getLogger(__name__)

def add_class_name_to_docstring(cls):
    for name, method in vars(cls).items():
        if callable(method) and hasattr(method, "__doc__"):
            method.__doc__ = method.__doc__.format(class_name_lower=cls.__name__.lower(),
                                                   class_name_camel=cls.__name__,
                                                   class_name_upper=cls.__name__.upper())
    return cls



Logger = logging.getLogger(__name__)

# @add_class_name_to_docstring
class Wyrm(object):
    """Fundamental base class template all other *Wyrm classes with the following
    template methods

    Wyrm().pulse() - performs multiple iterations of the following polymorphic subroutines
        Wyrm()._should_this_iteration_run() - check if this iteration in pulse should be run
        Wyrm()._unit_input_from_input() - get the object input for _unit_process()
        Wyrm()._unit_process() - execute a single operation on the input object
        Wyrm()._capture_unit_output() - do something to save the output of _unit_process
        Wyrm()._should_next_iteration_run() - decide if the next iteration should be run

    Wyrm.output - collector of outputs from core_process(). Should be modified along with
                Wyrm().core_process() to meet your needs.

    :param max_pulse_size: maximum number of iterations to run for wyrm.pulse(), defaults to 10
    :type max_pulse_size: int, optional
    
    """

    def __init__(self, max_pulse_size=10, meta_memory=3600, report_period=None):
        """Initialize a {class_name_camel} object

        :param max_pulse_size: maximum number of iterations to run for each call of :meth:`~ayahos.wyrms.wyrm.Wyrm.pulse`, defaults to 10
        :type max_pulse_size: int, optional
        :param meta_memory: maximum age of metadata logging in seconds, defaults to 3600.
        :type meta_memory: float, optional
        :param report_period: send report to logging every `report_period` seconds, defaults to None
                    Positive values turn reporting on
                    None value turns reporting off
        :type report_period: float or NoneType, optional
        """
        # Logger.debug('Initializing a Wyrm object')
        # Compatability check for max_pulse_size
        if isinstance(max_pulse_size, (int, float)):
            if 1 <= max_pulse_size:
                self.max_pulse_size = int(max_pulse_size)
            else:
                raise ValueError('max_pulse_size must be g.e. 1 ')
        else:
            raise TypeError('max_pulse_size must be positive int-like')
        
        # if isinstance(mute_pulse_logging, bool):
        #     self.mute_pulse_logging = mute_pulse_logging
        # else:
        #     Logger.warning('mute_pulse_logging is not type bool - defaulting to True')
        #     self.mute_pulse_logging = True
        # output holder
        self.output = deque()
        # Metadata keys for reporting
        self._keys_meta = ['time', '# proc', 'runtime', 'isize', 'osize', 'exit']
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
        
        if report_period is None:
            self.report_period = False
        elif isinstance(report_period, (int, float)):
            if 0 < report_period <= self.meta_memory:
                self.report_period = report_period
            else:
                raise ValueError
        else:
            raise TypeError
        
        self._last_report_time = pd.Timestamp.now().timestamp()
        self._pulse_rate = None

    def __name__(self):
        """Return the camel-case name of this class without
        the submodule extension

        alias of self.__class__.__name__
        :return: class name
        :rtype: str
        """        
        return self.__class__.__name__
    
    def __repr__(self):
        """
        Provide a string representation string of essential user data for this {class_name_camel}
        """
        # rstr = "~~wyrm~~\nFundamental Base Class\n...I got no legs...\n"
        rstr = f'{self.__class__}\n'
        rstr += f"Max Pulse Size: {self.max_pulse_size}\nOutput: {len(self.output)} (type {type(self.output)})"
        return rstr

    def __str__(self):
        """
        Return self.__class__ for this wyrm
        """
        rstr = self.__class__ 
        #(max_pulse_size={self.max_pulse_size})'
        return rstr
    
    def copy(self):
        """
        Return a deepcopy of this Wyrm object

        :return: copy of this object
        :rtype: ayahos.wyrms.wyrm.Wyrm-like
        """
        return deepcopy(self)

    def pulse(self, input):
        """
        TEMPLATE METHOD
        Last updated with :class:`~ayahos.wyrms.wyrm.Wyrm`

        Run up to max_pulse_size iterations of _unit_process()

        NOTE: Houses the following polymorphic methods that can be modified for decendent Wyrm-like classes
            _should_this_iteration_run: check if iteration should continue (early stopping opportunity)
            _unit_input_from_input: get input object for _unit_process
            _unit_process: run core process
            _capture_unit_output: attach _unit_process output to self.output
            _should_next_iteration_run: check if the next iteration should occur (early stopping opportunity)
            _update_metadata: update metadata dataframe with a summary of pulse activity

        :param input: standard input
        :type input: collections.deque
            see ayahos.core.wyrms.wyrm.Wyrm._unit_input_from_input()
        :return output: aliased access to this object's **output** attribute
        :rtype output: typically collections.deque or ayahos.core.dictstream.DictStream
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
        #     Logger.info(f'{self.__name__} {nproc} processes run (MAX: {self.max_pulse_size})')
        # Get alias of self.output as output
        output = self.output
        # If the 
        if _n + 1 == self.max_pulse_size:
            self._update_metadata(pulse_starttime, input_size, nproc, 0)

        # Check if reporting is turned on
        if self.report_period:
            # If so, get the now-time
            nowtime = pd.Timestamp.now().timestamp()
            # If the elapsed time since the last report transmission is exceeded
            if self.report_period > nowtime - self._last_report_time:
                # Transmit report to logging
                Logger.info(self._generate_report_string())
                # Update the last report time
                self._last_report_time = nowtime

        return output

    def _measure_input_size(self, input):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.wyrm.Wyrm`

        take a reference measurement for the input before starting
        iterations for :meth:`~ayahos.core.wyrm.Wyrm.pulse(input)` prior
        to starting the pulse.

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
        POLYMORPHIC - last updated with :class:`~ayahos.wyrms.wyrm.Wyrm`

        Should this iteration in :meth:`~ayahos.wyrms.wyrm.Wyrm.pulse()` be run?
        
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
        Last updated with :class: `~ayahos.wyrms.wyrm.Wyrm`

        Get the input object for this Wyrm's _unit_process

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
            Logger.error(f'input object was incorrect type')
            sys.exit(1)
            raise TypeError
        

    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last updated with :class: `~ayahos.wyrms.wyrm.Wyrm`

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
        Last updated with :class: `~ayahos.wyrms.wyrm.Wyrm`

        Append unit_output to self.output

        run Wyrm().output.append(unit_output)

        :param unit_output: standard output object from unit_process
        :type unit_output: any
        :return: None
        :rtype: None
        """        
        self.output.append(unit_output)
        return None
        
    def _should_next_iteration_run(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class: `~ayahos.wyrms.wyrm.Wyrm`

        check if the next iteration should be run based on unit_output

        Returns status = True unconditionally

        :param unit_output: output _unit_process
        :type unit_output: object
        :return status: should the next iteration be run?
        :rtype: bool
        """
        status = True
        return status

    def _update_metadata(self, pulse_starttime, input_size, nproc, exit_code):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.wyrm.Wyrm`

        Captures metadata from the most recent pulse and updates
        the representative reporting string attribute that is
        collected by :class:`~ayahos.wyrms.tubewyrm.TubeWyrm` objects
        to compose periodic process flow reports.

        :param pulse_starttime: time the pulse was triggered
        :type pulse_starttime: pandas.core.timestamp.Timestamp
        :param input_size: size of the input at the start of a call of pulse()
        :type input_size: int
        :param nproc: number of unit processes completed in this pulse
        :type nproc: int
        :param exit_code: integer standin for True (1) and False (0) if the pulse hit an early termination clause
        :type exit_code: int
        """        
        # Calculate/capture new metadata values for this pulse
        timestamp = pd.Timestamp.now().timestamp()
        runtime = (timestamp - pulse_starttime)
        output_size = len(self.output)
        S_line = pd.DataFrame(dict(zip(self._keys_meta,
                                       [timestamp, nproc, runtime, input_size, output_size, exit_code])),
                               index=[0])
        # Append line to dataframe
        self._metadata = self._metadata._append(S_line, ignore_index=True)

        # Trim outdated metadata
        oldest_time = timestamp - self.meta_memory
        self._metadata = self._metadata[self._metadata.time >= oldest_time]

        # Update Report       
        df_last = self._metadata[self._metadata.time == self._metadata.time.max()]
        df_last.index = ['last']
        self.report = df_last 
        self.report = pd.concat([self.report, self._metadata.agg(['mean','std','min','max'])], axis=0, ignore_index=False)
        # Calculate pulse rate
        self.pulse_rate = len(self._metadata)/(self._metadata.time.max() - self._metadata.time.min())

    def _generate_report_string(self):
        header = f'pulse rate: {self.pulse_rate:.2e} Hz'
        header += f'\nsample period: {self.meta_memory} sec'
        return f'\n{header}\n{self.report}'
    
# from ayahos import DictStream
# from obspy import read
# from ayahos.wyrms import Wyrm
# from collections import deque
# q = deque([DictStream(read())])
# wyrm = Wyrm()
# y = wyrm.pulse(q)