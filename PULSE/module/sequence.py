"""
:module: PULSED.module.unit.package
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This contains class definitions for data processing unit modules that bundle
    other unit modules for sequential or parallel execution


Classes
-------
:class:`~PULSED.module.bundle.SequenceMod`
:class:`~PULSED.module.bundle.ParallelMod` (WIP)
"""
"""
:module: module.coordinating.sequence
:auth: Nathan T Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains class defining submodules for coordinating sets
    of modules to form parts of fully operational Wyrm modules, including:

    SequenceMod - a submodule class for running a sequence of modules' pulse()
              methods in series starting with a single input to the first
              module in the series and providing a single output from the
              last module in the series

              y = moduleN.pulse(...pulse(module1.pulse(module0.pulse(x))))

TODO: Turn status from _capture_unit_out into a representation of nproc
    idea is to say "keep going" if the SequenceMod is conducting any processes

"""
import logging
import numpy as np
import pandas as pd
from collections import deque
from PULSE.module._base import _BaseMod


Logger = logging.getLogger(__name__)

class SequenceMod(_BaseMod):
    """
    Unit module class facilitating chained execution of pulse(x) class methods
    for a sequence module unit_inputects, with each module.pulse(x) taking the prior
    member's pulse(x) output as its input.

    The pulse method operates as follows
    for sequence = {key0: <module0>, key2: <module2> , key1: <module1>}
        SequenceMod.pulse(x) = module1.pulse(module2.pulse(module0.pulse(x)))

        Note that the dictionary ordering dictates the execution order!
    """

    def __init__(
            self,
            sequence={},
            log_pulse_summary=True,
            wait_sec=0.0,
            max_pulse_size=1,
            meta_memory=3600,
            report_period=False,
            max_output_size=1e10,
            report_fields=['last','mean']):
        """
        Create a SequenceMod unit_inputect
        :param sequence: collection of modules that are executed in their provided order. 
                    dict-type entries allow for naming of component modules for user-friendly assessment. list, deque, and Wyrm-type
                    all other input types use a 0-indexed key.
        :type sequence: dict, list, or tuple
        :param wait_sec: seconds to wait between execution of each *Mod in sequence, default is 0.0
        :type wait_sec: float
        :param max_pulse_size: number of times to run the sequence of pulses, default is 1
        :type max_pulse_size: int
        """
        # Inherit from _BaseMod
        super().__init__(max_pulse_size=max_pulse_size,
                         meta_memory=meta_memory,
                         report_period=report_period,
                         max_output_size=max_output_size)
        if isinstance(log_pulse_summary, bool):
            self.log_pulse_summary = log_pulse_summary
        # sequence compat. checks
        if isinstance(sequence, _BaseMod):
            self.sequence = {0: sequence}
        elif isinstance(sequence, dict):
            if all(isinstance(_w, _BaseMod) for _w in sequence.values()):
                self.sequence = sequence
            else:
                raise TypeError('All elements in sequence must be type _BaseMod')
            
        # Handle case where a list or deque are provided
        elif isinstance(sequence, (list, deque)):
            if all(isinstance(_w, _BaseMod) for _w in sequence):
                # Default keys are the sequence index of a given module
                self.sequence = {_k: _v for _k, _v in enumerate(sequence)}
            else:
                raise TypeError('All elements in sequence must be type _BaseMod')
        else:
            raise TypeError('sequence must be a single _BaseMod-type unit_input or a list or dictionary thereof')
        
        # wait_sec compat. checks
        if not isinstance(wait_sec, (int, float)):
            raise TypeError('wait_sec must be float-like')
        elif wait_sec < 0:
            raise ValueError('wait_sec must be non-negative')
        else:
            self.wait_sec = wait_sec
        # Create a list representation of keys
        self.names = list(sequence.keys())
        # Alias the output of the last module in sequence to self.output (inherited from _BaseMod)
        if len(self.sequence) > 0:
            self._alias_sequence_output()

        if isinstance(report_fields, str):
            report_fields = [report_fields]
        elif isinstance(report_fields, list):
            pass
        else:
            raise TypeError

        if all(e_ in ['last','mean','std','min','max'] for e_ in report_fields):
            self.report_fields = report_fields

    def update(self, new_dict):
        """
        Apply update to sequence using the 
        dict.update(new_dict) builtin_function_or_method
        and then update relevant attributes of this SequenceMod

        This will update existing keyed entries and append new
        at the end of self.sequence (same behavior as dict.update)

        :param new_dict: dictionary of {name:*Mod} pairs to  
        :type new_dict: dict
        :updates: 
            - **self.sequence** - updates sequence as specified
            - **self.names** - updates the list of keys in sequence
            - **self.output** - re-aliases to the output attribute of the last module in sequence

        """
        # Safety catches identical to those in __init__
        if not isinstance(new_dict, dict):
            raise TypeError('new_dict must be type dict')
        elif not all(isinstance(_m, _BaseMod) for _m in new_dict.values()):
            raise TypeError('new_dict can only have values of type PULSED.module._base._BaseMod')
        else:
            pass
        # Run updates on sequence
        self.sequence.update(new_dict)
        # Update names attribute
        self.names = list(self.sequence.keys())
        # Set outputs as alias to last module
        self._alias_sequence_output()


    def _alias_sequence_output(self):
        """
        Alias the self.output attribute of the last module-type unit_inputect
        in this SequenceMod's sequence to this SequenceMod's output attribute

        i.e., 
        SequenceMod.sequence = {0: module0, 1: module1, 2: module2}
        SequenceMod.output = module2.output
        """        
        self.output = list(self.sequence.values())[-1].output
    
    def remove(self, key):
        """
        Convenience wrapper of the dict.pop() method
        to remove an element from self.sequence and
        associated attributes

        :: INPUT ::
        :param key: valid key in self.sequence.keys()
        :type key: unit_inputect

        :: RETURN ::
        :return popped_item: popped (key, value) pair
        :rtype popped_item: tuple
        """
        if key not in self.sequence.keys():
            raise KeyError(f'key {key} is not in self.sequence.keys()')
        # Remove key/val combination from dict
        val = self.sequence.pop(key)
        # Update names attribute
        self.names = list(self.sequence.keys())
        # Return key and value
        self._alias_sequence_output()
        return (key, val)

    def reorder(self, reorder_list):
        """
        Reorder the current contents of sequence using either
        an ordered list of sequence

        :: INPUT ::
        :param reorder_list: unique list of keys from self.module
        :type reorder_list: list of Wyrm-likes
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
        # Handle input (re)ordered sequence key list
        if all(_e in self.sequence.keys() for _e in reorder_list):
            tmp = {_e: self.sequence[_e] for _e in reorder_list}
        # Handle input (re)ordered index list
        elif all(_e in np.arange(0, len(reorder_list)) for _e in reorder_list):
            tmp_keys = list(self.sequence.keys())
            tmp = {_k: self.sequence[_k] for _k in tmp_keys}

        # Run updates
        self.sequence = tmp
        self.names = list(tmp.keys())
        self._alias_sequence_output()


    def __repr__(self, extended=False):
        """Provide a user-friendly summary of the contents of this SequenceMod
        :: INPUT ::
        :param extended: show full __repr__ output of component Wyrms? , defaults to False
        :type extended: bool, optional

        :: OUTPUT ::
        :return rstr: string representation of this Wyrm's contents
        :rtype rstr: str
        """
        rstr = f'{super().__repr__()}\n'
        rstr = f"(wait: {self.wait_sec} sec)\n"
        for _i, (_k, _v) in enumerate(self.sequence.items()):
            # Provide index number
            rstr += f'({_i:<2}) '
            # Provide labeling of order
            if _i == 0:
                rstr += "(head) "
            elif _i == len(self.sequence) - 1:
                rstr += "(tail) "
            else:
                rstr += "  ||   "
            rstr += f"{_k} | "
            if extended:
                rstr += f'{_v}\n'
            else:
                rstr += f'{_v.__name__()}\n'
        return rstr
    
    #############################
    # PULSE POLYMORPHIC METHODS #
    #############################

    def _should_this_iteration_run(self, input, input_measure, iterno):
        """
        POLYMORPHIC

        Last updated with :class: `~PULSED.module.bundle.SequenceMod`

        always return status = True
        Execute max_pulse_size iterations regardless of internal processes

        :param input: Unused
        :type input: any
        :param iterno: Unused
        :type iterno: any
        :return status: continue iteration - always True
        :rtype: bool
        """        
        status = True
        return status
    
    def _unit_input_from_input(self, input):
        """
        POLYMORPHIC
        Last updated with :class: `~PULSED.module.bundle.SequenceMod` 

        Pass the standard input directly to the first module in sequence

        :param input: standard input unit_inputect
        :type input: varies, depending on input expected by first module in sequence
        :return unit_input: view of standard input unit_inputect
        :rtype unit_input: varies
        """        
        unit_input = input
        return unit_input

    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last updated with :class: `~PULSED.module.bundle.SequenceMod`

        Chain pulse() methods of modules in sequence
        passing unit_input as the input to the first 

            module2.pulse(module1.pulse(module0.pulse(unit_input)))

        In this example output is captured by module2's pulse and
        SequenceMod.output is a view of module2.output

        :param unit_input: standard input for the pulse method of the first module in sequence
        :type unit_input: varies
        :return unit_output: sum of nproc output by each module in moduledict for this iteration
        :rtype: int
        """ 
        for j_, module_ in enumerate(self.sequence.values()):
            if j_ == 0:
                y = module_.pulse(unit_input)
            else:
                y = module_.pulse(y)
        # TODO: Placeholder unconditional True - eventually want to pass a True/False
        # up the line to say if there should be early stopping
        unit_output = None
        return unit_output

    def _capture_unit_output(self, unit_output):
        """
        POLYMORPHIC
        Last updated by :class: `~PULSED.module.bundle.SequenceMod`

        Termination point - output capture is handled by the last
        Wyrm-Type object in sequence and aliased to this SequenceMod's
        output attribute.

        :param unit_output: sum of processes executed by the _unit_process call
        :type unit_output: int
        :return: None
        :rtype: None
        """
        return None   

    def _should_next_iteration_run(self, unit_output):
        """
        POLYMORPHIC
        Last updated by :class: `~PULSED.module.bundle.SequenceMod`

        Signal early stopping (status = False) if unit_output == 0

        :param unit_output: number of processes executed by _unit_process
            Sum of pulse 
        :type unit_output: int
        :return status: should the next iteration be run?
        :rtype status: bool
        """
        if unit_output == 0:
            status = False
        else:
            status = True
        return status

    def _update_report(self):
        """
        POLYMORPHIC
        Last updated with :class:`~PULSED.module.bundle.SequenceMod`

        Get the mean value line for each module and add information
        on the pulserate, number of logged pulses, and memory period
        for each module.
        """
        report_dict = {}
        for _n, _w in self.sequence.items():
            _r = _w.report
            _p = _w._pulse_rate
            for _m in self.report_fields:
                line = list(_r.loc[_m].values)
                line.append(_p)
                line.append(len(_w._metadata))
                line.append(_w.meta_memory)
                report_dict.update({f'{_n}({_m})': line})
        keys = self._keys_meta + ['p_rate','n_pulse','memory_sec']
        self.report = pd.DataFrame(report_dict, index=keys).T
        self.report.index.name = 'submod (stat)'