"""
:module: PULSE.module.unit.package
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This contains class definitions for data processing unit modules that bundle
    other unit modules for sequential or parallel execution


Classes
-------
:class:`~PULSE.module.bundle.SequenceMod`
:class:`~PULSE.module.bundle.ParallelMod` (WIP)
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
import sys, configparser
import numpy as np
import pandas as pd
from collections import deque
from PULSE.module._base import _BaseMod

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
            raise TypeError('new_dict can only have values of type PULSE.module._base._BaseMod')
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

        Last updated with :class: `~PULSE.module.bundle.SequenceMod`

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
        Last updated with :class: `~PULSE.module.bundle.SequenceMod` 

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
        Last updated with :class: `~PULSE.module.bundle.SequenceMod`

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
        Last updated by :class: `~PULSE.module.bundle.SequenceMod`

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
        Last updated by :class: `~PULSE.module.bundle.SequenceMod`

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
        Last updated with :class:`~PULSE.module.bundle.SequenceMod`

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




class SequenceBuilderMod(SequenceMod):
    """
    Defines a self-building :class:`~PULSE.module.sequence.SequenceMod` child-class using regularly formatted
    inputs from a configuration file (*.ini) using the ConfigParser Python library

    :param config_file: file name and path for the desired configuration file, which must contain the sections decribed below
    :type config_file: str
    :param starting_section: unique section name for this sequence, defaults to 'Sequence_Module'

    config_file.ini format
    '''
    [Sequence_Module]   # <- Re-assign to match with **starting_section** in your .ini file to allow for multiple calls
    class: PULSE.module.sequence.SequenceBuilderMod
    sequence: Sequence_0
    max_pulse_size: 1000
    ...

    [Sequence_0]
    mod0_name: mod0_section_name
    mod1_name: mod1_section_name
    mod2_name: mod1_section_name  # <- not a typo, section names can be called multiple times!
    ...
    '''



    """
    # Define special section keys (allows overwrite outside of __init__ for things like PulseMod_EW)
    special_keys = ['sequence','class','module']

    def __init__(self, config_file, starting_section='Sequence_Module'):
        """
        Initialize a :class:`~PULSE.module.sequence.SequenceBuilderMod` object
        """

        # Initialize config parser
        self.cfg = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        # Read configuration file
        self.cfg.read(config_file)

        # Compatablity checks with starting_section & cross-check with cfg
        if isinstance(starting_section, str):
            if starting_section not in self.cfg._sections.keys():
                self.Logger.critical('KeyError: starting_section not in config_file')
                sys.exit(1)
            else:
                self._modname = starting_section
        else:
            self.Logger.critical('TypeError: starting_section must be type str')
            sys.exit(1)
    
        # Get sequence name
        try:
            _seqname = self.cfg.get(self._modname, 'sequence')
        except:
            self.Logger.critical(f'sequence-definining entry for "sequence" in section {self._modname} not found')
            sys.exit(1)
        if _seqname in self.cfg._sections.keys():
            self._seqname = _seqname
        else:
            self.Logger.critical(f'"sequence" argument {_seqname} in {self._modname} not found in section names of config_file')
            sys.exit(1)
        
        # Get super init kwargs for SequenceMod inheritance
        super_init_kwargs = {}
        for _k, _v in self.cfg[self._modname].items():
            if _k not in self.special_keys:
                super_init_kwargs.update({_k: _v})

        # Build Special Fields
        self.special_kwargs = {}
        for key in self.special_keys:
            self._parse_special_kwarg(key)
        
        # Get sequence of sections
        module_sequence = {}
        # Iterate across unit module sections
        for _smname, _smsect in self.cfg[self._seqname].items():
            # Check that the defining section exists
            if _smsect not in self.cfg._sections.keys():
                self.Logger.critical(f'submodule {_smsect} not defined in config_file')
                sys.exit(1)
            else:
                # Parse config_file section
                section_kwargs = self.parse_cfg_section(_smsect)
                # Build unit module object
                imod = self._build_unit_module(section_kwargs)
                # Add unit module object to sequence
                module_sequence.update({_smname: imod})
        # Update sequence in init key-word arguments
        super_init_kwargs.update({'sequence': module_sequence})
        # super init from SequenceMod
        super().__init__(**super_init_kwargs)

    ###########################################
    ### MODULE CONFIGURATION PARSING METHOD ###
    ###########################################            

    def _parse_special_kwarg(self, key):
        if key in self.special_keys:
            self.special_kwargs.update({key: self.special_methods[key]()})
        

    def parse_cfg_section(self, section):
        """Parse a config_file section's keys and values to create construction information for a PULSE unit module object initialization

        :param section: name of the section defining the unit module object
        :type section: str
        :return section_kwargs: key-word arguments for initializing a PULSE unit module class object
        :rtype section_kwargs: dict
        :return special_fields: parsed special_keys values 
        :rtype special_fields: dict
        """        
        section_kwargs = {}
        special_kwargs = {}
        for _k, _v in self.cfg[section].items():
            if _k in self.special_keys:
                # Parse special keys with provided methods (must be something )
                if self.special_methods[_k] is not None:
                    _val = self.special_kwargs[_k]
                    special_kwargs.update({_k: _val})
                # Parse special keys with provided methods (None --> direct copy)
                else:
                    special_kwargs.update({_k: _v})
            else:
                # Parse boolean with configparser methods
                if _v in ['True','False','yes','no']:
                    _val = self.cfg.getboolean(section, _k)
                # Parse everything else with eval statements
                else:
                    _val = eval(self.cfg.get(section, _k))
                section_kwargs.update({_k: _val})
        return section_kwargs, special_kwargs
    
    def _build_special_fields(self):
        special_fields = {}
        for _k, _v in self.special_methods.items():
            
            if _v is not None:
                special_fields.update({_k: _v()})
            else:
                special_fields.update({_k: _v})

    def _build_unit_module(self, section_kwargs, special_fields):
        # Import class to local scope
        parts = special_fields['class'].split(".")
        path = '.'.join(parts[:-1])
        clas = parts[-1]
        try:
            exec(f'from {path} import {clas}')
        except ImportError:
            self.Logger.critical(f'failed to import {special_fields['class']}')
            sys.exit(1)
        # Pass copies of parsed special fields to section kwargs
        for _k, _v in special_fields.items():
            if _k != 'class':
                section_kwargs.update({_k, _v})
        # Construct unit module object
        unit_mod_obj = eval(clas)(**section_kwargs)
        return unit_mod_obj 

        # # Placeholder for use of "module" in child classes
        # if 'module' in special_fields.keys():
        #     section_kwargs.update({'module': special_fields['module']})
        #     # self.Logger.critical(f'NotImplementedError: "module" special_key not supported for SequenceBuilderMod - try PulseMod_EW')
        #     # sys.exit(1)
        # # Placeholder for use of "sequence" for later development of recursive builds.
        # if 'sequence' in special_fields.keys():
        #     section_kwargs.update({'sequence': special_fields['sequence']})
        #     # self.Logger.critical(f'NotImplementedError: recursive building of sub-sequences not yet supported')
        #     # sys.exit(1)




   
    # def parse_config_section(self, section):
    #     submod_init_kwargs = {}
    #     submod_class = None
    #     for _k, _v in self.cfg[section].items():
    #         # Handle special case where class is passed
    #         if _k == 'class':
    #             submod_class = _v
    #         # Handle special case where module is passed
    #         elif _k == 'module':
    #             if 'module' in dir(self):
    #                 _val = self.module
    #             else:
    #                 self.Logger.critical('Special key `module` not supported by SequenceBuilderMod - see PulseMod_EW - exiting')
    #             sys.exit(1)
    #             # _val = self.module
    #         # Handle case where the parameter value is bool-like    
    #         elif _v in ['True', 'False', 'yes', 'no']:
    #             _val = self.cfg.getboolean(section, _k)
    #         # For everything else, use eval statements
    #         else:
    #             _val = eval(self.cfg.get(section, _k))
            
    #         if _k != 'class':
    #             submod_init_kwargs.update({_k: _val})
        
    #     if submod_class is None:
    #         return submod_init_kwargs
    #     else:
    #         return submod_class, submod_init_kwargs


    # # Get kwargs for SequenceMod super init
        # sinitkw = {}
        # for _k, _v in self.cfg[self._modname].items():
        #     if _k not in ['class','sequence']:
        #         sinitkw.update({_k: _v})

        # # Sequence submodules
        # sequence = {}
        # demerits = 0

        # # Iterate across submodule names and section names
        # for submod_name, submod_section in self.cfg[self._seqname].items():
        #     # Log if there are missing submodules
        #     if submod_section not in self.cfg._sections.keys():
        #         self.Logger.critical(f'submodule {submod_section} not defined in config_file. Will not compile!')
        #         demerits += 1
        #     # Construct if the submodule has a section
        #     else:
        #         # Parse the class name and __init__ kwargs
        #         submod_class, submod_init_kwargs = \
        #             self.parse_config_section(submod_section)
        #         # Run import to local scope
        #         parts = submod_class.split('.')
        #         path = '.'.join(parts[:-1])
        #         clas = parts[-1]
        #         try:
        #             exec(f'from {path} import {clas}')
        #         except ImportError:
        #             self.Logger.critical(f'failed to import {submod_class}')
        #             sys.exit(1)
        #         submod_object = eval(clas)(**submod_init_kwargs)
        #         # Attach object to sequence
        #         sequence.update({submod_name: submod_object})
        #         self.Logger.info(f'{submod_name} initialized')
        # # If there are any things that failed to compile, exit
        # if demerits > 0:
        #     sys.exit(1)
        # else:
        #     sinitkw.update({'sequence': sequence})