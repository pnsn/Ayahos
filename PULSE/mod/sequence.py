"""
:module: PULSE.mod.unit.package
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This contains the class definitions for a PULSE module that hosts and facilitates
    sequences of other :class:`~PULSE.mod` modules and their :meth:`~pulse` methods
    starting with a single input to the first module in the series and providing a
    single output from the last module in the series:

    output = modN.pulse(...pulse(mod1.pulse(mod0.pulse(input))))
"""
import sys, os, typing, copy
import numpy as np
import pandas as pd
from collections import deque
from obspy import UTCDateTime
from PULSE.mod.base import BaseMod

##################################
##################################
### SEQUENCE OBJECT DEFINITION ###
##################################
##################################

class Sequence(dict):
    """A dict-like object for containing chains of
    :class:`~PULSE.mod.base.BaseMod`-type objects
    that provides support methods for validating
    and visualizing these chains.

    Parameters
    ----------
    :param modules: PULSE Module or iterable group thereof, defaults to [].
        Supported iterables: list, set, tuple, dict
    :type modules: PULSE.mod.base.BaseMod or iterable, optional

    Attributes
    ----------
    :var keys: Names of BaseMod objects in this sequence
    :var values: BaseMod objects
    :var first: First BaseMod-type object in this sequence
    :var last: Last BaseMod-type object in this sequence
    :var names: list-formatted set of module names in this sequence
    :var current_stats: pandas.DataFrame-formatted summary of the
            **stats** attributes of the modules in this sequence
    """    
    def __init__(self, modules=[]) -> None:
        """Initialize a :class:`~.Sequence` object

        :param modules: PULSE Module or iterable group thereof, defaults to [].
            Supported iterables: list, set, tuple, dict
        :type modules: PULSE.mod.base.BaseMod or iterable, optional
        """        
        # Initialize as dictionary
        super().__init__()
        self.update(modules)
        self.validate()
    
    def update(self, modules):
        if isinstance(modules, BaseMod):
            super().update({modules.name: modules})
        elif isinstance(modules, (list, set, tuple)):
            for _e, mod in enumerate(modules):
                if not isinstance(mod, BaseMod):
                    raise TypeError(f'Element {_e} in "modules" is not type PULSE.mod.base.BaseMod')
                else:
                    self.update({mod.name: mod})
        elif isinstance(modules, dict):
            for _k, _v in modules.items():
                if isinstance(_v, BaseMod):
                    if _k == _v.name:
                        super().update({_k:_v})
                    else:
                        raise KeyError(f'Key "{_k}" does not match module name "{_v.name}"')
                else:
                    raise TypeError(f'Item keyed to {_k} is not type PULSE.mod.base.BaseMod')
        else:
            raise TypeError('Input "modules" must be type PULSE.mod.base.BaseMod or an iterable collection thereof')

    def validate(self):
        """Determine if this is a valid sequence:
        1) all values are :class:`~PULSE.mod.base.BaseMod` objects
        2) all sequential outputs & input_types are compatable
        :raises TypeError: not all values are type BaseMod
        :raises SyntaxError: not all output-input couplets are compliant
        """
        for _e, (name, mod) in enumerate(self.items()):
            if not isinstance(mod, BaseMod):
                raise TypeError(f'validate: Module {name} is type {type(mod)}. Must be type PULSE.mod.base.BaseMod')
            if _e < len(self) - 1:
                otype = type(mod.output)
                itype = self[self.names[_e+1]]._input_types
                if otype not in itype:
                    msg = f'validate: Module {name} output type is not compatable with '
                    msg += f'subsequent module {self.names[_e+1]} input type(s).'
                    raise SyntaxError(msg)
                
    def copy(self):
        """Create a deep copy of this :class:`~.Sequence`

        :return: 
            - **copy** (*PULSE.mod.sequence.Sequence*) - deep copy'd object
        """        
        return copy.deepcopy(self)
        
    def get_current_stats(self):
        """Create a :class:`~pandas.DataFrame` object that
        summarizes the current :class:`~PULSE.util.header.PulseStats`
        contents for all modules in this sequence, in order.

        :return: current status dataframe
        :rtype: pandas.DataFrame
        """        
        df = pd.DataFrame()
        for _v in self.values():
            new_df = pd.DataFrame([dict(_v.stats)])
            df = pd.concat([df, new_df], ignore_index=False)
        if len(df) > 0:
            df.index.name = 'name'
        return df
    
    current_stats = property(get_current_stats)

    def get_first(self):
        """Return a view of the first module in this :class:`~.Sequence` object

        :return:
         - **first** (*PULSE.mod.base.BaseMod*) - first module in sequence
        """ 
        return self[self.names[0]]
    
    first = property(get_first)

    def get_last(self):
        """Return a view of the last module in this :class:`~.Sequence` object

        :return:
         - **last** (*PULSE.mod.base.BaseMod*) - last module in sequence
        """        
        return self[self.names[-1]]
    
    last = property(get_last)

    def get_names(self):
        """return a list-formatted set of module names

        :return:
         - **names** (*list*) - list of module names
        """        
        return list(self.keys())
    
    names = property(get_names)

    def get_input_types(self):
        if len(self) > 0:
            return self.first._input_types
        else:
            return None
    
    _input_types = property(get_input_types)

    def get_output(self):
        if len(self) > 0:
            return self.last.output
        else:
            return None
    
    output = property(get_output)
    
    def __repr__(self):
        rstr = f'Sequence of {len(self)} PULSE Mods:\n{self.current_stats}'
        return rstr
    
    def __str__(self):
        return 'PULSE.mod.sequence.Sequence'

##################################
##################################
### SEQUENCE MODULE DEFINITION ###
##################################
##################################
    
class SeqMod(BaseMod):
    """
    A :mod:`~PULSE.mod` class for hosting a sequence of :mod:`~PULSE.mod` class objects and
    facilitating chained execution of their :meth:`~PULSE.mod.base.BaseMod.pulse` method calls
    where the output of each call serves as the input to the next.

    The pulse method operates as follows for a **SequenceMod.sequence**:

    sequencemod.sequence = {key0: <module0>, key1: <module1> , key2: <module2>}
    
    results in 

    SequenceMod.pulse(x) = module2.pulse(module1.pulse(module0.pulse(x)))

    This class also collects metadata from each :mod:`~PULSE.mod` class objects' **stats** attribute values
    in the SequenceMod's **metadata** attribute. The age of these metadata are calculated as a
    the most recent *endtime* value minus a given line's *endtime* value. Lines with ages older
    thank meta_max_age are deleted.

    :param modules: collection of modules that are executed in their provided order, defaults to {}
    :type modules: :class:`~PULSE.mod.base.BaseMod`, or list / dict thereof, optional
    :param meta_max_age: maximum relative age of metadata in seconds, defaults to 60.
    :type meta_max_age: float-like, optional
    :param max_pulse_size: maximum number of iterations to run the sequence of pulses, defaults to 1.
    :type max_pulse_size: int, optional
    :param maxlen: maximum length of the **output** of this SequenceMod if it has an empty **sequence**, defaults to None
    :type maxlen: None or int, optional
    :param name: string or integer to append to the end of this SequenceMod's __name__ attribute as "name", defaults to None
    :type name: None, int, str, optional.

    :var sequence: An order-sensitive dictionary of :mod:`~PULSE.mod` class objects to run in sequence
    :var output: If this SequenceMod is non-empty, this is an alias to the **output** of the last :mod:`~PULSE.mod` object
        in **sequence**. If this SequenceMod is empty, this is a :class:`~collections.deque` inherited from :class:`~PULSE.mod.base.BaseMod`
    :var metadata: This :class:`~pandas.dataframe.DataFrame` object holds metadata collected from the **stats** attribute
        of each :mod:`~PULSE.mod` object in **sequence** when :meth:`~PULSE.mod.sequence.SequenceMod.pulse` is called.
        If the SequenceMod is empty, then metadata are collected from the **SequenceMod.stats** attribute
    :var names: A list-formatted, order-sensitive set of **sequence** keys.
    :var Logger: the Logger instance for a given SequenceMod object, named with it's __name__ attribute
    :var _nonempty: This flag denotes if the **SequenceMod.sequence** is non-empty (True) or empty (False)
    :var _max_age: Maximum relative age in seconds for entries in **metadata** based on their *endtime* values
    """

    def __init__(
            self,
            modules=[BaseMod()],
            maxlen=60.,
            max_pulse_size=1,
            name=None):
        """Create a :class:`~PULSE.mod.sequence.SequenceMod` object

        :param sequence: collection of modules that are executed in their provided order, defaults to {}
        :type sequence: PULSE.mod.base.BaseMod, or list / dict thereof, optional
        :param meta_max_age: maximum relative age of metadata in seconds, defaults to 60.
        :type meta_max_age: float-like, optional
        :param max_pulse_size: maximum number of iterations to run the sequence of pulses, defaults to 1.
        :type max_pulse_size: int, optional
        :param maxlen: maximum length of the **output** of this SequenceMod if it has an empty **sequence**, defaults to None
        :type maxlen: None or int, optional
        :param name: string or integer to append to the end of this SequenceMod's __name__ attribute as "name", defaults to None
        :type name: None, int, str, optional.
        """
        # Inherit from BaseMod
        super().__init__(max_pulse_size=max_pulse_size, maxlen=None, name=name)
        # Initialize Sequence
        # if isinstance(modules, Sequence):
        #     self.sequence = modules
        # else:
        try:
            self.sequence = Sequence(modules)
        except (TypeError, SyntaxError, KeyError) as msg:
            self.Logger.critical(f'Sequence: {msg}. Exiting')
            sys.exit(os.EX_USAGE)

        # Overwrite output
        self.output = self.sequence.output

        # Overwrite _input_types
        self._input_types = self.sequence._input_types

        # Additional compatability check for maxlen
        if maxlen is None:
            self.Logger.warning('NoneType maxlen for SeqMod will result in all metadata being stored on RAM! Exiting')
            sys.exit(os.EX_USAGE)
        elif isinstance(maxlen, (int, float)):
            if maxlen > 0:
                self.stats.maxlen = float(maxlen)
            else:
                self.Logger.critical('maxlen must be a positive value. Exiting.')
                sys.exit(os.EX_DATAERR)
        else:
            self.Logger.critical('maxlen must be a positive float-like value. Exiting.')
            sys.exit(os.EX_DATAERR)

        # Create dataframe holder for pulse metadata
        self.metadata = pd.DataFrame()


    def __repr__(self, full=False):
        rstr = self.stats.__str__()
        if full:
            rstr += f'\n{self.metadata.__str__()}'
        return rstr

    ###################
    ## PULSE METHODS ##
    ###################
            
    def pulse(self, input):
        """Execute a chain of :meth:`~PULSE.mod.base.BaseMod.pulse` 
        methods for the :class:`~PULSE.mod.base.BaseMod` objects contained
        in this :class:`~.SeqMod`'s **sequence** attribute and return
        a view of the output of the last module in the sequence

        POLYMORPHIC: last update with :class:`~.Seqmod`

        :param input: input object for the first module in this sequence
        :type input: object
        :raises AttributeError: If pulse is executed on an empty SeqMod
        :return:
         - **output** (*object*) -- output of the last module in this sequence
            after execution of **max_pulse_size** chained pulses.
        """        
        if len(self.sequence) == 0:
            self.Logger.critical('Cannot run "pulse" with an empty SeqMod. Exiting')
            sys.exit(os.EX_USAGE)
        else:
            super().pulse(input)
        return self.output
        
    def check_input(self, input):
        """Use the :meth:`~.check_input` method of the first
        :class:`~PULSE.mod.base.BaseMod`-type object in this SeqMod's
        **sequence** to make sure the provided input to :meth:`~.SeqMod.pulse`
        conforms to that first module's _input_types.

        POLYMORPHIC: last update with :class:`~.SeqMod`

        :param input: input object to :meth:`~.SeqMod.pulse`
        :type input: depends on first element in 
        """
        self.sequence.first.check_input(input)
    
    def measure_input(self, input) -> int:
        """Use the :meth:`~.measure_input` method of the first
        module in this SeqMod's **sequence** to measure the input
        provided to :meth:`~.SeqMod.pulse`

        POLYMORPHIC: last update with :class:`~.SeqMod`

        :param input: input object
        :type input: object
        :return:
            - **measure** (*int*) - representative measure of the input
        """        
        return self.sequence.first.measure_input(input)
    
    def measure_output(self) -> int:
        """Return the measure of the output attribute of
        the last module in this SeqMod's **sequence**

        POLYMORPHIC: last update with :class:`~.SeqMod`

        :return: 
         - **measure** (*int*) - representative measure of the output
        """        
        return self.sequence.last.measure_output()
    
    def get_unit_input(self, input: object) -> object:
        """Pass the input object provided to :meth:`~.SeqMod.pulse` to
        the :meth:`~PULSE.mod.base.BaseMod.pulse` method of the first

        POLYMORPHIC: last update with :class:`~.SeqMod`
        
        :param intput: _description_
        :type intput: object
        :return: _description_
        :rtype: object
        """        
        unit_input = input
        return unit_input
    
    def run_unit_process(self, unit_input) -> list:
        for _e, mod in enumerate(self.sequence.values()):
            if _e == 0:
                y = mod.pulse(unit_input)
            else:
                y = mod.pulse(y)
        unit_output = self.sequence.current_stats
        return unit_output

    def put_unit_output(self, unit_output):
        """Store the **current_stats** summary from a single
        sequence of pulses in the **metadata** attribute
        of this :class:`~.SeqMod` and flush out metadata
        that are older than **maxlen** seconds relative to
        the most curent endtime in **metadata**.

        :param unit_output: _description_
        :type unit_output: _type_
        """        
        # Concatenate 
        self.metadata = pd.concat([self.metadata, unit_output],
                                  ignore_index=True)
        # Get the maximum endtime for all pulses
        ld = self.metadata.endtime.max()
        # If the maximum endtime is a UTCDateTime object
        if isinstance(ld, UTCDateTime):
            # Keep items that are not-nan starttime & have starttimes within maxlen sec
            # of the most recent endtime
            self.metadata = self.metadata[(self.metadata.starttime.notna()) &\
                                          (self.metadata.starttime >= ld - self.stats.maxlen)]
        else:
            return
        
    


#     def put_unit_output(self, unit_output: list) -> None:
        


    # def check_input(self, input):
    #     self.get_first_element

    # def pulse_startup(self, input):


        
    # def get_ending_output(self):
    #     lastkey = list(self.sequence.keys())[-1]
    #     return self.sequence[lastkey].output
    
    # output = property(get_ending_output)

        # # Alias the output of the last module in sequence to self.output (inherited from BaseMod)
        # if len(self.sequence) > 0:
        #     self._nonempty = True
        #     self.output = self.sequence[self.names[-1]]
        # else:
        #     self._nonempty = False
        #     self.Logger.info(f'Empty {self.__name__()}.sequence - defaulting output to BaseMod.output (collections.deque)')


    # def __setattr__(self, key, value):
    #     if key == 'sequence':
    #         #  TODO: Have setting sequence automatically re-alias output and update stats
    #         if isinstance(value, dict):
    #             if all(isinstance(_e, BaseMod) for _e in value.values()):
    #                 vkeys = list(value.keys())
    #                 self.output = self.sequence[vkeys[-1]]
    #     else:
    #         pass
    #     return super(SeqMod, self).__setattr__(key, value)


    # #################################
    # # POLYMORPHIC METHODS FOR PULSE #
    # #################################
    
    # def pulse(self, input):
    #     """WRAPPER METHOD

    #     This wraps :meth:`~PULSE.mod.base.BaseMod.pulse` to provide class-specific
    #     documentation for :class:`~PULSE.mod.sequence.SequenceMod`

    #     If **SequenceMod.sequence** is non-empty:
    #     This version of pulse executes a single call of pulse for each element of 
    #     the sequence, feeding the first element the input provided here, and each
    #     subsequent element receives the output of the prior element's pulse call
    #     as an input. The output returned by this method is a view of the **output**
    #     attribute of the last element in **sequence**.

    #     If **SequenceMod.sequence** is empty:
    #     The SequenceMod **pulse** method adopts the behavior of :meth:`~PULSE.mod.base.BaseMod.pulse`

    #     :param input: input to :meth:`~pulse` for the first :mod:`~PULSE.mod` module object in **sequence**,
    #         or a collection of objects to pass through an empty **sequence** following :class:`~PULSE.mod.base.BaseMod.pulse` behavior
    #     :type input: object
    #     :return: output, either a view of the **output** of the last :class:`~PULSE.mod` module object in **sequence**,
    #         or the **output** attribute of this SequenceMod if **sequence** is empty
    #     :rtype: object, or deque
    #     """        
    #     output = super().pulse(input)
    #     return output


    # def check_input(self, input):
    #     """POLYMORPHIC METHOD

    #     Last updated with :class:`~PULSE.mod.sequence.SequenceMod`

    #     Aliases the :meth:`~check_input` method of the first module
    #     in this SequenceMod's **sequence** attribute.

    #     :param input: see description for aliased method
    #     :type input: see description for aliased method
    #     :return input_size: size of **input**
    #     :rtype input_size: int
    #     """        
    #     # Use the check_input method from the first module in the sequence
    #     if self._nonempty:
    #         input_size = self.sequence[self.names[0]].check_input(input)
    #     else:
    #         input_size = super().check_input(input)
    #     return input_size
    
    # def measure_output(self):
    #     """POLYMORPHIC METHOD

    #     Last updated with :class:`~PULSE.mod.sequence.SequenceMod`

    #     Aliases the :meth:`~check_input` method of the first module
    #     in this SequenceMod's **sequence** attribute.

    #     :param input: see description for aliased method
    #     :type input: see description for aliased method
    #     :return: 
    #      - **input_size** (*int*) -- size of the input
    #     """
    #     if len(self.sequence) > 0:
    #         output_size = self.sequence[self.names[-1]].measure_output()
    #     else:
    #         output_size = super().measure_output()
    #     return output_size
    
    # def get_unit_input(self, input):
    #     """POLYMORPHIC METHOD

    #     Last updated with :class:`~PULSE.mod.sequence.SequenceMod`

    #     If this SequenceMod is non-empty, pass **input** to **unit_input**

    #     If this SequenceMod is empty, use behaviors of :meth:`~PULSE.mod.base.BaseMod.get_unit_input`

    #     :param input: pulse input object
    #     :type input: object
    #     :return:
    #      - **unit_input** (*object*) - unit process input object
    #     """        
    #     if self._nonempty:
    #         unit_input = input
    #     else:
    #         # Use BaseMod behavior
    #         unit_input = super().get_unit_input(input)
    #     return unit_input

    # def run_unit_process(self, unit_input):
    #     """POLYMORPHIC METHOD

    #     Last updated with :class:`~PULSE.mod.sequence.SequenceMod`

    #     Execute a chained sequence of :meth:`~pulse` calls in the order
    #     set in **sequence**. If **sequence** is empty, this method uses
    #     the behaviors from :class:`~PULSE.mod.base.BaseMod`

    #     :param unit_input: input object for the first element in **sequence**,
    #         or a double-ended queue of objects if **sequence** is empty
    #     :type unit_input: object, or collections.deque
    #     :return:
    #      - **unit_output** (*list*) - copies of :class:`~PULSE.mod.base.PulseStats` metadata objects generated
    #      during this call of :meth:`~PULSE.mod.sequence.SequenceMod.pulse`.
    #     """        
    #     unit_output = []
    #     if self._nonempty:
    #         for position, module in enumerate(self.sequence.values()):
    #             if position == 0:
    #                 y = module.pulse(unit_input)
    #             else:
    #                 y = module.pulse(y)
    #             unit_output.append(module.stats.copy())
    #     else:
    #         unit_output = super().run_unit_process(unit_input)
    #     return unit_output
    
    # def store_unit_output(self, unit_output):
    #     """POLYMORPHIC METHOD

    #     Last updated with :class:`~PULSE.mod.sequence.SequenceMod`

    #     Appends collected copies of newly-generated **stats** metadata from
    #     each module in **sequence** (if non-empty) to the **metadata** DataFrame
    #     and trims off outdated data (relative ages older than max_meta_age)

    #     :param unit_output: collection of :class:`~PULSE.mod.base.PulseStats` objects
    #     :type unit_output: list
    #     """        
    #     # Ingest new metadata into self.metadata
    #     if self._nonempty:
    #         for stats in unit_output:
    #             self.metadata = pd.concat([self.metadata,
    #                                     pd.DataFrame(stats)], 
    #                                     axis=0, ignore_index=True)
    #         max_endtime = self.metadata.endtime.max()
    #         # Trim off outdated data
    #         self.metadata = self.metadata[(self.metadata.endtime >= max_endtime - self._memory)].sort_values('endtime')

    #     else:
    #         super().store_unit_output(unit_output)

    # ##############################
    # # SEQUENCE MODIFYING METHODS #
    # ##############################

    # def update(self, new_dict):
    #     """
    #     Apply update to sequence using the 
    #     dict.update(new_dict) builtin_function_or_method
    #     and then update relevant attributes of this SequenceMod

    #     This will update existing keyed entries and append new
    #     at the end of self.sequence (same behavior as dict.update)

    #     :param new_dict: dictionary of {name:*Mod} pairs to  
    #     :type new_dict: dict
    #     :updates: 
    #         - **self.sequence** - updates sequence as specified
    #         - **self.names** - updates the list of keys in sequence
    #         - **self.output** - re-aliases to the output attribute of the last module in sequence

    #     """
    #     # Safety catches identical to those in __init__
    #     if not isinstance(new_dict, dict):
    #         raise TypeError('new_dict must be type dict')
    #     elif not all(isinstance(_m, BaseMod) for _m in new_dict.values()):
    #         raise TypeError('new_dict can only have values of type PULSE.mod._base.BaseMod')
    #     else:
    #         pass
    #     # Run updates on sequence
    #     self.sequence.update(new_dict)
    #     # Update names attribute
    #     self.names = list(self.sequence.keys())
    #     if len(self.sequence) > 0:
    #         self._nonempty = True
    #         self.output = self.sequence[self.names[-1]].output
    #     else:
    #         self._nonempty = False
    #         self.output = deque(maxlen=self.maxlen)
        
    # def remove(self, key):
    #     """
    #     Convenience wrapper of the dict.pop() method
    #     to remove an element from self.sequence and
    #     associated attributes

    #     :: INPUT ::
    #     :param key: valid key in self.sequence.keys()
    #     :type key: unit_inputect

    #     :: RETURN ::
    #     :return popped_item: popped (key, value) pair
    #     :rtype popped_item: tuple
    #     """
    #     if key not in self.sequence.keys():
    #         raise KeyError(f'key {key} is not in self.sequence.keys()')
    #     # Remove key/val combination from dict
    #     val = self.sequence.pop(key)
    #     # Update names attribute
    #     self.names = list(self.sequence.keys())
    #     if len(self.sequence) > 0:
    #         self._nonempty = True
    #         self.output = self.sequence[self.names[-1]].output
    #     else:
    #         self._nonempty = False
    #         self.output = deque(maxlen=self.maxlen)
        
    #     return (key, val)

    # def reorder(self, reorder_list):
    #     """
    #     Reorder the current contents of sequence using either
    #     an ordered list of sequence

    #     :: INPUT ::
    #     :param reorder_list: unique list of keys from self.module
    #     :type reorder_list: list of Wyrm-likes
    #     """
    #     # Ensure reorder_list is a list
    #     if not isinstance(reorder_list, list):
    #         raise TypeError('reorder_list must be type list')

    #     # Ensure reorder_list is a unique set
    #     tmp_in = []
    #     for _e in reorder_list:
    #         if _e not in tmp_in:
    #             tmp_in.append(_e)
    #     if tmp_in != reorder_list:
    #         raise ValueError('reorder_list has repeat entries - all entries must be unique')

    #     # Conduct reordering if checks are passed
    #     # Handle input (re)ordered sequence key list
    #     if all(_e in self.sequence.keys() for _e in reorder_list):
    #         tmp = {_e: self.sequence[_e] for _e in reorder_list}
    #     # Handle input (re)ordered index list
    #     elif all(_e in np.arange(0, len(reorder_list)) for _e in reorder_list):
    #         tmp_keys = list(self.sequence.keys())
    #         tmp = {_k: self.sequence[_k] for _k in tmp_keys}

    #     # Run updates
    #     self.sequence = tmp
    #     self.names = list(tmp.keys())
    #     self.output = self.sequence[self.names[-1]].output

    # ########################
    # # DUNDER/MAGIC METHODS #
    # ########################
        
    # def __repr__(self, extended=False):
    #     """Provide a user-friendly summary of the contents of this SequenceMod
    #     :: INPUT ::
    #     :param extended: show full __repr__ output of component Wyrms? , defaults to False
    #     :type extended: bool, optional

    #     :: OUTPUT ::
    #     :return rstr: string representation of this Wyrm's contents
    #     :rtype rstr: str
    #     """
    #     rstr = f'{super().__repr__()}\n'
    #     rstr = f"(wait: {self.wait_sec} sec)\n"
    #     for _i, (_k, _v) in enumerate(self.sequence.items()):
    #         # Provide index number
    #         rstr += f'({_i:<2}) '
    #         # Provide labeling of order
    #         if _i == 0:
    #             rstr += "(head) "
    #         elif _i == len(self.sequence) - 1:
    #             rstr += "(tail) "
    #         else:
    #             rstr += "  ||   "
    #         rstr += f"{_k} | "
    #         if extended:
    #             rstr += f'{_v}\n'
    #         else:
    #             rstr += f'{_v.__name__()}\n'
    #     return rstr
    
    # #############################
    # # PULSE POLYMORPHIC METHODS #
    # #############################

    # def _should_this_iteration_run(self, input, input_measure, iterno):
    #     """
    #     POLYMORPHIC

    #     Last updated with :class:`~PULSE.mod.bundle.SequenceMod`

    #     always return status = True
    #     Execute max_pulse_size iterations regardless of internal processes

    #     :param input: Unused
    #     :type input: any
    #     :param iterno: Unused
    #     :type iterno: any
    #     :return status: continue iteration - always True
    #     :rtype: bool
    #     """        
    #     status = True
    #     return status
    
    # def _unit_input_from_input(self, input):
    #     """
    #     POLYMORPHIC
    #     Last updated with :class: `~PULSE.mod.bundle.SequenceMod` 

    #     Pass the standard input directly to the first module in sequence

    #     :param input: standard input unit_inputect
    #     :type input: varies, depending on input expected by first module in sequence
    #     :return unit_input: view of standard input unit_inputect
    #     :rtype unit_input: varies
    #     """        
    #     unit_input = input
    #     return unit_input

    # def _unit_process(self, unit_input):
    #     """
    #     POLYMORPHIC
    #     Last updated with :class: `~PULSE.mod.bundle.SequenceMod`

    #     Chain pulse() methods of modules in sequence
    #     passing unit_input as the input to the first 

    #         module2.pulse(module1.pulse(module0.pulse(unit_input)))

    #     In this example output is captured by module2's pulse and
    #     SequenceMod.output is a view of module2.output

    #     :param unit_input: standard input for the pulse method of the first module in sequence
    #     :type unit_input: varies
    #     :return unit_output: sum of nproc output by each module in moduledict for this iteration
    #     :rtype: int
    #     """ 
    #     for j_, module_ in enumerate(self.sequence.values()):
    #         if j_ == 0:
    #             y = module_.pulse(unit_input)
    #         else:
    #             y = module_.pulse(y)
    #     # TODO: Placeholder unconditional True - eventually want to pass a True/False
    #     # up the line to say if there should be early stopping
    #     unit_output = None
    #     return unit_output

    # def _capture_unit_output(self, unit_output):
    #     """
    #     POLYMORPHIC
    #     Last updated by :class: `~PULSE.mod.bundle.SequenceMod`

    #     Termination point - output capture is handled by the last
    #     Wyrm-Type object in sequence and aliased to this SequenceMod's
    #     output attribute.

    #     :param unit_output: sum of processes executed by the _unit_process call
    #     :type unit_output: int
    #     :return: None
    #     :rtype: None
    #     """
    #     return None   

    # def _should_next_iteration_run(self, unit_output):
    #     """
    #     POLYMORPHIC
    #     Last updated by :class: `~PULSE.mod.bundle.SequenceMod`

    #     Signal early stopping (status = False) if unit_output == 0

    #     :param unit_output: number of processes executed by _unit_process
    #         Sum of pulse 
    #     :type unit_output: int
    #     :return status: should the next iteration be run?
    #     :rtype status: bool
    #     """
    #     if unit_output == 0:
    #         status = False
    #     else:
    #         status = True
    #     return status

    # def _update_report(self):
    #     """
    #     POLYMORPHIC
    #     Last updated with :class:`~PULSE.mod.bundle.SequenceMod`

    #     Get the mean value line for each module and add information
    #     on the pulserate, number of logged pulses, and memory period
    #     for each module.
    #     """
    #     report_dict = {}
    #     for _n, _w in self.sequence.items():
    #         _r = _w.report
    #         _p = _w._pulse_rate
    #         for _m in self.report_fields:
    #             line = list(_r.loc[_m].values)
    #             line.append(_p)
    #             line.append(len(_w._metadata))
    #             line.append(_w.meta_memory)
    #             report_dict.update({f'{_n}({_m})': line})
    #     keys = self._keys_meta + ['p_rate','n_pulse','memory_sec']
    #     self.report = pd.DataFrame(report_dict, index=keys).T
    #     self.report.index.name = 'submod (stat)'




# class SequenceBuilderMod(SequenceMod):
#     """
#     Defines a self-building :class:`~PULSE.mod.sequence.SequenceMod` child-class using regularly formatted
#     inputs from a configuration file (*.ini) using the ConfigParser Python library

#     :param config_file: file name and path for the desired configuration file, which must contain the sections decribed below
#     :type config_file: str
#     :param starting_section: unique section name for this sequence, defaults to 'Sequence_Module'

#     config_file.ini format
#     '''
#     [Sequence_Module]   # <- Re-assign to match with **starting_section** in your .ini file to allow for multiple calls
#     class: PULSE.mod.sequence.SequenceBuilderMod
#     sequence: Sequence_0
#     max_pulse_size: 1000
#     ...

#     [Sequence_0]
#     mod0_name: mod0_section_name
#     mod1_name: mod1_section_name
#     mod2_name: mod1_section_name  # <- not a typo, section names can be called multiple times!
#     ...
#     '''



#     """
#     # Define special section keys (allows overwrite outside of __init__ for things like PulseMod_EW)
#     special_keys=['sequence','class']
#     def __init__(self, config_file, starting_section='Sequence_Module', additional_special_keys=[]):
#         """
#         Initialize a :class:`~PULSE.mod.sequence.SequenceBuilderMod` object
#         """

#         # Initialize config parser
#         self.cfg = configparser.ConfigParser(
#             interpolation=configparser.ExtendedInterpolation()
#         )
#         # Read configuration file
#         self.cfg.read(config_file)

#         # Compatablity checks with starting_section & cross-check with cfg
#         if isinstance(starting_section, str):
#             if starting_section not in self.cfg._sections.keys():
#                 self.Logger.critical('KeyError: starting_section not in config_file')
#                 sys.exit(1)
#             else:
#                 self._modname = starting_section
#         else:
#             self.Logger.critical('TypeError: starting_section must be type str')
#             sys.exit(1)
    
#         if isinstance(additional_special_keys, list):
#             if all(isinstance(e, str) for e in additional_special_keys):
#                 self.special_keys += additional_special_keys
#             else:
#                 self.Logger.critical('TypeError: special_keys elements must be all type str')
#                 sys.exit(1)
#         else:
#             self.Logger.critical('TypeError: special_keys must be a list')
#             sys.exit(1)

#         # Populate specials
#         for key in self.special_keys:
#             self.populate_specials(key)

#         # Get sequence name
#         try:
#             _seqname = self.cfg.get(self._modname, 'sequence')
#         except:
#             self.Logger.critical(f'sequence-definining entry for "sequence" in section {self._modname} not found')
#             sys.exit(1)
#         if _seqname in self.cfg._sections.keys():
#             self._seqname = _seqname
#         else:
#             self.Logger.critical(f'"sequence" argument {_seqname} in {self._modname} not found in section names of config_file')
#             sys.exit(1)
        
#         # Get super init kwargs for SequenceMod inheritance
#         super_init_kwargs = {}
#         for _k, _v in self.cfg[self._modname].items():
#             if _k not in self.special_keys:
#                 super_init_kwargs.update({_k: _v})
        
#         # Get sequence of sections
#         module_sequence = {}
#         # Iterate across unit module sections
#         for _smname, _smsect in self.cfg[self._seqname].items():
#             # Check that the defining section exists
#             if _smsect not in self.cfg._sections.keys():
#                 self.Logger.critical(f'submodule {_smsect} not defined in config_file')
#                 sys.exit(1)
#             else:
#                 # Parse config_file section
#                 section_kwargs = self.parse_cfg_section(_smsect)
#                 # Build unit module object
#                 imod = self._build_unit_module(section_kwargs)
#                 # Add unit module object to sequence
#                 module_sequence.update({_smname: imod})
#         # Update sequence in init key-word arguments
#         super_init_kwargs.update({'sequence': module_sequence})
#         # super init from SequenceMod
#         super().__init__(**super_init_kwargs)

#     ###########################################
#     ### MODULE CONFIGURATION PARSING METHOD ###
#     ###########################################            

#     def parse_cfg_section(self, section):
#         """Parse a config_file section's keys and values to create construction information for a PULSE unit module object initialization

#         :param section: name of the section defining the unit module object
#         :type section: str
#         :return section_kwargs: key-word arguments for initializing a PULSE unit module class object
#         :rtype section_kwargs: dict
#         :return special_fields: parsed special_keys values 
#         :rtype special_fields: dict
#         """        
#         section_kwargs = {}
#         special_kwargs = {}
#         for _k, _v in self.cfg[section].items():
#             if _k in self.special_keys:
#                 # Parse special keys with provided methods (must be something )
#                 if self.special_methods[_k] is not None:
#                     _val = self.special_kwargs[_k]
#                     special_kwargs.update({_k: _val})
#                 # Parse special keys with provided methods (None --> direct copy)
#                 else:
#                     special_kwargs.update({_k: _v})
#             else:
#                 # Parse boolean with configparser methods
#                 if _v in ['True','False','yes','no']:
#                     _val = self.cfg.getboolean(section, _k)
#                 # Parse everything else with eval statements
#                 else:
#                     _val = eval(self.cfg.get(section, _k))
#                 section_kwargs.update({_k: _val})
#         return section_kwargs, special_kwargs
    
#     def _build_special_fields(self):
#         special_fields = {}
#         for _k, _v in self.special_methods.items():
            
#             if _v is not None:
#                 special_fields.update({_k: _v()})
#             else:
#                 special_fields.update({_k: _v})

#     def _build_unit_module(self, section_kwargs, special_fields):
#         # Import class to local scope
#         parts = special_fields['class'].split(".")
#         path = '.'.join(parts[:-1])
#         clas = parts[-1]
#         try:
#             exec(f'from {path} import {clas}')
#         except ImportError:
#             self.Logger.critical(f'failed to import {special_fields['class']}')
#             sys.exit(1)
#         # Pass copies of parsed special fields to section kwargs
#         for _k, _v in special_fields.items():
#             if _k != 'class':
#                 section_kwargs.update({_k, _v})
#         # Construct unit module object
#         unit_mod_obj = eval(clas)(**section_kwargs)
#         return unit_mod_obj 

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