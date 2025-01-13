
import typing, copy

import pandas as pd

from PULSE.mod.base import BaseMod

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
    def __init__(self, modules={}):
        """Initialize a :class:`~.Sequence` object

        :param modules: PULSE Module or iterable group thereof, defaults to {}.
            Supported iterables: list, set, tuple, dict
        :type modules: PULSE.mod.base.BaseMod or iterable, optional
        :returns: **sequence** (*PULSE.mod.sequence.Sequence*) - sequence of BaseMod objects
        """        
        # Initialize as dictionary
        super().__init__()
        self.update(modules)
        self.validate()
    
    
    def update(self, modules):
        """Update this sequence with one or more :class:`~.BaseMod`-type objects
        while preserving the order of the input **modules**

        Note: :class:`~dict` type inputs must have keys matching the **name** attribute
        of the the associated :class:`~.BaseMod` value.

        :param modules: a single :class:`~.BaseMod` object or iterable collection thereof
        :type modules: :class:`~.BaseMod` or list, dict, set, or tuple thereof
        """        
        # If modules is basemod
        if isinstance(modules, BaseMod):
            super().update({modules.name: modules})

        # If modules is dict
        elif isinstance(modules, dict):
            for _k, _v in modules.items():
                if isinstance(_v, BaseMod):
                    if _k == _v.name:
                        super().update({_k:_v})
                    else:
                        raise KeyError(f'Key "{_k}" does not match module name "{_v.name}"')
                else:
                    raise TypeError(f'Item keyed to {_k} is not type PULSE.mod.base.BaseMod')
                
        # otherwise if modules is iterable
        elif hasattr(modules, '__iter__'):
            bad_elements = []
            for _e, mod in enumerate(modules):
                if not isinstance(mod, BaseMod):
                    bad_elements.append(_e)

                else:
                    self.update({mod.name: mod})
            if len(bad_elements) > 0:
                # breakpoint()
                raise TypeError(f'Element(s) {bad_elements} in "modules" is not type PULSE.mod.base.BaseMod')
        
        # Catch everything else
        else:
            raise TypeError('Input "modules" must be type PULSE.mod.base.BaseMod or an iterable thereof')

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
        """Create a deep copy of this :class:`~.Sequence` object

        :return: **newmod** (*PULSE.mod.sequence.Sequence*) - deepcopy of this sequence
        """        
        return copy.deepcopy(self)
        
    def get_current_stats(self) -> pd.DataFrame:
        """Create a :class:`~pandas.DataFrame` object that
        summarizes the current :class:`~PULSE.util.header.PulseStats`
        contents for all modules in this sequence, in order.

        :returns: **df** (*pandas.DataFrame*) -- pulse stats summary
            for the most recent pulse sequence triggered
        """        
        df = pd.DataFrame()
        for _v in self.values():
            new_df = pd.DataFrame([dict(_v.stats)])
            df = pd.concat([df, new_df], ignore_index=False)
        if len(df) > 0:
            df.index.name = 'name'
        return df
    
    current_stats = property(get_current_stats)

    def get_first(self) -> BaseMod:
        """Return a view of the first module in this :class:`~.Sequence` object

        :returns: **first** (*PULSE.mod.base.BaseMod*) -- first module in sequence
        """ 
        return self[self.names[0]]
    
    first = property(get_first)

    def get_last(self):
        """Return a view of the last module in this :class:`~.Sequence` object

        :return: **last** (*PULSE.mod.base.BaseMod*) - last module in sequence
        """        
        return self[self.names[-1]]
    
    last = property(get_last)

    def get_names(self):
        """return a list-formatted set of module names

        :returns: **names** (*list*) - list of module names
        """        
        return list(self.keys())
    
    names = property(get_names)

    def get_input_types(self) -> typing.Union[list, None]:
        """Get the **_input_types** attribute for the first
        element in this :class:`~.Sequence`

        :returns: **_input_types** (*list* or **NoneType**) -- list of input type(s)
            accepted by the first element in this Sequence
        """        
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
        return 'PULSE.seq.sequence.Sequence'