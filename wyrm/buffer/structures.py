from collections import deque
from wyrm.buffer.trace import BuffTrace
import wyrm.util.input_compatability_checks as icc
import inspect
import fnmatch

class TieredBuffer(dict):
    """
    Standardized structure and search methods for an 2-tiered dictionary with 
    the structure
        tier0 - a primary key-set, generally an instrument code string
            tier1 - a secondary key-set, generally an instrument component code or model name
                buff - some data holding class object

    The rationale of this structure is to accelerate searching by limiting the number
    of keys that are searched in a given list of keys, and leveraging the query speed
    of hash-tables underlying the Python `dict` class. Earlier tests showed that this
    structured system was significantly faster at retrieving data from large sets of
    objects compared to an unstructured list-like holder.

    NOTE
    This version of TieredBuffer limits the structure to 10 tiers maximum to have
    kwarg values for most methods conform to some form of the string '*T?'.
    
    If you need more, your probably want to and know how to code your own
    structure and can base it on the code here.

    """
    def __init__(self, buff_class=BuffTrace, **buff_init_kwargs):
        # Inherit dict properties
        super().__init__()
        # Compatability check for buff_class input
        if not isinstance(buff_class, type):
            raise TypeError('buff_class must be a class-defining type')
        # Ensure that buff_class has an append method
        elif 'append' not in dir(buff_class):
            raise TypeError('buff_class does not have an "append" method - incompatable')
        else:
            self.buff_class = buff_class

        # Compatability check for buff_init_kwargs input
        bcargs = inspect.getfullargspec(self.buff_class).args[1:]
        bcdefs = inspect.getfullargspec(self.buff_class).defaults
        bdefaults = dict(zip(bcargs, bcdefs))
        emsg = ''
        for _k in buff_init_kwargs.keys():
            if _k not in bcargs:
                if emsg == '':
                    emsg += f'The following kwargs are not compabable with {buff_class}:'
                emsg += f'\n{_k}'
        # If emsg has any information populated in it, kick error with emsg report
        if emsg != '':
            raise TypeError(emsg)
        # Otherwise clear init_kwargs
        else:
            for _k in bcargs:
                if _k not in buff_init_kwargs and _k != 'self':
                    buff_init_kwargs.update({_k: bdefaults[_k]})
            self.bkwargs = buff_init_kwargs

    def fetch_slice(self, TK0='*', TK1='*'):
        """
        Create a copy of the structure sliced from this TieredBuffer using fnmatch.filter
        compatable strings for TierKey0 and TierKey1

        :: INPUTS ::
        :param TK0: [str] Tier Key 0 - input string for fnmatch.filter(self.keys(), TK0)
        :param TK1: [str] Tier Key 1 - input string for fnmatch.filter(self[TK0-matches].keys(), TK1)

        :: OUTPUT ::
        :return view: [wyrm.buffer.structures.TieredBuffer] copy of matching elements of
                    the source TieredBuffer
        """
        view = TieredBuffer(buff_class=self.buff_class, buff_init_kwargs=self.buff_init_kwargs)
        for _k0 in fnmatch.filter(self.keys(), TK0):
            for _k1 in fnmatch.filter(self[_k0].keys(), TK1):
                view.update({_k0: {_k1: self[_k0][_k1]}})
        return view
    

    def add_branch(self, TK0, TK1=None):
        if TK0 not in self.keys():
            if TK1 is not None:
                self.update({TK0: {TK1: self.buff_class(**self.bkwargs)}})
            else:
                self.update({TK0: {}})

        elif TK1 is not None:
            if TK1 not in self[TK0].keys():
                self[TK0].update({TK1: self.buff_class(**self.bkwargs)})
            else:
                print(f'Warning - buffer [{TK0}][{TK1}] already initialized')
        return self

    def apply_buffer_method(self, method=__init__, TK0='*', TK1='*', *args, **kwargs):
        if method not in dir(self.buff_class):
            raise SyntaxError(f'specified "method" {method} is not compatable with buffer type {self.buff_class}')

        for _k0 in fnmatch.filter(self.keys(), TK0):
            for _k1 in fnmatch.filter(self[_k0], TK1):
                self[_k0][_k1].method(*args, **kwargs)
        return self

    def append(self, obj, TK0_eval_str='.id[:-1]', TK1_eval_str='.id[-1]', **options):
        """
        Append a single object to a single buffer object in this TieredBuffer using
        the buffer_object.append() method
        """
        TK0 = eval(f'obj{TK0_eval_str}')
        TK1 = eval(f'obj{TK1_eval_str}')
        self.add_branch(TK0, TK1=TK1)
        self[TK0][TK1].append(obj, **options)
        return self
    

    def __repr__(self, extended=False)


    def __bufftrace_rline__(self):
        if not isinstance(self.buff_class, BuffTrace):
