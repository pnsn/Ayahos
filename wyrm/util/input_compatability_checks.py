import numpy as np
from collections import deque
import seisbench.models as sbm

def bounded_intlike(x, name="x", minimum=1, maximum=1e9):
    """
    
    """
    if not isinstance(x, (int, float)):
        raise TypeError(f"{name} must be int-like")
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite")
    if minimum <= int(x) <= maximum:
        return int(x)
    else:
        raise ValueError(f"{name} must be in the bounds [{minimum}, {maximum}]")


def bounded_floatlike(x, name="x", minimum=1., maximum=1e9):
    if not isinstance(x, (int, float)):
        raise TypeError(f"{name} must be float-like")
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite")
    if minimum <= float(x) <= maximum:
        return float(x)
    else:
        raise ValueError(f"{name} must be in the bounds [{minimum}, {maximum}]")


def iterable_characters(x, name='x', listlike_types=(list, deque, np.array)):
    if isinstance(x, listlike_types):
        bool_list = []
        for _x in x:
            bool_list.append(isinstance(_x, str))
            bool_list.append(len(_x) == 1)
        if all(bool_list):
            return x
        else:
            raise TypeError(f'list-like {name} must only comprise single-character strings')
    elif isinstance(x, str):
        return [_x for _x in x]
    else:
        raise TypeError(f'{name} must be type str or a list-like object of single characters')


def none_str(x, name='x', warn=False):
    if not isinstance(x, (type(None), str)):
        raise TypeError(f"{name} must be type str or None")
    elif isinstance(x, type(None)):
        if warn:
            print(f'{name} = None is strictly a placeholder')
            return x
        else:
            return x
    else:
        return x
    
def iscamelcase_str(x):
    """
    Quick check if input string appears to be in CamelCase

    """
    if x.lower() != x and x.upper() != x:
        return True
    else:
        return False

def isiterable(x):
    try:
        for _i in x:
            pass
        return True
    except TypeError:
        return False
    
def validate_seisbench_model_name(model_name, arg_name='x'):
    mname = None
    # Seek camel-case name
    for x in dir(sbm):
        if iscamelcase_str(x):
            if x.lower() == model_name.lower():
                mname = x
    if mname is None:
        raise ValueError(f'(case-insensitive) model name {model} is not included in seisbench.models')
    else:
        return mname
            