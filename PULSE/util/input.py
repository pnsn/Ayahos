"""
:module: wyrm.util.input_compatability_checks.py
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    Provide defined methods for checking the validity of
    common input types to classes/methods in wyrm that require
    several case tests.

"""
import numpy as np
from collections import deque
# import seisbench.models as sbm
from math import inf


def isPyEWwave(x):
    """
    Assess if an input dict object has the necessary keys
    to meet the definition of a PyEarthworm Wave object:

    wave_template = {
        "station": str,
        "network": str,
        "channel": str,
        "location": str,
        "nsamp": int,
        "samprate": float,
        "startt": float,
        "endt": float,
        "datatype": str,
        "data": np.ndarray,
    }

    :: INPUT ::
    :param x: [dict] candidate dictionary

    :: OUTPUT ::
    :return status: [bool] all requirements met?
    """
    wave_template = {
        "station": str,
        "network": str,
        "channel": str,
        "location": str,
        "nsamp": int,
        "samprate": float,
        "startt": float,
        "endt": float,
        "datatype": str,
        "data": np.ndarray,
    }
    bool_list = []
    if isinstance(x, dict):
        if all([_k in wave_template.keys() for _k in x.keys()]):
            bool_list.append(True)
        else:
            bool_list.append(False)
        if all([isinstance(x[_k], wave_template[_k]) for _k in wave_template.keys()]):
            bool_list.append(True)
        else:
            bool_list.append(False)
        status = all(bool_list)
    else:
        status = False

    return status


def bounded_value(x, minimum=None, maximum=None, inclusive=True):
    """Check if input value x falls in a specified numerical value range

    :: INPUTS ::
    :param x: int-like or float-like value to assess
    :param minimum: [float-like], [int-like], -inf, None
                    Minimum value for bounded interval
    :param maximum: [float-like], [int-like], inf, None
                    Maximum value for bounded interval
    :param inclusive [bool] include minimum and maximum values in interval?
                    True -> x \in [minimum, maximum]
                    False -> x \in (minimum, maximum)
                    NOTE: x \in (min, max] and x \in [min, max) currently
                        not supported.
    :: OUTPUT ::
    :return status: [bool] Is x in the specified bounds?
    """
    if minimum in [None, -inf, -np.inf]:
        minb = -np.inf
    elif np.isfinite(minimum):
        minb = minimum
    else:
        raise ValueError("specified minimum must be None, inf, or a finite value")

    if maximum in [None, inf, np.inf]:
        maxb = np.inf
    elif np.isfinite(maximum):
        maxb = maximum
    else:
        raise ValueError("specified maximum must be None, -inf or a finite value")

    if not inclusive:
        status = minb < x < maxb
    else:
        status = minb <= x <= maxb
    return status


def bounded_intlike(x, name="x", minimum=1, maximum=None, inclusive=True):
    """
    If input x is an int-like value in a specified bounded interval,
    return int(x), otherwise raise errors

    :: INPUTS ::
    :param x: [int] or [float] value to assess
    :param name: [str] name of parameter to include in error messages
    :param minimum: [int-like], [-inf], [None]
                minimum bound value - see bounded_value()
    :param maximum: [int-like], [inf], [None]
                maximum bound value - see bounded_value()
    :param inclusive: [bool] - include min/max in bound?

    :: OUTPUT ::
    :return int(x): if x is int-like and in bounds, return output of int(x)
                    if x is not int-like, raise TypeError
                    if x is int-like but out of bounds, raise ValueError
    """
    if not isinstance(x, (int, float)):
        raise TypeError(f"{name} must be int-like")
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite")
    if bounded_value(x, minimum=minimum, maximum=maximum, inclusive=inclusive):
        return int(x)
    else:
        if inclusive:
            raise ValueError(f"{name} must be in the bounds [{minimum}, {maximum}]")
        else:
            raise ValueError(f"{name} must be in the bounds ({minimum}, {maximum})")


def bounded_floatlike(x, name="x", minimum=1, maximum=None, inclusive=True):
    """
    If input x is an float-like value in a specified bounded interval,
    return float(x), otherwise raise errors

    :: INPUTS ::
    :param x: [int] or [float] value to assess
    :param name: [str] name of parameter to include in error messages
    :param minimum: [float-like], [-inf], [None]
                minimum bound value - see bounded_value()
    :param maximum: [float-like], [inf], [None]
                maximum bound value - see bounded_value()
    :param inclusive: [bool] - include min/max in bound?

    :: OUTPUT ::
    :return float(x): if x is float-like and in bounds, return output of float(x)
                    if x is not float-like, raise TypeError
                    if x is float-like but out of bounds, raise ValueError
    """
    if not isinstance(x, (int, float)):
        raise TypeError(f"{name} must be float-like")
    if not np.isfinite(x):
        if maximum not in [None, inf, np.inf] or minimum not in [None, -inf, -np.inf]:
            raise ValueError(f"{name} must be finite")
    if bounded_value(x, minimum=minimum, maximum=maximum, inclusive=inclusive):
        return float(x)
    else:
        if inclusive:
            raise ValueError(f"{name} must be in the bounds [{minimum}, {maximum}]")
        else:
            raise ValueError(f"{name} must be in the bounds ({minimum}, {maximum})")


def iterable_characters(x, name="x", listlike_types=(list, deque, np.array)):
    if isinstance(x, listlike_types):
        bool_list = []
        for _x in x:
            bool_list.append(isinstance(_x, str))
            bool_list.append(len(_x) == 1)
        if all(bool_list):
            return x
        else:
            raise TypeError(
                f"list-like {name} must only comprise single-character strings"
            )
    elif isinstance(x, str):
        return [_x for _x in x]
    else:
        raise TypeError(
            f"{name} must be type str or a list-like object of single characters"
        )


def none_str(x, name="x", warn=False):
    if not isinstance(x, (type(None), str)):
        raise TypeError(f"{name} must be type str or None")
    elif isinstance(x, type(None)):
        if warn:
            print(f"{name} = None is strictly a placeholder")
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


# def validate_seisbench_model_name(model_name, arg_name="x"):
#     mname = None
#     # Seek camel-case name
#     for x in dir(sbm):
#         if iscamelcase_str(x):
#             if x.lower() == model_name.lower():
#                 mname = x
#     if mname is None:
#         raise ValueError(
#             f"(case-insensitive) model name {model} is not included in seisbench.models"
#         )
#     else:
#         return mname


# def parse_bounded_interval(x, str_repr):
#     if not isinstance(str_repr, str):
#         raise TypeError('str_repr must be type str')
#     if ',' not in str_repr:
#         raise SyntaxError('interval ends must be comma-delimited')

#     if not len(str_repr[1:-1].split(','))==2:
#         raise SyntaxError
#     if str_repr[0] in ['[', '(']:

        

#     if all(_val.isnumeric() for _val in str_repr[1:-1].split(','))

#     if str_repr[0] == '[':
#         lower_status = 