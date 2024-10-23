"""
:module: PULSE.util.docs
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose: This module contains helper methods for docstrings
:attribution: This is directly copied from the ObsPlus project's
    obsplus.utils.docs module.

"""
import textwrap
from typing import Dict, Any, Union, Sequence
def compose_docstring(**kwargs: Union[str, Sequence[str]]):
    """
    Decorator for composing docstrings.

    This allows components of docstrings which are often repeated to be
    specified in a single place. Values provided to this function should
    have string keys and string or list values. Keys are found in curly
    brackets in the wrapped functions docstring and their values are
    substituted with proper indentation.

    Notes
    -----
    A function's docstring can be accessed via the `__docs__` attribute.

    Examples
    --------

    @compose_docstring(some_value='10')
    def example_function():
        '''
        Some useful description

        The following line will be the string '10':
        {some_value}
        '''
    """

    def _wrap(func):
        docstring = func.__doc__
        # iterate each provided value and look for it in the docstring
        for key, value in kwargs.items():
            value = value if isinstance(value, str) else "\n".join(value)
            # strip out first line if needed
            value = value.lstrip()
            search_value = "{%s}" % key
            # find all lines that match values
            lines = [x for x in docstring.split("\n") if search_value in x]
            for line in lines:
                # determine number of spaces used before matching character
                spaces = line.split(search_value)[0]
                # ensure only spaces precede search value
                assert set(spaces) == {" "}
                new = textwrap.indent(textwrap.dedent(value), spaces)
                docstring = docstring.replace(line, new)

        func.__doc__ = docstring
        return func

    return _wrap