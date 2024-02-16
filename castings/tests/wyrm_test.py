"""
:module: wyrm.tests.wyrm_test
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose: 
    This module provides a test suite for the Wyrm class
"""
import pytest
from wyrm.core.wyrms.wyrm import Wyrm


class test_wyrm:
    """
    Test suite for wyrm.core.wyrm.Wyrm
    """
    def __init__(self):
        self.type_list = [True, 1, 'a', Wyrm(), str, 1.]
    
    def test_init(self):
        assert type(Wyrm()) is Wyrm
        for x in self.type_list:
            with pytest.raises(TypeError):
                Wyrm(x)

    def test_repr(self):
        # Assert that __repr__() returns string
        assert isinstance(Wyrm().__repr__(), str)
        # Check TypeError raise if arbitrary argument is supplied to __repr__
        for x in self.type_list:
            with pytest.raises(TypeError):
                Wyrm().__repr__(x)

    def test_pulse(self):
        for x in self.type_list:
            assert Wyrm().pulse(x) == x
            for y in self.type_list:
                with pytest.raises(TypeError):
                    Wyrm().pulse(x, y)
