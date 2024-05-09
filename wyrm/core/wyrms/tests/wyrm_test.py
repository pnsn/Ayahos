import pytest, sys, os
# TODO: Need to sort out path addition for local 
from wyrm.core.wyrm import Wyrm

class test_wyrm:
    """
    Test suite for wyrm.core.wyrm.Wyrm class
    """
    # Initialize an object that will be tested on
    def test_init(self):
        # Assert the initialized object is of the appropriate type
        assert type(Wyrm()) is Wyrm

        # Check bool inputs
        for v in [True, False]:
            for k in ['timestamp','debug']:
                input = {v: k}
                assert getattr(Wyrm(**input), k) == v

        # Check numeric inputs
        for v in [None, -1, -1.3, 0, 1, 2, 1.2]:
            if isinstance(v, (int, float, type(None))):
                if isinstance(v, (int, float)):
                    assert getattr(Wyrm(max_pulse_size=v), 'max_pulse_size') == int(v)
                elif v is None:
                    assert getattr(Wyrm(max_pulse_size=v), 'max_pulse_size') is None
                else:
                    with pytest.raises(ValueError):
                        Wyrm(max_pulse_size=v)
            else:
                with pytest.raises(TypeError):
                    Wyrm(max_pulse_size=v)
    
    def test_copy(self):
        obj1 = Wyrm()
        obj2 = obj1.copy()
        obj2.max_pulse_size = 1
        assert obj2.max_pulse_size == 1
        assert obj1.max_pulse_size is None

    def test_pulse(self):
        obj1 = Wyrm()
        for x in [None, 0,'a',int, True, Wyrm(), [1,2,3]]:
            y = obj1.pulse(x)
            assert x == y
    
    def test___repr__(self):
        assert isinstance(Wyrm().__repr__, str)

    def test___str__(self):
        assert isinstance(Wyrm().__str__, str)