"""
:module: wyrm.tests.tubewyrm_test
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    Provides a test suite for the TubeWyrm class
"""

from wyrm.wyrms.tubewyrm import TubeWyrm
from wyrm.wyrms.wyrm import Wyrm
from collections import deque
import numpy as np
import pytest
from time import time
from copy import deepcopy


class test_tubewyrm:
    def __init__(self):
        """
        Initialize some example data for testing TubeWyrm
        """
        self.eg_wyrm = Wyrm()
        self.eg_val = 1.0
        self.eg_val_list = [1.0, 2.0, 3.0]
        self.eg_val_array = np.array(self.eg_val_list)
        self.eg_val_deque = deque([self.eg_list])
        self.eg_wyrm_deque = deque([
            Wyrm(scalar=1),
            Wyrm(scalar=2),
            Wyrm(scalar=3)
        ])
        self.eg_wyrm_list = [Wyrm(scalar=1), Wyrm(scalar=2), Wyrm(scalar=3)]
        self.eg_mixed_list = [1.0, Wyrm(scalar=1), "a"]
        return self

    def test_init(self):
        """
        Test different forms of __init__ and exceptions
        """
        assert isinstance(TubeWyrm(), Wyrm)
        assert isinstance(TubeWyrm(), TubeWyrm)
        # # Test empty init defaults
        assert isinstance(TubeWyrm().queue, deque)
        assert len(TubeWyrm().queue) == 0
        assert TubeWyrm().wait_sec == 0.0
        assert TubeWyrm().max_pulse_size is None
        assert not TubeWyrm().debug
        assert TubeWyrm(debug=True).debug

        # # Test wyrm_queue valid inputs
        tubewyrm = TubeWyrm(wyrm_queue=self.eg_wyrm_deque)
        assert tubewyrm.wyrm_queue == self.eg_wyrm_deque
        # Test conversion of a single wyrm to deque
        assert isinstance(TubeWyrm(wyrm_queue=Wyrm()).wyrm_queue, deque)
        # Test conversion of list of wyrms to deque
        assert isinstance(TubeWyrm(wyrm_queue=self.eg_wyrm_list), deque)
        # Test exceptions
        with pytest.raises(TypeError):
            TubeWyrm(wyrm_queue=self.eg_mixed_list)
            TubeWyrm(wyrm_queue=self.eg_val)
            TubeWyrm(wyrm_queue=self.eg_val_list)
            TubeWyrm(wyrm_queue=self.eg_val_deque)

        # Test wait_sec valid inputs
        assert tubewyrm.wait_sec == 0.0
        assert TubeWyrm(wait_sec=0.0)
        # Test wait_sec exceptions (coupled to icc.bounded_floatlike)
        with pytest.raises(TypeError):
            TubeWyrm(wait_sec="a")
            TubeWyrm(wait_sec=[1])
        with pytest.raises(ValueError):
            TubeWyrm(wait_sec=-1)
            TubeWyrm(wait_sec=np.inf)
            TubeWyrm(wait_sec=6000 + 1e-12)
        assert TubeWyrm(wait_sec=6000).wait_sec == 6000

    def test_append(self):
        """
        Test TubeWyrm's append() method
        """
        # Create an empty wyrm
        tubewyrm = TubeWyrm()
        # Do a "basic" append
        tubewyrm.append(self.eg_wyrm_deque)
        # assert that appended is identical to source material
        assert tubewyrm.wyrm_queue == self.eg_wyrm_deque
        # Create another empty tubewyrm object
        tubewyrm = TubeWyrm()
        # Append a series of wyrms
        for _wyrm in self.eg_wyrm_list:
            tubewyrm.append(_wyrm)
        # Assert that this also produces an identical 
        assert tubewyrm.wyrm_queue == self.eg_wyrm_deque
        # Do the same as above, but for the appendleft() equivalent
        tubewyrm = TubeWyrm()
        for _wyrm in self.eg_wyrm_list:
            tubewyrm.append(_wyrm, end="left")
        assert tubewyrm.wyrm_queue == deque(self.eg_wyrm_list[::-1])
        # Run checks on case insensitivity for append "end" argument
        # Compose list of example tubewyrms using append
        tubewyrm_list = []
        for _rs in ['right', 'RIGHT', 'Right', 'r', 'R']:
            _tubewyrm = TubeWyrm().append(self.eg_wyrm_deque, end=_rs)
            tubewyrm_list.append(_tubewyrm)
        # Do pariwise comparisons including auto-comparisons
        for _i, _itw in enumerate(tubewyrm_list):
            for _j, _jtw in enumerate(tubewyrm_list):
                if _i >= _j:
                    assert _itw == _jtw

    def test_pop(self):
        """
        Test self.pop() method for TubeWyrm
        """
        # For a sampling of aliases for end='left')
        for _la in ["left", "l", "LEFT", "L", "Left"]:
            tubewyrm = TubeWyrm(wyrm_queue=deepcopy(self.eg_wyrm_deque))
            popl = tubewyrm.pop(end=_la)
            # Check that popped entry for TubeWyrm is identical to action on source deque
            assert popl == deepcopy(self.eg_wyrm_deque).popleft()
        # For a sampling of aliases for end='right')
        for _ra in ["left", "l", "LEFT", "L", "Left"]:
            tubewyrm = TubeWyrm(wyrm_queue=deepcopy(self.eg_wyrm_deque))
            popl = tubewyrm.pop(end=_ra)
            # Check that popped entry for TubeWyrm is identical to action on source deque
            assert popl == deepcopy(self.eg_wyrm_deque).pop()
        # Assert that the queue from the last iterated test above has
        # one less Wyrm than its __init__ queue
        assert len(tubewyrm.wyrm_queue) == len(self.eg_wyrm_deque) - 1

    def test_pulse(self):
        """
        Test self.pulse(x) method for TubeWyrm
        """
        # Initialize example tubewyrm with 0 wait_sec
        tubewyrm = TubeWyrm(wyrm_queue=self.eg_wyrm_deque)
        # Run a pulse example
        yt = tubewyrm.pulse(x=self.eg_val_array)
        # Set an initial value
        yc = self.eg_val_array
        # Run chained wyrms in order
        for _wyrm in self.eg_wyrm_deque:
            yc = _wyrm.pulse(yc)
        # Assert pulse and chained wyrms return the same thing
        assert yc == yt
        # Initialize example tubewyrm with non-0 wait_sec
        tubewyrm = TubeWyrm(wyrm_queue=self.eg_wyrm_deque, wait_sec=3.0)
        # Time runtime
        tick = time()
        y = tubewyrm.pulse(x=self.eg_val_array)
        tock = time()
        # Assert that runtime is g.e. the number of sleep seconds expected
        assert tock - tick >= tubewyrm.wait_sec*(len(tubewyrm.wyrm_queue) - 1)
        # Assert that pulse in this case also returns the same value as the
        # tested pulses above
        assert y == yc == yt
