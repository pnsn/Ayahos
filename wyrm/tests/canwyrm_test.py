"""
:module: wyrm.tests.canwyrm_test.py
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose: 
    Provides tests for the CanWyrm class
"""

from wyrm.wyrms.canwyrm import CanWyrm
from wyrm.wyrms.wyrm import Wyrm
from collections import deque
import pytest


class test_canwyrm:
    def __init__(self):
        """
        Create some example data
        """
        self.eg_wyrm_list = [Wyrm(scalar=1), Wyrm(scalar=2), Wyrm(scalar=3)]
        self.eg_wyrm_deque = deque(self.eg_wyrm_list)
        self.eg_val = 2.0
        return self

    def test_init(self):
        """
        Test suite for the CanWyrm.__init__ method
        """
        # Initialize default args CanWyrm
        canwyrm = CanWyrm()
        # Check parameter types/values
        assert isinstance(canwyrm.queue, deque)
        assert len(canwyrm.queue) == 0
        assert canwyrm.wait_sec == 0
        assert canwyrm.output_type == deque
        assert canwyrm.concat_method == "appendleft"
        # Show that compatability check on output works for default inputs
        assert canwyrm.concat_method in canwyrm.output_type.__dict__.keys()
        # 
        assert canwyrm.max_pulse_size is None
        assert canwyrm.debug is False
        assert CanWyrm(debug=True) is True

        # Test wyrm_queue inputs (rigorous tests are done in TubeWyrm)
        canwyrm = CanWyrm(wyrm_queue=self.eg_wyrm_deque)
        assert canwyrm.queue == self.eg_wyrm_deque
        canwyrm = CanWyrm(wyrm_queue=self.eg_wyrm_list)
        assert canwyrm.queue == self.eg_wyrm_deque
        assert len(canwyrm.queue) == len(self.eg_wyrm_deque)

        # Test output_type
        assert isinstance(CanWyrm().output_type, type)
        # Test error exception for mismatch on output_type and concat_method
        with pytest.raises(AttributeError):
            CanWyrm(output_type=list)
        assert CanWyrm(output_type=list, concat_method='append')
        # Test alternative output_type
        canwyrm = CanWyrm(output_type=list, concat_method='append')
        assert isinstance(canwyrm.output_type, type)
        assert canwyrm.output_type == list
        assert isinstance(canwyrm.concat_method, str)
        assert canwyrm.concat_method == 'append'

    def test_pulse(self):
        """
        Test suite for the CanWyrm.pulse method
        """
        canwyrm = CanWyrm(wyrm_queue=self.eg_wyrm_deque)
        y = canwyrm.pulse(x=self.eg_val)
        # Assert output y is expected format
        assert isinstance(y, canwyrm.output_type)
        # Assert output y has same number of elements as number of wyrms in queue
        assert len(y) == len(canwyrm.wyrm_queue)
        # Assert that outputs are in expected order and values are correct
        bool_list = []
        for _i, _y in enumerate(y):
            status = _y == canwyrm.wyrm_queue[_i].pulse(self.eg_val)
            bool_list.append(status)
        assert all(bool_list)
