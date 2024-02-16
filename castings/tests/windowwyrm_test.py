from wyrm.wyrms.window import WindowWyrm
from wyrm.wyrms._base import Wyrm
from wyrm.structures.rtbufftrace import RtBuffTrace
from wyrm.structures.rtinststream import RtInstStream
from collections import deque
from obspy import Trace, Stream
import numpy as np
import pytest

class test_windowwyrm:

    def __init__(self):
        """
        Initialize example data as testing suite attributes
        """
        self.eqtd_wkwargs = {'model_name': 'EQTransformer',
                      'target_sr': 100.,
                      'target_npts': 6000,
                      'target_channels': 3,
                      'target_order': 'ZNE',
                      'target_overlap': 1800,
                      'target_blinding': 500,
                      'fill_value': 0,
                      'missing_component_rule': 'Zeros'}
        self.eqtf_wkwargs = {'model_name': 'EQTcloneZ',
                      'target_sr': 100.,
                      'target_npts': 6000,
                      'target_channels': 3,
                      'target_order': 'ENZ',
                      'target_overlap': 3000,
                      'target_blinding': 500,
                      'fill_value': -np.inf,
                      'missing_component_rule': 'CloneZ'}


    def test_init(self):
        # Test default init
        windowwyrm = WindowWyrm()
        assert isinstance(windowwyrm, WindowWyrm)
        assert isinstance(windowwyrm, Wyrm)
        # Check default attributes
        assert windowwyrm.windowing_attr == self.eqtd_wkwargs
        # Check that codemap is as expected and key-order insensitive
        assert windowwyrm.code_map == {'E':"E2", 'Z': 'Z3', "N": 'N1'}
        assert windowwyrm.tcf == {'N': 0.8, 'Z': 0.95, 'E': 0.8}
        assert not windowwyrm.debug
        assert windowwyrm.max_pulse_size == 20
        assert windowwyrm.default_starttime is None
        assert isinstance(windowwyrm.queue, deque)
        assert len(windowwyrm.queue) == 0
        assert isinstance(windowwyrm.index, dict)
        assert len(windowwyrm.index) == 0
        # Run tests on following inputs with subroutine test
        # fill_value, missing_component_rule, target_sr, target_npts,
        # target_channels, target_order, target_overlap ,target_blinding
        # model_name
        self.test_set_windowing_attr()

        # Test derivative attributes response to changing windowing_attr values

        # Run tests on input code_map

        # Run tests on input trace_comp_fract

        # Test tolsec response to changing trace_comp_fract values

    def test_set_windowing_attr(self):
        # Test inputs for fill_value
        for _v in [-np.inf, -1, 0, 1., np.inf, None]:
            if _v is not None:
                assert WindowWyrm(fill_value=_v).windowing_attr['fill_value'] == _v
            else:
                assert WindowWyrm(fill_value=_v).windowing_attr['fill_value'] is None
        for _v in ['a', str, True]:
            with pytest.raises(TypeError):
                WindowWyrm(fill_value=_v)
        