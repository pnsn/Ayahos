"""
:module: wyrm.classes.tests.test_pyew_msg
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:attribution:
    This testing suite is based on the structure of the 
    obspy.realtime.rttrace test suite.
"""

import pytest
from wyrm.core.pyew_msg import *


class Test_PyEW_Msg:
    """
    Test suite for wyrm.classes.pyew_msg.PyEW_Msg
    """
    def test_init(self):
        """
        Thest the __init__ method of the PyEW base class
        """
        msg = PyEW_Msg()
        assert msg


class Test_PyEW_WaveMsg:
    """
    Test suite for wyrm.classes.pyew_msg.PyEW_WaveMsg
    """
    def test_init(self)
        msg = PyEW_WaveMsg()
        assert len(msg) == 0:
            