from wyrm.message.window import WindowMsg
from wyrm.structures.rtbufftrace import RtBuffTrace
from obspy.core.tests.test_stream import TestStream
from obspy import read, Trace
import os


class TestWindowMsg:
    """
    Test suite for wyrm.message.window.WindowMsg
    """

    def mseed_stream(self):
        st = read(
            os.path.join("..", "..", "example", "uw61957912", "CC.ARAT..BH_ENZ.mseed")
        )
        return st

    def test_init(self):
        """
        Tests the __init__ method fo the WindowMsg object.
        """
        st = self.mseed_stream()
        # Run tests from test_init() from obspy's test stream first
        # Empty
        wind0 = WindowMsg()
        assert len(wind0) == 0
        assert wind0.ref_starttime is None
        # One vertical trace
        assert len(WindowMsg(Z=st[2])) == 1

        # Check basic handling of rules & data ingestion
        # One vertical trace and one horizontal with 'zeros' rule
        assert len(WindowMsg(Z=st[2], N=st[0], window_fill_rule="zeros")) == 1
        # One vertical trace and one horizontal trace with 'cloneZ' rule
        assert len(WindowMsg(Z=st[2], N=st[0], window_fill_rule="cloneZ")) == 1
        # One vertical trace and one horizontal trace with 'cloneHZ' rule
        assert len(WindowMsg(Z=st[2], N=st[0], window_fill_rule="cloneHZ")) == 2
        # One horizontal trace
        assert len(WindowMsg(Z=None, N=st[0], window_fill_rule="zeros")) == 1
        # Two horizontal traces
        assert len(WindowMsg(Z=None, N=st[0], E=st[1], window_fill_rule="zeros")) == 2

        # Z compatability check tests
        
