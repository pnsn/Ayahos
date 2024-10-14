import pytest
from obspy.core.stream import Stream, read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.trace import Trace
from obspy.core.tests.test_stream import TestStream
from PULSE.data.dictstream import DictStream


class TestDictStream(TestStream):

    def test_init(self):
        # Setup
        ds = DictStream()
        # Test inheritance
        assert isinstance(ds, Stream)
        # Assert traces is dict
        assert isinstance(ds.traces, dict)
        assert ds.traces == {}
        # Assert has stats
        assert hasattr(ds, 'stats')
        # assert isinstance(ds.stats, DSStats)
        # assert isinstance(ds.stats, AttribDict)
        # Assert key_attr
        assert hasattr(ds, 'key_attr')
        assert hasattr(ds, 'supported_keys')
        assert ds.key_attr in ds.supported_keys
        assert ds.key_attr == 'id'
        
        # Test provided traces
        st = read()
        ds = DictStream(st)

        # Test alternative key_attr


        