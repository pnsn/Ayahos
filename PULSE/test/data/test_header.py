import pytest
from obspy import read
from obspy.core.trace import Stats
from obspy.core.util import AttribDict
from obspy.core.tests.test_stats import TestStats
from PULSE.data.header import MLStats

class TestMLStats(TestStats):

    def test_init(self):
        header = MLStats()
        # Assert inheritance
        assert isinstance(header, Stats)
        assert isinstance(header, AttribDict)
        # Assert defaults
        assert header.location == '--'
        assert header.processing == []
        assert header.model == ''
        assert header.weight == ''
        # Test dict input
        header = MLStats({'station': 'GPW', 'network': 'UW', 'location':'', 'channel': 'EHZ'})
        assert header.network == 'UW'
        assert header.station == 'GPW'
        assert header.location == '--'
        assert header.channel == 'EHZ'
        # Test Stats input
        tr = read()[0]
        header = MLStats(tr.stats)
        for _k, _v in tr.stats.items():
            if _k == 'location' and _v == '':
                assert header[_k] == '--'
            else:
                assert header[_k] == _v
        
        # Other types inputs raises
        with pytest.raises(TypeError):
            header = MLStats('abc')
            header = MLStats(['abc'])