import pytest
from obspy import read
from obspy.core.trace import Stats
from obspy.core.util import AttribDict
from obspy.core.tests.test_stats import TestStats
from obspy.core.tests.test_util_attribdict import TestAttribDict
from pandas import Series
from PULSE.data.header import MLStats, PulseStats

class TestMLStats(TestStats):

    egstats = read()[0].stats

    def test_init(self):
        header = MLStats()
        # Assert inheritance
        assert isinstance(header, Stats)
        assert isinstance(header, AttribDict)
        # Assert defaults
        assert header.location == ''
        assert header.processing == []
        assert header.model == ''
        assert header.weight == ''
        # Test dict input
        header = MLStats({'station': 'GPW', 'network': 'UW',
                          'location':'', 'channel': 'EHZ',
                          'model':'EQTransformer','weight':'pnw'})
        assert header.network == 'UW'
        assert header.station == 'GPW'
        assert header.location == ''
        assert header.channel == 'EHZ'
        assert header.model == 'EQTransformer'
        assert header.weight == 'pnw'
        # Test Stats input
        header = MLStats(self.egstats)
        for _k, _v in self.egstats.items():
            assert header[_k] == _v
        # Test Stats input
        tr = read()[0]
        header = MLStats(tr.stats)
        for _k, _v in tr.stats.items():
            if _k == 'location' and _v == '':
                assert header[_k] == ''
            else:
                assert header[_k] == _v
        
        # Other types inputs raises
        with pytest.raises(TypeError):
            header = MLStats('abc')
            header = MLStats(['abc'])

    def test_str(self):
        header = MLStats()
        assert isinstance(header.__str__(), str)

    def test_utc2nearest_index(self):
        header = MLStats(self.egstats)
        t0 = self.egstats.starttime
        dt = self.egstats.delta
        eta = dt*0.1
        # Test starttime equivalent
        assert 0 == header.utc2nearest_index(t0)
        # Test small offset
        assert 0 == header.utc2nearest_index(t0 + eta)
        # Test exact delta offset
        assert 1 == header.utc2nearest_index(t0 + dt)
        # Test nearby delta offset
        assert 1 == header.utc2nearest_index(t0 + dt + eta)
        assert 1 == header.utc2nearest_index(t0 + dt - eta)
        # Test negative
        assert -1 == header.utc2nearest_index(t0 - dt)
        assert -1 == header.utc2nearest_index(t0 - dt + eta)
        assert -1 == header.utc2nearest_index(t0 - dt - eta)

    def test_copy(self):
        header = MLStats(self.egstats)
        header2 = header.copy()
        header2.station='YEAH'
        assert header != header2
        assert header2.station == 'YEAH'
        assert header.station == 'RJOB'

    def test_properties(self):
        header = MLStats(self.egstats)
        # Test all default values (base values) occur
        for _attr in self.egstats.defaults.keys():
            assert self.egstats[_attr] == header[_attr]
            assert self.egstats[_attr] == getattr(header, _attr)
        # Assert that compund values are present
        for _attr in ['inst','site','comp','mod','nslc','sncl']:
            assert hasattr(header, _attr)
        # Assert that nslc for empty mod/weight matches ID
        assert header.id == header.nslc
        # Assert that nslc does not match if mod or weight are assigned
        header.model='EQTransformer'
        assert header.id == f'{header.nslc}.{header.model}.'
        header.weight='pnw'
        assert header.id == f'{header.nslc}.{header.model}.{header.weight}'
        header.model=''
        assert header.id == f'{header.nslc}..{header.weight}'
        # Test compound codes
        header.model='EQTransformer'
        assert header.comp == header.channel[-1]
        assert header.inst == header.nslc[:-1]
        assert header.site == f'{header.network}.{header.station}.{header.location}'
        assert header.mod == f'{header.model}.{header.weight}'

    def test_get_id_keys(self):
        header = MLStats(self.egstats)
        assert isinstance(header.get_id_keys(), AttribDict)



class TestPulseStats(TestAttribDict):
    """Tests for the :class:`~PULSE.data.header.PulseStats` class
    """    
    def test_class_variables(self):
        """Test suite for PulseStats class variables
        """        
        assert PulseStats.readonly == ['pulserate','runtime']
        assert isinstance(PulseStats._refresh_keys, set)
        assert PulseStats._refresh_keys == {'starttime','endtime','niter'}
        assert isinstance(PulseStats.defaults, dict)
        assert isinstance(PulseStats._types, dict)

    def test_init(self):
        """Test suite for PulseStats.__init__
        """        
        header = PulseStats()
        assert isinstance(header, AttribDict)
        # Test _types restrictions
        for _v in ['a', int(1), float(1.1)]:
            for _k in PulseStats.defaults.keys():
                # Catch readonly attribute assignment error
                if _k in PulseStats.readonly:
                    with pytest.raises(AttributeError):
                        PulseStats(header={_k:_v})
                # Test string attributes
                elif _k in ['modname','stop']:
                    if isinstance(_v, str):
                        assert PulseStats(header={_k:_v})
                    else:
                        assert isinstance(PulseStats(header={_k:_v})[_k], str)
                # Test int attributes
                elif _k in ['niter','in0','in1','out0','out1']:
                    if isinstance(_v, int):
                        assert PulseStats(header={_k:_v})[_k] == _v
                    elif isinstance(_v, float):
                        assert PulseStats(header={_k:_v})[_k] == int(_v)

                    else:
                        with pytest.raises(ValueError):
                            PulseStats(header={_k:_v})
                # Test float attributes
                else:
                    if isinstance(_v, float):
                        assert PulseStats(header={_k:_v})[_k] == _v
                    elif isinstance(_v, int):
                        assert PulseStats(header={_k:_v})[_k] == float(_v)
                    else:
                        with pytest.raises(ValueError):
                            PulseStats(header={_k:_v})
    
    def test_setattr(self):
        # Test _types restrictions
        for _v in ['a', int(1), float(1.1)]:
            for _k in PulseStats.defaults.keys():
                # SETUP Create new object
                header = PulseStats()
                # Catch readonly attribute assignment error
                if _k in PulseStats.readonly:
                    with pytest.raises(AttributeError):
                        header[_k]= _v
                # Test string attributes
                elif _k in ['modname','stop']:
                    # Test string input
                    if isinstance(_v, str):
                        header[_k] = _v
                        assert header[_k] == _v
                        assert getattr(header, _k) == _v
                    # Test other input
                    else:
                        header[_k] = _v
                        assert getattr(header, _k) == str(_v)
                # Test int attributes
                elif _k in ['niter','in0','in1','out0','out1']:
                    if isinstance(_v, (int, float)):
                        header[_k] = _v
                        assert getattr(header, _k) == int(_v)
                    else:
                        with pytest.raises(ValueError):
                            header[_k] = _v
                # Test float attributes
                else:
                    if isinstance(_v, (int, float)):
                        header[_k] = _v
                        assert getattr(header, _k) == float(_v)
                    else:
                        with pytest.raises(ValueError):
                            header[_k] = _v
        # Test updates
        header = PulseStats(header={'starttime':0, 'endtime':1, 'niter':3})
        # Test positive runtime
        assert header.runtime == 1.
        assert header.pulserate == 3.
        # Test longer runtime
        header.endtime = 2
        assert header.runtime == 2.
        assert header.pulserate == 1.5
        # Test 0 runtime
        header.endtime = 0
        assert header.runtime == 0
        assert header.pulserate == 0
        # Test negative runtime
        header.starttime = 1
        assert header.runtime == -1.
        assert header.pulserate == 0

    def test_copy(self):
        """Test suite for PulseStats.copy
        """        
        header = PulseStats()
        header2 = header.copy()
        assert header == header2
        header2.niter=3
        header2.endtime=2
        assert header.niter != header2.niter
        assert header.endtime != header2.endtime
        assert header2.runtime == 2.
        assert header2.pulserate == 1.5
        assert header.pulserate == 0
        assert header.runtime == 0
    
    def test_asdict(self):
        """Test suite for PulseStats.asdict
        """        
        header = PulseStats()
        assert isinstance(header.asdict(), dict)
        hd = header.copy().asdict()
        for _k, _v in header.items():
            assert _v == hd[_k]

    def test_asseries(self):
        """Test suite for PulseStats.asseries
        """        
        header = PulseStats()
        ser = header.copy().asseries()
        assert isinstance(ser, Series)
        for _ind in header.keys():
            assert ser[_ind] == header[_ind]
        

