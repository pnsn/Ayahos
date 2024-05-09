import pytest, copy, warnings, io, pickle
from obspy import UTCDateTime, Stream, Trace, UTCDateTime, read
from obspy.core.util.testing import WarningsCapture
from obspy.core.util import AttribDict
# Get obspy.core.uti.Stats test suite for inheritance
from obspy.core.tests.test_stats import TestStats
# Alias MLStats to Stats to overwrite Stats calls in inherited class test methods
from wyrm.core.trace.mltrace import MLStats as Stats


class MLStats_Test(TestStats):
    """
    Test suite inheriting from ObsPy's Stats test suite that
    only includes updated methods where appropriate
    """
    nslcmw = ['network','station','location','channel','model','weight']
    def test_deepcopy(self):
        """
        Tests deepcopy method of Stats object.
        Modified from ObsPy test suite
        """
        stats = Stats()
        stats.network = 'BW'
        stats['station'] = 'ROTZ'
        stats['other1'] = {'test1': '1'}
        stats['other2'] = AttribDict({'test2': '2'})
        stats['other3'] = 'test3'
        stats2 = copy.deepcopy(stats)
        stats.location = '01'
        stats.network = 'CZ'
        stats.station = 'RJOB'
        # Next 2 lines are new
        stats.model = 'EQTransformer'
        stats.weight = 'pnw'

        assert stats2.__class__ == Stats
        assert stats2.network == 'BW'
        assert stats2.station == 'ROTZ'
        assert stats2.other1.test1 == '1'
        assert stats2.other1.__class__ == AttribDict
        assert len(stats2.other1) == 1
        assert stats2.other2.test2 == '2'
        assert stats2.other2.__class__ == AttribDict
        assert len(stats2.other2) == 1
        assert stats2.other3 == 'test3'
        assert stats.network == 'CZ'
        assert stats.station == 'RJOB'
        # Next 4 lines are new
        assert stats2.location == '--'
        assert stats.location == '01'
        assert stats.model == 'EQTransformer'
        assert stats.weight == 'pnw'


    def test_compare_with_dict(self):
        """
        Checks if Stats is still comparable to a dict object.
        """
        adict = {
            'network': '', 'sampling_rate': 1.0, 'test': 1, 'station': '',
            'location': '--', 'starttime': UTCDateTime(1970, 1, 1, 0, 0),
            'delta': 1.0, 'calib': 1.0, 'npts': 0,
            'endtime': UTCDateTime(1970, 1, 1, 0, 0), 'channel': '',
            'model': '', 'weight': ''}
        ad = Stats(adict)
        assert ad == adict
        assert adict == ad

    def test_non_str_in_nsclmw_raise_warning(self):
        """
        Ensure assigning a non-str value to network, station, location, or
        channel issues a warning, then casts value into str. See issue # 1995
        """
        stats = Stats()

        for val in self.nslcmw:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('default')
                setattr(stats, val, 42)
            # make sure a warning was issued
            assert len(w) == 1
            exp_str = 'Attribute "%s" must be of type ' % val
            assert exp_str in str(w[-1].message)
            # make sure the value was cast to a str
            new_val = getattr(stats, val)
            assert new_val == '42'

    def test_nsclmw_cannot_be_none(self):
        """
        Ensure the nslcmw values can't be assigned to None but rather None
        gets converted to a str
        added method based on same in obspy for `nslc`
        """
        stats = Stats()
        for val in self.nslcmw:
            with pytest.warns(UserWarning):
                setattr(stats, val, None)
            assert getattr(stats, val) == 'None'
    
    @pytest.mark.filterwarnings('ignore:Attribute')
    def test_different_string_types(self):
        """
        Test the various types of strings found in the wild get converted to
        native_str type.
        """
        nbytes = bytes('HHZ', 'utf8')
        the_strs = ['HHZ', nbytes, u'HHZ']

        stats = Stats()

        for a_str in the_strs:
            for nslcmw in self.nslcmw:
                setattr(stats, nslcmw, a_str)
                assert isinstance(getattr(stats, nslcmw), str)