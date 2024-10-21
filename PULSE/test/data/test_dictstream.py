import pytest
import numpy as np

from obspy.core.stream import Stream, read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.trace import Trace
from obspy.core.tests.test_stream import TestStream

from PULSE.data.dictstream import DictStream
from PULSE.data.foldtrace import FoldTrace
from PULSE.test.data.util import load_townsend_example, assert_common_trace

class TestDictStream(TestStream):

    # Do setup imports of waveform data once!
    # Just make sure to .copy()!
    sm_st = read()
    lg_st = load_townsend_example()
    # xl_st = load_puget_sound_example()

    def test_init(self):
        # Setup
        ds = DictStream()
        # Test inheritance
        assert isinstance(ds, Stream)
        # Assert traces is dict
        assert isinstance(ds.traces, dict)
        assert ds.traces == {}
        # Assert key_attr
        assert hasattr(ds, 'key_attr')
        assert hasattr(ds, 'supported_keys')
        assert hasattr(ds, 'traces')
        assert isinstance(ds.traces, dict)
        assert ds.key_attr in ds.supported_keys
        assert ds.key_attr == 'id'
        
        # Test example stream input
        assert all([isinstance(_e, Trace) for _e in self.sm_st])
        assert not any([isinstance(_e, FoldTrace) for _e in self.sm_st])
        # Read st as input to ds
        ds = DictStream(self.sm_st.copy())
        # Assert length property
        assert len(ds) == 3
        # Assert that converted traces have changed class, but 
        # their data share a common root
        for _e, _ft in enumerate(ds):
            assert isinstance(_ft, FoldTrace)
            assert_common_trace(_ft, self.sm_st.copy()[_e])
        ids = [_tr.id for _tr in ds]
        # Test that all IDs are used as keys
        for _e in ds.traces.keys():
            assert _e in ids
        # Test that all traces are converted to fold traces
        assert all([isinstance(_e, FoldTrace) for _e in ds])

        # Test alternative key_attr's
        for key in DictStream.supported_keys:
            if key != 'id':
                tr = read()[0]
                assert isinstance(tr, Trace)
                ds = DictStream(tr)
                assert len(ds) == 1
                ft = ds[0]
                # Test conversion
                assert isinstance(ft, FoldTrace)
                # Test key matches
                assert ft.id_keys[key] in ds.traces.keys()

    def test_eq(self):
        """Test suite for the __eq__ method of DictStream
        """        
        # Fetch example st
        st = self.sm_st.copy()
        # Create forward load order DictStream
        ds1 = DictStream(st)
        # Create reverse load order DictStream
        ds2 = DictStream(st[::-1])
        # Assert that the ordered contents of each dictstream do not match
        assert not all(ds1[_e] == ds2[_e] for _e in range(3))
        # Assert that all the keys match 
        assert ds1.traces.keys() == ds2.traces.keys()
        # assert that the traces match
        assert ds1.traces == ds2.traces
        # Assert that the DictStreams match
        assert ds1 == ds2

    def test_iter(self):
        """Test suite for the __iter__ method of DictStream
        """        
        ds = DictStream(load_townsend_example())
        assert hasattr(ds, '__iter__')
        for _ft in ds:
            assert isinstance(_ft, FoldTrace)

    def test_getitem(self):
        """Test suite for the __getitem__ method of DictStream
        """        
        # Setup
        ds = DictStream(self.sm_st.copy())
        # Test Indexing
        assert isinstance(ds[0], FoldTrace)
        assert_common_trace(ds[0], read()[0])
        assert isinstance(ds[-1], FoldTrace)
        assert_common_trace(ds[-1], read()[-1])
        # Slicing
        assert isinstance(ds[:2], DictStream)

        # Key value
        assert isinstance(ds['BW.RJOB..EHZ'], FoldTrace)
        assert ds['BW.RJOB..EHZ'] == ds[0]
        # List of keys
        ds2 = ds[['BW.RJOB..EHZ','BW.RJOB..EHN']]
        assert ds[:2] == ds2
        # others raise
        with pytest.raises(KeyError):
            ds['a']
        with pytest.raises(KeyError):
            ds[['BW.RJOB..EHZ', 1]]
        with pytest.raises(KeyError):
            ds[[3.]]
        
    def test_setitem(self):
        """Test suite for the __setitem__ method of DictStream
        """     
        # Setup   
        st = self.lg_st.copy()
        ds = DictStream(self.sm_st.copy())
        ds2 = ds.copy()
        # Make sure we're asserting st[0] is an ObsPy trace to start with
        assert isinstance(st[0], Trace)
        ds2[2] = st[0].copy()
        # Assert that setting the item converts into FoldTrace
        assert isinstance(ds2[2], FoldTrace)
        # Assert that the data are not changed (TODO: probably move to test extend?)
        np.testing.assert_array_equal(ds2[2], st[0])
        # Assert that
        assert_common_trace(st[0], ds2[2])

        # Test Error catches
        with pytest.raises(IndexError):
            ds[4] = st[0]
        with pytest.raises(TypeError):
            ds[4] = 'abc'
        with pytest.raises(TypeError):
            ds[int] = st[0]

    def test_repr(self):
        ds = DictStream(self.sm_st.copy())
        ds2 = DictStream(self.lg_st.copy())
        assert isinstance(ds.__repr__(), str)
        len0 = len(ds.__repr__())
        len1 = len(ds.__repr__(extended=True))
        assert len0 == len1
        len0 = len(ds2.__repr__())
        len1 = len(ds2.__repr__(extended=True))
        assert len0 < len1

    def test_str(self):
        ds = DictStream(read())
        assert isinstance(ds.__str__(), str)
        assert ds.__str__() == 'PULSE.data.dictstream.DictStream'
        assert ds.__str__(short=True) == 'DictStream'


    # def test_search(self):
    #     # Setup
    #     ds = DictStream(self.lg_st.copy())
    #     ds2 = ds.search(['UW.P*..EN?*', 'UW.N*..EH?*'])

    def test_split(self):
        ds = DictStream(self.lg_st.copy())
