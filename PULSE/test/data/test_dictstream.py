import pytest
import numpy as np
from pathlib import Path

from obspy import Trace, Stream, Inventory, UTCDateTime, read
# from obspy.core.tests.test_stream import TestStream

from PULSE.data.dictstream import DictStream
from PULSE.data.foldtrace import FoldTrace
from PULSE.test.example_data import (load_townsend_example,
                                  load_seattle_example,
                                  assert_common_trace)

class TestDictStream():

    # Class Variables - load in example data once!
    # Just be sure to use :meth:`~.(Dict)Stream.copy`
    # in the setup of each test
    # Stream from ObsPy default example (3 traces, 1 station)
    sm_st = read()
    # Stream from M4.3 near Port Townsend, WA in Oct 2023 (33 traces, 8 stations, 13 instruments)
    med_st, med_inv, townsend_cat = load_townsend_example(Path().cwd())
    # Stream from M2.1 in North Seattle, WA in Sept 2023 (395 traces, 109 stations, 133 instruments)
    lg_st, lg_inv, seattle_cat = load_seattle_example(Path().cwd())

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
        # Cleanup memory
        del st, ds1, ds2

    def test_iter(self):
        """Test suite for the __iter__ method of DictStream
        """        
        ds = DictStream(self.med_st.copy())
        assert hasattr(ds, '__iter__')
        for _ft in ds:
            assert isinstance(_ft, FoldTrace)
        del ds

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
        # Cleanup
        del ds, ds2
        
    def test_setitem(self):
        """Test suite for the __setitem__ method of DictStream
        """     
        # Setup 
        ds = DictStream(self.sm_st.copy())
        ds2 = ds.copy()
        # Make sure we're asserting st[0] is an ObsPy trace to start with
        assert isinstance(self.sm_st[0], Trace)
        ds2[2] = self.sm_st[0].copy()
        # Assert that setting the item converts into FoldTrace
        assert isinstance(ds2[2], FoldTrace)
        # Assert that the data are not changed (TODO: probably move to test extend?)
        np.testing.assert_array_equal(ds2[2], self.sm_st[0])
        # Assert that
        assert_common_trace(self.sm_st[0], ds2[2])

        # Test Error catches
        with pytest.raises(IndexError):
            ds[4] = self.sm_st[0]
        with pytest.raises(TypeError):
            ds[4] = 'abc'
        with pytest.raises(TypeError):
            ds[int] = self.sm_st[0]
        # Cleanup
        del ds, ds2

    def test_repr(self):
        # Setup
        ds = DictStream(self.sm_st.copy())
        ds2 = DictStream(self.med_st.copy())
        # Tests
        assert isinstance(ds.__repr__(), str)
        len0 = len(ds.__repr__())
        len1 = len(ds.__repr__(extended=True))
        assert len0 == len1
        len0 = len(ds2.__repr__())
        len1 = len(ds2.__repr__(extended=True))
        assert len0 < len1
        # Cleanup
        del ds, ds2

    def test_str(self):
        # Setup
        ds = DictStream(self.sm_st.copy())
        assert isinstance(ds.__str__(), str)
        assert ds.__str__() == 'PULSE.data.dictstream.DictStream'
        assert ds.__str__(short=True) == 'DictStream'
        # Cleanup
        del ds

    #######################
    ## SELECT TEST SUITE ##
    #######################
    def test_fnsearch(self):
        # Setup
        ds = DictStream(self.lg_st)
        # Simple search for all UW stations
        uwkey = ds.fnsearch(idstring='UW.*')
        assert isinstance(uwkey, set)
        assert all([_k[:2] == 'UW' for _k in uwkey])
        # Check ? wildcard behaviors - just get station UW.GNW broadband channels
        gnwkey = ds.fnsearch(idstring='UW.GNW..HH?')
        assert len(gnwkey) == 3
        assert all([_k[:-1] == 'UW.GNW..HH' for _k in gnwkey])
        # Check [] wildcard behaviors - get GNW and GMW SMA channels
        grwkey = ds.fnsearch(idstring='UW.G[NM]W..[EH]NZ')
        assert len(grwkey) == 2
        assert grwkey == set(['UW.GNW..ENZ','UW.GMW..HNZ'])
        # Check compound wildcard search
        wildkey = ds.fnsearch(idstring='*.???.*.[EH]H?')
        # Sanity check assert that there are data for this example
        assert len(wildkey) > 0
        for _k in wildkey:
            n,s,l,c = _k.split('.')
            assert len(s) == 3
            assert c[:-1] in ['EH','HH']

    def test_inverse_set(self):
        ## Setup
        ds = DictStream(self.lg_st)
        # Create key set
        uwkey = ds.fnsearch(idstring='UW.*')
        # Create inverse set
        ikey = ds.inverse_set(uwkey)
        ## Type Test
        assert isinstance(ikey, set)
        assert len(ikey) > 0
        assert len(uwkey) > 0
        assert len(ikey) + len(uwkey) == len(ds)
        ## Test Inverse Set
        # Assert no overlaps
        assert not any([_k in uwkey for _k in ikey])
        assert not any([_k in ikey for _k in uwkey])
        # Assert union is the full set of keys
        assert ikey.union(uwkey) == ds.traces.keys()
        # Assert intersection is an empty set
        assert ikey.intersection(uwkey) == set()

    def test_attrsearch(self):
        ## Setup
        ds = DictStream(self.lg_st)
        attrs = ['sampling_rate','npts','calib','delta']
        # Run tests on all individual attributes
        for ii, _ia in enumerate(attrs):
            valset = set([ft.stats[_ia] for ft in ds])
            for _val in valset:
                ikey = ds.attrsearch(**{_ia:_val})
                assert len(ikey) > 0
                # Assert that combinations match
                assert all([ds[_k].stats[_ia] == _val for _k in ikey])
                # Assert that inverse sets always mismatch
                assert not any([ds[_k].stats[_ia] == _val for _k in ds.inverse_set(ikey)])
                # Run tests on pairwise combinations
                for jj, _ja in enumerate(attrs):
                    if jj > ii:
                        for _jval in set([ft.stats[_ja] for ft in ds]):
                            kwargs = {_ia: _val, _ja: _jval}
                            ikey = ds.attrsearch(**kwargs)
                            # If no keys match, assert that no entries have both attribute values
                            if len(ikey) == 0:
                                assert not any([ds[_k].stats[_ia] == _val and ds[_k].stats[_ja] == _jval for _k in ds.traces.keys()])
                            # If at least one key matches, assert it has the
                            else:
                                assert all([ds[_k].stats[_ia] == _val and ds[_k].stats[_ja] == _jval for _k in ikey])
        # for other attrs raises
        type_scramble = {int: 'a', float: 'a', UTCDateTime: 2, str: 1}
        for _attr in ds[0].stats.keys():
            if _attr not in attrs:
                with pytest.raises(AttributeError):
                    ikey = ds.attrsearch(**{_attr: ds[0].stats[_attr]})
            else:
                other_val = type_scramble[type(ds[0].stats[_attr])]
                with pytest.raises(TypeError):
                    ikey = ds.attrsearch(**{_attr:other_val})

    def test_select(self):
        ## Setup
        ds = DictStream(self.lg_st)
        inv = self.lg_inv
        
        ## Test inventory Select
        inv_AM = inv.select(network='AM')
        dsAM = ds.select(inventory=inv_AM)
        assert all([ft.stats.network == 'AM' for ft in dsAM])

        ## Test id select
        dsid = ds.select(id = '*.G?W..?[HN][ZN]')
        # Test id select with implicit first position argument
        dsid2 = ds.select('*.G?W..?[HN][ZN]')
        # assert that both contain traces
        assert len(dsid) > 0
        assert len(dsid2) > 0
        # assert that the result is identical
        assert dsid == dsid2

        ## Test component select
        dsZ = ds.select(component='Z')
        assert all([ft.stats.component == 'Z' for ft in dsZ])
        assert dsZ == ds.select('*Z')

        ## Test network select
        dsUW = ds.select(network='UW')
        assert all([ft.stats.network == 'UW' for ft in dsUW])
        # assert same as id search
        assert dsUW == ds.select('UW*')
        
        ## Test station select + * wildcard
        dsG = ds.select(station='G*')
        assert all([ft.stats.station[0] == "G" for ft in dsG])
        assert dsG == ds.select('*.G*.*.*')
        
        ## Test location select + ? wildcard
        dsloc = ds.select(location='0?')
        assert not any([ft.stats.location == '' for ft in dsloc])
        assert all([ft.stats.location in ['00','01','02','03'] for ft in dsloc])
        assert dsloc == ds.select('*.*.0?.*')
        
        ## Test channel select + list wild
        dscha = ds.select(channel='E[HN]Z')
        assert all([ft.stats.channel in ['EHZ','ENZ'] for ft in dscha])
        
        ## Test sampling_rate
        ds200 = ds.select(sampling_rate=200)
        assert all([ft.stats.sampling_rate == 200 for ft in ds200])
        
    def test_split(self):
        # Setup
        # TODO: need example that has predictions
        ds = DictStream(self.lg_st)
        ## Network split
        for attr in set(ds[0].id_keys.keys()):
            uniques = {ft.stats.network for ft in ds}
            split_dict = ds.split(attr='network')
            assert uniques == set(split_dict.keys())
            for _val in uniques:
                assert all([ft.id_keys[attr] == _val for ft in split_dict[_val]])







#     def test_fnsearch(self):
#         ds = DictStream()

#     # def test_search(self):
#     #     # Setup
#     #     ds = DictStream(self.townsend_st.ctownsendy())
#  townsend #     ds2 = ds.search(['UW.P*..EN?*', 'UW.N*..EH?*'])

#     def test_split(self):
#         ds = DictStream(self.townsend_st.ctownsendy())
# townsend