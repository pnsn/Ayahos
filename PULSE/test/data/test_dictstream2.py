import unittest, fnmatch
from pathlib import Path

from obspy import Stream, read, Trace, UTCDateTime

from PULSE.util.header import MLStats
from PULSE.data.foldtrace import FoldTrace
from PULSE.data.dictstream import DictStream
from PULSE.test.example_data import (load_townsend_example,
                                     assert_common_trace,
                                     load_seattle_example)


class TestDictStream(unittest.TestCase):
    # Load waveform data once
    sm_st = read()
    md_st, md_inv = load_townsend_example(Path().cwd())[:2]
    lg_st, lg_inv = load_seattle_example(Path().cwd())[:2]


    def setUp(self):
        self.ft = FoldTrace(self.md_st[0].copy())
        self.ds0 = DictStream()
        self.sm_ds = DictStream(self.sm_st.copy())
        self.md_ds = DictStream(self.md_st.copy())
        self.id_keys = MLStats().get_id_keys().keys()

    def tearDown(self):
        del self.ft
        del self.ds0
        del self.sm_ds
        del self.md_ds
        del self.id_keys

    ## INIT TESTS ##

    def test_init(self):
        self.assertIsInstance(self.sm_ds, Stream)
        self.assertIsInstance(self.sm_ds, DictStream)
        self.assertIsInstance(self.sm_ds.traces, dict)
        self.assertEqual(self.id_keys, self.sm_ds.supported_keys)
        self.assertEqual(self.sm_ds.key_attr, 'id')
        id_set = set([_ft.id for _ft in self.sm_ds])
        self.assertEqual(id_set, self.sm_ds.traces.keys())

    def test_init_key_attr(self):
        # Run tests on empty DictStream to
        # just test the interchangeability of the key_attr attribute
        for key in self.id_keys:
            ds = DictStream(key_attr=key)
            self.assertIsInstance(ds , DictStream)
            self.assertEqual(ds.key_attr, key)
        with self.assertRaises(KeyError):
            DictStream(key_attr='abc')

    ## EXTEND TESTS ##

    def test_add_trace(self):
        # Prove initial state
        self.assertIsInstance(self.sm_st[0], Trace)
        self.assertNotIsInstance(self.sm_st[0], FoldTrace)
        self.assertEqual(len(self.ds0), 0)
        # Extend
        self.ds0._add_trace(self.sm_st[0].copy())
        # Test length
        self.assertEqual(len(self.ds0), 1)
        # Test not-equal
        self.assertNotEqual(self.ds0[0], self.sm_st[0])
        # Test conversion from Trace to FoldTrace
        self.assertIsInstance(self.ds0[0], FoldTrace)
        # Assert that ID is the same as before
        self.assertEqual(set([self.sm_st[0].id]), self.ds0.traces.keys())

    def test_add_trace_foldtrace(self):
        self.ds0._add_trace(self.ft)
        self.assertEqual(self.ds0[0], self.ft)
        self.assertEqual(len(self.ds0), 1)

    def test_extend_stream(self):
        # Prove initial state
        self.assertIsInstance(self.sm_st, Stream)
        self.assertNotIsInstance(self.sm_st, DictStream)
        self.assertTrue(all(isinstance(tr, Trace) for tr in self.sm_st))
        self.assertFalse(any(isinstance(tr, FoldTrace) for tr in self.sm_st))
        self.assertEqual(len(self.ds0), 0)
        # Extend
        self.ds0.extend(self.sm_st.copy())
        # Test length
        self.assertEqual(len(self.ds0), 3)
        # Test conversion to FoldTrace
        self.assertTrue(all(isinstance(_ft, FoldTrace) for _ft in self.ds0))
        # Test tha IDs are preserved
        self.assertEqual({tr.id for tr in self.sm_st}, self.ds0.traces.keys())

    def test_add_trace_errors(self):
        with self.assertRaises(TypeError):
            self.ds0._add_trace('abc')

    def test_extend_errors(self):
        with self.assertRaises(TypeError):
            self.ds0.extend('abc')
        with self.assertRaises(SyntaxError):
            self.ds0.extend(self.ft, kwarg=2)

    ## PROPERTY TESTS ##
    def test_get_keys(self):
        self.assertEqual(self.sm_ds.get_keys(), list(self.sm_ds.traces.keys()))
        self.assertEqual(self.sm_ds.keys, list(self.sm_ds.traces.keys()))
        self.assertIsInstance(self.sm_ds.keys, list)
        self.assertTrue(all(isinstance(key, str) for key in self.sm_ds.keys))

    def test_get_values(self):
        self.assertEqual(self.sm_ds.get_values(), list(self.sm_ds.traces.values()))
        self.assertEqual(self.sm_ds.values, list(self.sm_ds.traces.values()))
        self.assertIsInstance(self.sm_ds.values, list)
        self.assertTrue(all(isinstance(_ft, FoldTrace) for _ft in self.sm_ds.values))


    ## DUNDER TESTS ##

    def test_eq(self):
        ds1 = DictStream(self.sm_st[::-1])
        # assert indexing does not return the same
        self.assertFalse(all(self.sm_ds[_e] == ds1[_e] for _e in range(3)))
        self.assertEqual(sum(self.sm_ds[_e] == ds1[_e] for _e in range(3)), 1)
        # assert all keys are the same
        self.assertEqual(self.sm_ds.traces.keys(), ds1.traces.keys())
        # assert that the traces attribute is the same
        self.assertEqual(self.sm_ds.traces, ds1.traces)
        # Finally, test equal operator for DictStream
        self.assertEqual(self.sm_ds, ds1)
        self.assertTrue(self.sm_ds == ds1)

    def test_iter(self):
        self.assertTrue(hasattr(self.sm_ds, '__iter__'))
        for _ft in self.sm_ds:
            self.assertIsInstance(_ft, FoldTrace)

    def test_getitem(self):
        # Indexing
        self.assertEqual(self.sm_ds[0], self.sm_ds['BW.RJOB..EHZ'])
        assert_common_trace(self.sm_ds[0], read()[0])
        # Slicing
        self.assertEqual(self.sm_ds[:2], self.sm_ds[['BW.RJOB..EHZ','BW.RJOB..EHN']])
        # Errors
        with self.assertRaises(KeyError):
            self.sm_ds['a']
        with self.assertRaises(ValueError):
            self.sm_ds[20]
        with self.assertRaises(IndexError):
            self.sm_ds[['BW.RJOB..EHZ',1]]
        with self.assertRaises(TypeError):
            self.sm_ds[1.1]

    def test_setitem(self):
        with self.assertRaises(AttributeError):
            self.ds0[0] = self.sm_st[0].copy()
        # Implicit Key
        self.ds0.extend(self.sm_st[0].copy())
        ds0 = self.ds0.copy()
        ds0[0] = self.sm_st[0].copy().filter('lowpass', freq=20)
        self.assertNotEqual(self.ds0, ds0)
        # Explicit Key
        ds0 = self.ds0.copy()
        ds0['BW.RJOB..EHZ'] = self.sm_st[0].copy().filter('lowpass', freq=20)
        self.assertNotEqual(self.ds0, ds0)
        # Mismatched implicit key
        with self.assertRaises(KeyError):
            self.ds0[0] = self.sm_st[1].copy()
        # Mismatched explicit key
        with self.assertRaises(KeyError):
            self.ds0['BW.RJOB..EHZ'] = self.sm_ds['BW.RJOB.EHN']

    def test_split_on(self):
        # Iterate across all supported keys
        for id_key in DictStream().supported_keys:
            # Run split
            split = self.md_ds.split_on(id_key=id_key)
            # Assert split result type
            self.assertIsInstance(split, dict)
            # Assert that all keys are consistent 
            checkset = {_ft.id_keys[id_key] for _ft in self.md_ds}
            self.assertEqual(checkset, split.keys())
            # Assert that the number of traces split equals the original number
            total = sum([len(_v) for _v in split.values()])
            self.assertEqual(total, len(self.md_ds))
            # Check contents
            for _k, _v in split.items():
                # Assert type
                self.assertIsInstance(_v, DictStream)
                # Assert not-empty
                self.assertGreater(len(_v), 0)
                # Check individual traces
                for _ft in _v:
                    # Check type
                    self.assertIsInstance(_ft, FoldTrace)
                    # Check that key matches
                    self.assertEqual(_ft.id_keys[id_key], _k)

    ###################################
    ## SET-LOGIC SEARCH METHOD TESTS ##
    ###################################

    def test_match_inventory(self):
        with self.assertRaises(NotImplementedError):
            self.md_ds._match_inventory(self.md_inv)
        with self.assertRaises(TypeError):
            self.md_ds._match_inventory('abc')

    def test_match_id_keys(self):
        for _k, _v in self.ft.id_keys.items():
            subset = self.md_ds._match_id_keys(id_key=_k, pat=_v)
            # Assert subset is type set
            self.assertIsInstance(subset, set)
            # Assert is a subset of traces.keys
            self.assertTrue(subset <= self.md_ds.traces.keys())
            # Assert that all keys in subset match _v
            for _k2 in subset:
                self.assertIsInstance(self.md_ds[_k2], FoldTrace)
                self.assertEqual(self.md_ds[_k2].id_keys[_k], _v)
        with self.assertRaises(KeyError):
            self.md_ds._match_id_keys(id_key='abc', pat='*')
        with self.assertRaises(TypeError):
            self.md_ds._match_id_keys(id_key='site', pat=1)
    
    def test_match_id_keys_wild(self):
        for _k, _v in self.ft.id_keys.items():
            # ALTER _v
            _v = _v[:-1] + '?'
            match_set = self.md_ds._match_id_keys(id_key=_k, pat=_v)
            # Assert match_set is type set
            self.assertIsInstance(match_set, set)
            # Assert is a match_set of traces.keys
            self.assertTrue(match_set <= self.md_ds.traces.keys())
            # Assert that all keys in match_set match _v
            for _k2 in match_set:
                self.assertIsInstance(self.md_ds[_k2], FoldTrace)
                self.assertTrue(fnmatch.fnmatch(self.md_ds[_k2].id_keys[_k], _v))

    def test_match_id_keys_subset(self):
        subset = set(self.md_ds.keys[:5])
        fullset = set(self.md_ds.keys)
        # Make sure example trace is in set
        self.assertTrue(self.ft.id in subset)
        for _k, _v in self.ft.id_keys.items():
            # MAKE IT WILD
            _v = _v[:-1] + '?'
            # Create match_sets with different initial subsets
            match_set0 = self.md_ds._match_id_keys(id_key=_k, pat=_v)
            match_set1 = self.md_ds._match_id_keys(id_key=_k, pat=_v, subset=fullset)
            match_set2 = self.md_ds._match_id_keys(id_key=_k, pat=_v, subset=subset)
            # Assert that subset=None and subset=fullset give same result
            self.assertEqual(match_set0, match_set1)
            # Assert that subset=subset always is a subset default initialization's output
            self.assertTrue(match_set2 <= match_set0)
            # Assert taht all keys in match_set2 match _v
            for _k2 in match_set2:
                self.assertTrue(fnmatch.fnmatch(self.md_ds[_k2].id_keys[_k], _v))

    # def test_match_stats(self):
    #     # Setup
    #     lg_ds = DictStream(self.lg_st)
    #     # Add non 1. valued calib
    #     for _ft in lg_ds[::5]:
    #         _ft.stats.calib = 2.
    #     # Set individual 
    #     test_vals = {'npts': 4001,'sampling_rate': 100., 'calib': 2., 'delta': 1./200.}
    #     for _k, _v in test_vals:
    #     npts_set = lg_ds._match_stats(npts=4001)
    #     self.assertTrue(all(lg_ds[_k].stats.npts == 4001 for _k in npts_set))
    #     for _k, _v in tes




    # def test_inverse_set


        # String List raises error 




    # def test_id_subset(self):
    #     for id in ['UW*', 'UW.*.*.*', 'UW.*.*.???']:
    #         subset = self.sm_ds.id_subset(id=id)
    #         self.assertIsInstance(subset, set)
    #         self.assertEqual(len(subset), 21)
    #         self.assertTrue(subset <= self.sm_ds.traces.keys())
    #     subset = self.sm_ds.id_subset(id='*.??[NZ]')
    #     self.assertIsInstance(subset, set)
    #     self.assertEqual(len(subset), 19)
    #     for id in subset:
    #         self.assertTrue(id[-1] in 'NZ')

    # def test_match_id_keys(self):
    #     for id_key in self.id_keys:
    #         subset = self.sm_ds.id_keys_subset(self, id_key, '*')
    #         self.assertIsInstance()







