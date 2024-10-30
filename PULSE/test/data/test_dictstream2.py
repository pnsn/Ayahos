import unittest
from pathlib import Path

from obspy import Stream, read, Trace, UTCDateTime

from PULSE.util.header import MLStats
from PULSE.data.foldtrace import FoldTrace
from PULSE.data.dictstream import DictStream
from PULSE.test.example_data import load_townsend_example


class TestDictStream(unittest.TestCase):

    def setUp(self):
        st, _, _ = load_townsend_example(Path().cwd())
        self.st = st
        self.ft = FoldTrace(st[0])
        self.ds = DictStream(st)
        self.id_keys = MLStats().get_id_keys().keys()

    def tearDown(self):
        del self.st
        del self.ds

    def test_init(self):
        self.assertIsInstance(self.ds, Stream)
        self.assertIsInstance(self.ds, DictStream)
        self.assertIsInstance(self.ds.traces, dict)
        self.assertEqual(self.id_keys, self.ds.supported_keys)
        self.assertEqual(self.ds.key_attr, 'id')
        id_set = set([_ft.id for _ft in self.ds])
        self.assertEqual(id_set, self.ds.traces.keys())

    def test_id_subset(self):
        for id in ['UW*', 'UW.*.*.*', 'UW.*.*.???']:
            subset = self.ds.id_subset(id=id)
            self.assertIsInstance(subset, set)
            self.assertEqual(len(subset), 21)
            self.assertTrue(subset <= self.ds.traces.keys())
        subset = self.ds.id_subset(id='*.??[NZ]')
        self.assertIsInstance(subset, set)
        self.assertEqual(len(subset), 19)
        for id in subset:
            self.assertTrue(id[-1] in 'NZ')







