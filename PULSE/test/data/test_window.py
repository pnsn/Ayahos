import unittest
from pathlib import Path

from obspy import read, UTCDateTime, Trace, Stream

from PULSE.data.foldtrace import FoldTrace
from PULSE.data.dictstream import DictStream
from PULSE.data.window import Window
from PULSE.util.header import WindowStats
from PULSE.test.example_data import load_townsend_example

class TestWindow(unittest.TestCase):
    st, _, _ = load_townsend_example(Path().cwd())

    def setUp(self):
        self.sub_st = self.st.select(station='MCW').copy()
        self.ds = DictStream(self.sub_st.copy())
        self.sub_header={'primary_id': 'UW.MCW.--.ENZ',
                         'secondary_components': 'EN',
                         'target_starttime': self.st[0].stats.starttime,
                         'target_sampling_rate': 50.,
                         'target_npts': 100}
        self.win = Window(traces=DictStream(self.sub_st.select(channel='?N?')),
                          header=self.sub_header)
    def tearDown(self):
        del self.sub_st
        del self.ds
        del self.sub_header

    def test_init(self):
        win = Window()
        # Assert types
        self.assertIsInstance(win, Stream)
        self.assertIsInstance(win, DictStream)
        self.assertIsInstance(win.stats, WindowStats)
        # Assert key_attr is set to component
        self.assertEqual(win.key_attr, 'component')
    
    def test_init_traces(self):
        # Create window with data
        self.assertIsInstance(self.win, Window)
        # Assert that all foldtraces in Window are in the source dictstream
        for ft in self.win:
            self.assertIn(ft, self.ds)

    def test_validate(self):
        # Primary not present error
        with self.assertRaises(ValueError):
            Window(traces=self.ds.select(id='*.EN[EN]'),
                         header=self.sub_header)
        # Not all traces are FoldTrace Error
        with self.assertRaises(ValueError):
            Window(traces=self.sub_st,
                         header=self.sub_header)
        # Not all traces are from the same instrument
        ds = DictStream(self.st.copy().select(channel='??Z'))
        with self.assertRaises(ValueError):
            Window(traces=ds,header=self.sub_header)
        
        # Too many traces from same instrument
        with self.assertRaises(ValueError):
            Window(traces=self.ds, header=self.sub_header)
    
    def test_get_primary(self):
        # Assert that get_primary returns the same as primary_id from source
        self.assertEqual(self.win.get_primary(), self.ds[self.sub_header['primary_id']])
        # Assert that empty window get_primary returns none
        self.assertIsNone(Window().get_primary())
        # Assert that window missing primary returns None
        self.win.pop('Z')
        self.assertIsNone(self.win.get_primary())

    # def test_check_targets(self):
    #     win = 


    