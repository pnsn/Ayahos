import unittest
from pathlib import Path

from obspy import read, UTCDateTime, Trace, Stream

from PULSE.data.foldtrace import FoldTrace
from PULSE.data.dictstream import DictStream
from PULSE.data.window import Window
from PULSE.util.header import WindowStats
from PULSE.test.example_data import load_townsend_example

class TestWindow(unittest.TestCase):

    def setUp(self):
        st, _, _ = load_townsend_example(Path().cwd())
        self.ds = DictStream(st.select(station='MCW').copy())
        self.win = Window()

    def tearDown(self):
        del self.ds
        # del self.win

    def test_init(self):
        self.assertIsInstance(self.win, Stream)
        self.assertIsInstance(self.win, DictStream)
        self.assertIsInstance(self.win.stats, WindowStats)
        self.assertEqual(self.win.supported_keys, ['comp'])
        self.assertEqual(self.win.key_attr, 'comp')
    
    def test_init_prescribed(self):
        window = Window(traces=self.ds.select(id='*.EN?'))
