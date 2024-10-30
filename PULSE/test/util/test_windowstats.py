import unittest

from obspy.core.util.attribdict import AttribDict
from PULSE.util.header import WindowStats


class TestWindowStats(unittest.TestCase):

    def setUp(self):
        self.stats = WindowStats()

    def tearDown(self):
        del self.stats

    def test_init(self):
        self.assertIsInstance(self.stats, dict)
        self.assertIsInstance(self.stats, AttribDict)
        self.assertIsInstance(self.stats, WindowStats)
        # Assert default values are in place
        for _k, _v in WindowStats.defaults.items():
            self.assertEqual(self.stats[_k], _v)
            
