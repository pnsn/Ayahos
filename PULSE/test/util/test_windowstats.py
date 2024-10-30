import unittest

from obspy import UTCDateTime
from obspy.core.util.attribdict import AttribDict

from PULSE.util.header import WindowStats


class TestWindowStats(unittest.TestCase):

    def setUp(self):
        self.stats = WindowStats()
        self.alt_good_hdr = {'primary_id': 'UW.GPW..HHZ',
                             'secondary_components': 'NE',
                             'pthresh': 0.5,
                             'sthresh': 0.2,
                             'target_starttime': UTCDateTime(50),
                             'target_sampling_rate': 100.,
                             'target_npts': 3001}
        self.alt_bad_hdr = {'primary_id': 1,
                            'secondary_components': 1,
                            'pthresh': 1.1,
                            'sthresh': -0.2,
                            'target_starttime': 2,
                            'target_npts': 'a'}
    def tearDown(self):
        del self.stats

    def test_init(self):
        self.assertIsInstance(self.stats, AttribDict)
        self.assertIsInstance(self.stats, WindowStats)
        # Assert default values are in place
        for _k, _v in WindowStats.defaults.items():
            self.assertEqual(self.stats[_k], _v)

    
    def test_init_prescribed(self):
        stats = WindowStats(header=self.alt_good_hdr)
        self.assertIsInstance(stats, WindowStats)
        for _k, _v in self.alt_good_hdr.items():
            self.assertEqual(stats[_k], _v)
        self.assertEqual(stats.target_endtime, UTCDateTime(50) + 30)
        for _k, _v in self.alt_bad_hdr.items():
            print(_k)
            with self.assertRaises(ValueError):
                WindowStats(header={_k: _v})
        with self.assertRaises(KeyError):
            WindowStats(header={'target_endtime': UTCDateTime()})
    
    def test_get_primary_component(self):
        self.assertIsNone(self.stats.get_primary_component())
        stats = WindowStats(header=self.alt_good_hdr)
        self.assertEqual(stats.get_primary_component(), 'Z')
        stats.primary_id = 'UW.GPW..HHN.EQTransformer.pnw'
        self.assertEqual(stats.get_primary_component(), 'N')

    def test_get_secondary_ids(self):
        stats = WindowStats(header=self.alt_good_hdr)
        self.assertEqual(stats.get_secondary_ids(),
                         ['UW.GPW..HHN','UW.GPW..HHE'])
        stats.primary_id += '.EQTransformer.pnw'
        self.assertEqual(stats.get_secondary_ids(),
                         ['UW.GPW..HHN.EQTransformer.pnw',
                          'UW.GPW..HHE.EQTransformer.pnw'])
    

