import unittest
from pathlib import Path

import numpy as np
from obspy import Stream

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
                         'target_starttime': self.sub_st[0].stats.starttime,
                         'target_sampling_rate': 50.,
                         'target_npts': 100}
        self.win = Window(traces=self.sub_st.select(channel='?N?'),
                          header=self.sub_header)
        self.tflds = {'starttime','npts','sampling_rate'}

    def tearDown(self):
        del self.sub_st
        del self.ds
        del self.sub_header
        del self.tflds
        del self.win

    def test__init__(self):
        win = Window()
        # Assert types
        self.assertIsInstance(win, Stream)
        self.assertIsInstance(win, DictStream)
        self.assertIsInstance(win.stats, WindowStats)
        # Assert key_attr is set to component
        self.assertEqual(win.key_attr, 'component')
    
    def test___init___traces(self):
        # Assert type
        self.assertIsInstance(self.win, Window)
        # Assert that all foldtraces in Window are in the source dictstream
        for ft in self.win:
            self.assertIn(ft, self.ds)

    def test___init___primary_component(self):
        # create window with a specified primary component
        for _c in ['E','N','Z']:
            win = Window(traces=self.sub_st.copy().select(channel='?N?'),
                         primary_component=_c)
            # Assert the component is in the keys
            self.assertIn(_c, win.keys)
            # Assert the component matches the primary component
            self.assertEqual(_c, win.stats.primary_id[-1])
            # Assert that the primary matches the primary component entry
            self.assertEqual(win[_c], win.primary)
            del win

        # Test integer inputs
        for _c in range(10):
            tr = self.sub_st[0].copy()
            tr.stats.channel = tr.stats.channel[:-1] + f'{_c}'
            win = Window(traces=[tr], primary_component=_c)
            self.assertIn(str(_c), win.keys)
            self.assertEqual(str(_c), win.stats.primary_id[-1])
            self.assertEqual(win[str(_c)], win.primary)
            del win
        # Raise error tests
        for pcpt in [-1, 'abc']:
            with self.assertRaises(ValueError):
                win = Window(traces=self.sub_st.copy().select(channel='?N?'),
                            primary_component=pcpt)
        for pcpt in [1.1, bool]:
            with self.assertRaises(TypeError):
                win = Window(traces=self.sub_st.copy().select(channel='?N?'),
                            primary_component=pcpt)

    def test___init___autoheader(self):
        win = Window(traces=self.sub_st.select(channel='?N[ZE]'))
        # Assert that Z component is present
        self.assertIn('Z', win.keys)
        # Assert that Z component is the primary
        self.assertEqual('Z', win.stats.primary_id[-1])
        # Assert that Z component
        tp = win.primary
        # Assert that targets are all inherited from primary
        for fld in self.tflds:
            self.assertEqual(win.stats[f'target_{fld}'], tp.stats[fld])
        # Assert that secondary components are present
        for _k in win.keys:
            if _k != 'Z':
                self.assertIn(_k, win.stats.secondary_components)

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


    def test_check_targets(self):
        self.win.stats.target_starttime += 0.1

        flds = {'starttime','sampling_rate','npts'}
        for ft in self.win:
            result = self.win.check_targets(ft.stats.component)
            self.assertEqual(flds, result)
        counter = 0
        for fld in flds:
            tmp_win = self.win.copy()
            tmp_win.stats[f'target_{fld}'] = self.win[0].stats[fld]
            tmp_flds = flds.copy()
            tmp_flds.remove(fld)
            counter += 1
            for ft in tmp_win:
                result = tmp_win.check_targets(ft.stats.component)
                self.assertEqual(tmp_flds, result)

    def test_check_starttime_alignment(self):
        for ft in self.win:
            result = self.win.check_starttime_alignment(ft.stats.component)
            self.assertTrue(result)
        self.win[0].stats.starttime += 0.0001
        for _e, ft in enumerate(self.win):
            result = self.win.check_starttime_alignment(ft.stats.component)
            if _e == 0:
                self.assertFalse(result)
            else:
                self.assertTrue(result)

    def test_get_nearest_starttime(self):
        # Test slight positive misalignment
        self.win.stats.target_starttime += 1.00001
        for _c in self.win.keys:
            nearest = self.win.get_nearest_starttime(_c)
            self.assertEqual(nearest, self.win[_c].stats.starttime + 0.00001)
        # Test slight negative misalignment with/without roundup
        self.win.stats.target_starttime -= 2.00002
        for _c in self.win.keys:
            nearest = self.win.get_nearest_starttime(_c)
            self.assertEqual(nearest, self.win[_c].stats.starttime + 0.01999)
            nearest = self.win.get_nearest_starttime(_c, roundup=False)
            self.assertEqual(nearest, self.win[_c].stats.starttime - 0.00001)

    def test_check_fvalid(self):
        # Test all passing case
        for _c in self.win.keys:
            self.assertTrue(self.win.check_fvalid(_c))
        # Test case where only secondary traces pass
        self.win.stats.target_starttime -= 6/50
        for _c in self.win.keys:
            if _c == 'Z':
                self.assertFalse(self.win.check_fvalid(_c))
            else:
                self.assertTrue(self.win.check_fvalid(_c))
        # Test case where none pass
        self.win.stats.target_starttime += 300
        for _c in self.win.keys:
            self.assertFalse(self.win.check_fvalid(_c))
    
    def test_sync_samplign(self):
        # Misalign Z component by a very small amount
        self.win['Z'].stats.starttime -= 0.00001
        # Misalign N component by half a sample
        self.win['N'].stats.starttime += 0.005
        # Check that both fail alignment test
        for _c in ['Z','N']:
            self.assertFalse(self.win.check_starttime_alignment(_c))
        # Create copy to sync
        win2 = self.win.copy()
        
        # Check no-change case
        win2.sync_sampling('E')
        self.assertEqual(win2['E'], self.win['E'])
        
        # Check minor adjustment case
        win2.sync_sampling('Z')
        # Assert that the data have not changed
        np.testing.assert_array_equal(win2['Z'].data, self.win['Z'].data)
        # Assert that the metadata have changed
        self.assertNotEqual(win2['Z'], self.win['Z'])
        # Assert that the shifted starttime aligns with target
        self.assertTrue(win2.check_starttime_alignment('Z'))
        # Assert no change in processing
        self.assertEqual(win2['Z'].stats.processing,
                         self.win['Z'].stats.processing)

        # Check interpolation case
        win2.sync_sampling('N')
        # Assert that the interpolated starttime is at or after the original starttime
        self.assertLessEqual(self.win['N'].stats.starttime,
                             win2['N'].stats.starttime)
        # Assert that the interpolated starttime aligns with target
        self.assertTrue(win2.check_starttime_alignment('N'))
        # Assert that processing shows interpolation
        self.assertIn('interpolate',
                      win2['N'].stats.processing[-1])
        
        # Assert compatability errors
        with self.assertRaises(TypeError):
            win2.sync_sampling('N', sample_tol='abc')
        for val in [-1., 2., 0.50000001]:
            with self.assertRaises(ValueError):
                win2.sync_sampling('N', sample_tol=val)


    


    