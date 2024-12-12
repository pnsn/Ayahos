import unittest
from pathlib import Path

import numpy as np
from obspy import Stream

from PULSE.data.dictstream import DictStream
from PULSE.data.window import Window
from PULSE.util.header import WindowStats
from PULSE.test.example_data import load_townsend_example

class TestWindow(unittest.TestCase):

    ### TEST SETUP ###

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

        self.win_pert = Window(traces=self.sub_st.copy().select(channel='?N?'))
        # setup scruffed window  
        for _e, comp in enumerate(['Z','N','E']):
            ft = self.win_pert[comp]
            # Shift starttime: one unshifted, one req' petty, one req' interpolated
            ft.stats.starttime += _e*0.01*ft.stats.delta
            # Shift sampling rate: lower, on target, higher
            ft.stats.sampling_rate += (_e - 1)*0.8
            # Trim out some samples: short, long, on target
            perts = [-1, 1, 0]
            ft.trim(starttime=ft.stats.starttime+perts[_e]*0.5, endtime=ft.stats.endtime-perts[_e]*0.5, pad=True, fill_value=0.)
        # FIXME: QC these and re-apply test in :meth:`~.test_preprocess_component`
        self.expected_processing = {'Z': ['filter','detrend','resample','taper'],
                                    'N': ['filter','detrend','align_starttime','taper','trim'],
                                    'E': ['interpolate','filter','detrend','resample','taper','trim']}
        
        self.winx = Window(traces=self.sub_st.select(channel='?N?'))

    def tearDown(self):
        del self.sub_st
        del self.ds
        del self.sub_header
        del self.tflds
        del self.win

    ### __INIT__ ###

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

    ### __eq__ ###
    def test___eq__(self):
        w1 = Window(traces=self.sub_st.select(channel='?N?').copy())
        w2 = Window(traces=self.sub_st.select(channel='?N?').copy())
        # Test mutual equalities
        self.assertEqual(self.winx, w1)
        self.assertEqual(self.winx, w2)
        self.assertEqual(w1, w2)
        
        # Test change metadata
        w1.stats.secondary_components='12'
        self.assertNotEqual(self.winx, w1)
        self.assertEqual(self.winx, w2)
        # Test change traces
        w2.pop('E')
        self.assertNotEqual(self.winx, w2)


    ### PRIVATE (SUBROUTINE) METHODS TESTING ###

    def test__validate(self):
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
    
    def test__get_primary(self):
        # Assert that _get_primary returns the same as primary_id from source
        self.assertEqual(self.win._get_primary(), self.ds[self.sub_header['primary_id']])
        # Assert that empty window _get_primary returns none
        self.assertIsNone(Window()._get_primary())
        # Assert that window missing primary returns None
        self.win.pop('Z')
        self.assertIsNone(self.win._get_primary())


    def test__check_targets(self):
        self.win.stats.target_starttime += 0.1

        flds = {'starttime','sampling_rate','npts'}
        for ft in self.win:
            result = self.win._check_targets(ft.stats.component)
            self.assertEqual(flds, result)
        counter = 0
        for fld in flds:
            tmp_win = self.win.copy()
            tmp_win.stats[f'target_{fld}'] = self.win[0].stats[fld]
            tmp_flds = flds.copy()
            tmp_flds.remove(fld)
            counter += 1
            for ft in tmp_win:
                result = tmp_win._check_targets(ft.stats.component)
                self.assertEqual(tmp_flds, result)

    def test__check_starttime_alignment(self):
        for ft in self.win:
            result = self.win._check_starttime_alignment(ft.stats.component)
            self.assertTrue(result)
        self.win[0].stats.starttime += 0.0001
        for _e, ft in enumerate(self.win):
            result = self.win._check_starttime_alignment(ft.stats.component)
            if _e == 0:
                self.assertFalse(result)
            else:
                self.assertTrue(result)

    def test__get_nearest_starttime(self):
        # Test slight positive misalignment
        self.win.stats.target_starttime += 1.00001
        for _c in self.win.keys:
            nearest = self.win._get_nearest_starttime(_c)
            self.assertEqual(nearest, self.win[_c].stats.starttime + 0.00001)
        # Test slight negative misalignment with/without roundup
        self.win.stats.target_starttime -= 2.00002
        for _c in self.win.keys:
            nearest = self.win._get_nearest_starttime(_c)
            self.assertEqual(nearest, self.win[_c].stats.starttime + 0.01999)
            nearest = self.win._get_nearest_starttime(_c, roundup=False)
            self.assertEqual(nearest, self.win[_c].stats.starttime - 0.00001)

    def test__check_fvalid(self):
        # Test all passing case
        for _c in self.win.keys:
            self.assertTrue(self.win._check_fvalid(_c))
        # Test case where only secondary traces pass
        self.win.stats.target_starttime -= 6/50
        for _c in self.win.keys:
            if _c == 'Z':
                self.assertFalse(self.win._check_fvalid(_c))
            else:
                self.assertTrue(self.win._check_fvalid(_c))
        # Test case where none pass
        self.win.stats.target_starttime += 300
        for _c in self.win.keys:
            self.assertFalse(self.win._check_fvalid(_c))


    def test__get_order(self):
        self.assertEqual('ZEN', self.winx.order)
        self.assertEqual(self.winx.order, self.winx._get_order())

    ### PUBLIC METHODS TESTING ###
    def test_copy(self):
        w1 = self.winx.copy()
        w2 = self.winx.copy()
        self.assertEqual(self.winx, w1)
        self.assertEqual(self.winx, w2)
        self.assertEqual(w1, w2)
        # Assert that changing metadata doesn't back-propagate
        w1.stats.secondary_components='12'
        self.assertNotEqual(self.winx, w1)
        # Assert that dropping traces doesn't back propagate
        w2.pop('E')
        self.assertIn('E', self.winx.keys)
        self.assertNotIn('E', w2.keys)
        
    def test_preprocess_component(self):
        # Pretest
        for comp in self.win_pert.keys:
            self.assertNotEqual(self.win_pert._check_targets(comp), set([]))
        # Run preprocessing
        for comp in self.win_pert.keys:
            self.win_pert.preprocess_component(comp)
        # Check that all targets are met
        for ft in self.win_pert:
            self.assertEqual(ft.stats.starttime, self.win_pert.stats.target_starttime)
            self.assertEqual(ft.stats.sampling_rate, self.win_pert.stats.target_sampling_rate)
            self.assertEqual(ft.stats.npts, self.win_pert.stats.target_npts)

        # FIXME: Revitalize these processing log tests
        # for comp, expectation in self.expected_processing.items():
        #     ft = self.win_pert[comp]
        #     if len(expectation) != len(ft.stats.processing):
        #         breakpoint()
        #     self.assertEqual(len(expectation), len(ft.stats.processing))
        #     for _e, E in enumerate(expectation):
        #         self.assertIn(E, ft.stats.processing[_e])

    def test_preprocess_component_gappy(self):
        for ft in self.win_pert:
            ft.data = np.ma.MaskedArray(data=ft.data,
                                        mask=[False]*ft.count())
            ft.data.mask[2000:3000] = True
        for comp in self.win_pert.keys:
            # Assert corrections are needed
            self.assertNotEqual(self.win_pert._check_targets(comp), set([]))
            # Assert data are masked
            self.assertTrue(np.ma.is_masked(self.win_pert[comp].data))
            # Run process
            self.win_pert.preprocess_component(comp)
            # Assert all targets are now met
            result = self.win_pert._check_targets(comp)
            self.assertEqual(len(result), 0)
            # self.assertEqual(self.win_pert._check_targets(comp), set([]))
            # Assert that data are now continuous
            self.assertFalse(np.ma.is_masked(self.win_pert[comp].data))

    def test_preprocess_component_errors(self):
        # Test KeyError on comp
        for _arg in ['Q',1]:
            with self.assertRaises(KeyError):
                self.win_pert.preprocess_component(_arg)
        # Test Type Error on required
        for _arg in ['resample','trim']:
            for _val in ['a', ['a'], None]:
                inp = {_arg: _val}
                with self.assertRaises(TypeError):
                    self.win_pert.preprocess_component('Z', **inp)
        # Test TypeError on optional
        for _arg in ['filter','detrend','taper']:
            for _val in ['a', ['a']]:
                inp = {_arg: _val}
                with self.assertRaises(TypeError):
                    self.win_pert.preprocess_component('Z', **inp)
        # Test AttributeError on required kwargs
        for _arg in ['filter','detrend','resample']:
            inp = {_arg: {}}
            with self.assertRaises(AttributeError):
                self.win_pert.preprocess_component('Z', **inp)

    def test_fill_missing_traces(self):
        # Test no effect on passing window
        fill_win = self.win.copy()
        fill_win.fill_missing_traces()
        self.assertEqual(fill_win, self.win)

        winx = self.win.copy()
        # Remove North Trace
        winx.pop('N')
        # Assert North trace is gone
        self.assertNotIn('N', winx.keys)
        # Iterate across fill rule and aliases
        for rule in ['zeros','primary','secondary', 0, 1, 2]:
            winxN = winx.copy()
            # Apply fill rule to copy
            self.assertNotIn('N', winxN.keys)
            winxN.fill_missing_traces(rule=rule)
            # Assert N is now present
            self.assertIn('N', winxN.keys)

            # Assert N has the rule-appropriate data
            if rule in ['zeros',0]:
                self.assertTrue(all(winxN['N'].data == 0))
            elif rule in ['primary', 1]:
                np.testing.assert_array_equal(winxN['Z'].data, winxN['N'].data)
            elif rule in ['secondary', 2]:
                np.testing.assert_array_equal(winxN['E'].data, winxN['N'].data)

            # Assert N has 0 fold
            self.assertTrue(all(winxN['N'].fold == 0))

            # Assert donor still has 1 fold & clone has almost all the same metadata
            if rule in ['zeros','primary',0, 1]:
                self.assertTrue(all(winxN['Z'].fold == 1))
                for _k, _v in winxN['Z'].stats.items():
                    if _k != 'channel':
                        self.assertEqual(_v, winxN['N'].stats[_k])
                    else:
                        self.assertEqual(_v[:-1] + 'N', winxN['N'].stats[_k])
            else:
                self.assertTrue(all(winxN['E'].fold == 1))
                for _k, _v in winxN['E'].stats.items():
                    if _k != 'channel':
                        self.assertEqual(_v, winxN['N'].stats[_k])
                    else:
                        self.assertEqual(_v[:-1] + 'N', winxN['N'].stats[_k])
            
    def test_fill_missing_traces_secondary(self):
        self.winx.pop('N')
        # TEST1 Setup: E at sthresh
        winx1 = self.winx.copy()
        winx1['E'].trim(endtime=winx1.stats.target_endtime - 30)
        # Assert that 'E' still passes
        self.assertTrue(winx1._check_fvalid('E'))

        # Apply fill rule
        winx1.fill_missing_traces(rule=2)

        # Assert that 'N' is present
        self.assertIn('N', winx1.keys)
        # Assert N data matches E data
        np.testing.assert_array_equal(winx1['N'].data, winx1['E'].data)
        
        # TEST2 Setup: E below sthresh
        winx1 = self.winx.copy()
        # FIXME: If processing logging is reintroduced to FoldTrace - will need to correct this marker
        winx1['E'].trim(endtime=winx1.stats.target_endtime - 50)
        winx1['Z'].stats.processing.append('marker')
        # setup checks
        self.assertNotIn('N', winx1.keys)
        self.assertFalse(winx1._check_fvalid('E'))
        self.assertEqual(winx1['Z'].stats.processing, ['marker'])
        self.assertEqual(winx1['E'].stats.processing, [])

        # Apply rule
        winx1.fill_missing_traces(rule=2)
        self.assertIn('N', winx1.keys)
        for comp in 'EN':
            for _k, _v in winx1['Z'].stats.items():
                # Assert that stats are almost identical
                if _k != 'channel':
                    self.assertEqual(_v, winx1[comp].stats[_k])
                    # Assert that processing is empty on clone
                    if _k == 'processing':
                        self.assertEqual(_v, ['marker'])
                else:
                    self.assertEqual(_v[:-1]+comp, winx1[comp].stats[_k])
                # Assert clone data matches donor
                np.testing.assert_array_equal(winx1['Z'].data, winx1[comp].data)
                # Assert clone fold is 0
                self.assertTrue(all(winx1[comp].fold==0))
    
    def test_fill_missing_traces_errors(self):
        # setup precheck
        self.assertTrue(self.winx._check_fvalid('Z'))
        # Unapproved rule TypeError
        for _arg in ['a', 3, 1.1]:
            with self.assertRaises(ValueError):
                self.winx.fill_missing_traces(rule=_arg)

        # Insufficient data in primary component ValueError
        self.winx['Z'].trim(endtime = self.winx.stats.target_endtime - 50)
        # setup check
        self.assertFalse(self.winx._check_fvalid('Z'))
        with self.assertRaises(ValueError):
            self.winx.fill_missing_traces()

        # Missing primary component ValueError (inherited from _validate)
        self.winx.pop('Z')
        self.assertEqual(self.winx.stats.get_primary_component(), 'Z')
        self.assertNotIn('Z', self.winx.keys)
        with self.assertRaises(ValueError):
            self.winx.fill_missing_traces()
        

    def test_fill_missing_traces_mismatched(self):
        # TEST 1 SINGLE MISMATCHED SECONDARY
        self.winx.stats.secondary_components='E1'
        self.assertEqual(set('ZNE'), set(self.winx.keys))
        self.winx.fill_missing_traces()
        self.assertEqual(set('ZNE1'), set(self.winx.keys))
        self.assertTrue(all(self.winx['1'].fold==0))

    
    def test_to_npy_tensor(self):
        x = self.winx.to_npy_tensor()
        # Assert shape is correct
        self.assertEqual(x.shape, (3, 15000))
        order = self.winx.order
        # Assert data are in the correct order and identical
        for _e, _c in enumerate(order):
            np.testing.assert_array_equal(self.winx[_c].data, x[_e,:])
        # Change order
        x = self.winx.to_npy_tensor(components='ENZ')
        self.assertNotEqual('ENZ', order)
        self.assertEqual(x.shape, (3, 15000))
        # Assert data are in the new specified order
        for _e, _c in enumerate('ENZ'):
            np.testing.assert_array_equal(self.winx[_c].data, x[_e, :])

        # ERROR TESTS
        # AttributeError from components input without __iter__
        with self.assertRaises(AttributeError):
            self.winx.to_npy_tensor(components=1)
        # ValueError from mismatching iterable
        for _arg in ['a','123','ZE1',[1,2,3]]:
            with self.assertRaises(ValueError):
                self.winx.to_npy_tensor(components=_arg)
        # AttributeError from having failing traces
        with self.assertRaises(AttributeError):
            self.win_pert.to_npy_tensor()
   
    def test_preprocess(self):
        wp1 = self.win_pert.copy()
        wp2 = self.win_pert.copy()



    # def test_preprocess_component_errors(self):


    #     # Test wrong component error
    #     for _arg in ['Q', 1]:
    #         with self.assertRaises(KeyError):
    #             self.win_pert.preprocess_component(_arg)
    #     # Test non-dict required errors
    #     for _arg in [['a'], 1]:
    #         for _fld in ['align','resample','trim']
    #         with self.assertRaises
        # # Test wrong format for kwarg holder (combinations)
        # for _arg in ['aaa', ['foo','bar','baz'], 1, 1.]:
        #     for _fld in ['filter','detrend','resample','taper','trim']:
        #         kwarg = {_fld: _arg}
        #         with self.assertRaises(TypeError):
        #             self.win.preprocess_component('Z', **kwarg)
        # for badkwarg in [{'filter': {'method': 'bandpass'}, 'detrend': {'type': 'linear'}}]:
        #     with self.assertRaises(AttributeError):
        #         self.win.preprocess_component('Z', **badkwarg)          

        # # Test large, but close perturbation
        # for _e, _pert in enumerate([0.005, 1.005]):
        #     self.win['Z'].stats.starttime += _pert
        #     self.assertNotEqual(self.win['Z'].stats.starttime)

    def test_preprocess(self):
        self.assertTrue(False)

    def test_collapse_fold(self):
        self.assertTrue(False)


    

    

    # def test_sync_sampling(self):
    #     # Misalign Z component by a very small amount
    #     self.win['Z'].stats.starttime -= 0.00001
    #     # Misalign N component by half a sample
    #     self.win['N'].stats.starttime += 0.005
    #     # Check that both fail alignment test
    #     for _c in ['Z','N']:
    #         self.assertFalse(self.win._check_starttime_alignment(_c))
    #     # Create copy to sync
    #     win2 = self.win.copy()
        
    #     # Check no-change case
    #     win2.sync_sampling('E')
    #     self.assertEqual(win2['E'], self.win['E'])
        
    #     # Check minor adjustment case
    #     win2.sync_sampling('Z')
    #     # Assert that the data have not changed
    #     np.testing.assert_array_equal(win2['Z'].data, self.win['Z'].data)
    #     # Assert that the metadata have changed
    #     self.assertNotEqual(win2['Z'], self.win['Z'])
    #     # Assert that the shifted starttime aligns with target
    #     self.assertTrue(win2._check_starttime_alignment('Z'))
    #     # Assert no change in processing
    #     self.assertEqual(win2['Z'].stats.processing,
    #                      self.win['Z'].stats.processing)

    #     # Check interpolation case
    #     win2.sync_sampling('N')
    #     # Assert that the interpolated starttime is at or after the original starttime
    #     self.assertLessEqual(self.win['N'].stats.starttime,
    #                          win2['N'].stats.starttime)
    #     # Assert that the interpolated starttime aligns with target
    #     self.assertTrue(win2._check_starttime_alignment('N'))
    #     # Assert that processing shows interpolation
    #     self.assertIn('interpolate',
    #                   win2['N'].stats.processing[-1])
        
    #     # Assert compatability errors
    #     with self.assertRaises(TypeError):
    #         win2.sync_sampling('N', sample_tol='abc')
    #     for val in [-1., 2., 0.50000001]:
    #         with self.assertRaises(ValueError):
    #             win2.sync_sampling('N', sample_tol=val)


    


    