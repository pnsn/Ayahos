import unittest
from pathlib import Path
from collections import deque

from PULSE.util.header import WindowStats
from PULSE.mod.base import BaseMod
from PULSE.mod.windower import WindMod
from PULSE.data.dictstream import DictStream
from PULSE.test.example_data import load_seattle_example

class TestWindMod(unittest.TestCase):
    st, _, _ = load_seattle_example(Path().cwd())

    def setUp(self):
        self.ds = DictStream(self.st.copy())
        self.instruments = self.ds.split_on().keys()
        self.test_mod = WindMod(target_npts=1000,
                                target_sampling_rate=100,
                                overlap_npts=200,
                                name=None)

    
    def tearDown(self):
        del self.ds
        del self.test_mod
        del self.instruments

    def test___init__(self):
        mod = WindMod()
        # Test types & presets
        self.assertIsInstance(mod, BaseMod)
        self.assertIsInstance(mod, WindMod)
        self.assertEqual(mod._input_types, [DictStream])
        self.assertIsInstance(mod.index, dict)
        self.assertIsInstance(mod.output, deque)
        self.assertIsInstance(mod.window_stats, WindowStats)
        self.assertEqual(mod.window_stats.pthresh, 0.9)
        self.assertEqual(mod.window_stats.sthresh, 0.8)
        self.assertEqual(mod.window_stats.target_npts, 6000)
        self.assertEqual(mod.window_stats.target_sampling_rate, 100.)
        self.assertEqual(mod.overlap_npts, 1800)
        self.assertEqual(mod.primary_components, set(['3','Z']))
        self.assertEqual(mod.secondary_components, set(['12','NE']))
        self.assertEqual(mod.name,'WindMod_EQTransformer')

    def test_properties(self):
        mod = WindMod()
        self.assertEqual(mod.window_dt, 60.)
        self.assertEqual(mod.overlap_dt, 18.)
        self.assertEqual(mod.advance_dt, 41.99)
        self.assertEqual(mod.padded_dt, 42.)

    def test___init___target_npts(self):
        for _arg in [6000., 3000, 10000]:
            self.assertEqual(WindMod(target_npts=_arg).window_stats.target_npts, _arg)

        for _arg in [1., 1, 1799]:
            with self.assertRaises(ValueError):
                WindMod(target_npts=_arg, overlap_npts=1800)

        for _arg in [-1, 1.1, 'a']:
            with self.assertRaises(Exception):
                WindMod(target_npts=_arg)

        # mod = WindMod(target_npts=_arg)
        # # mod = WindMod(window_stats={'pthresh': 0.5})
        # # Assert default/assigned values
        # for _k, _v in WindowStats().items():
        #     if _k == 'pthresh':
        #         self.assertEqual(mod.window_stats[_k], 0.5)
        #     else:
        #         self.assertEqual(mod.window_stats[_k], _v)
        # mod = WindMod(window_stats=None)
        # ERRORS - TypeError for bad type

        

    # def test_check_input(self):
    #     self.assertIsNone(self.test_mod.check_input(self.ds))
    #     with self.assertRaises(TypeError):
    #         self.test_mod.check_input(deque([]))

    # def test_get_unit_input(self):
    #     # Make an appropriately scaled WindMod for the seattle record-section example
    #     ds_save = self.ds.copy()
    #     self.assertEqual(self.test_mod.index, {})
    #     ui = self.test_mod.get_unit_input(self.ds)
    #     # Setup check - make sure unit_input is non-empty
    #     self.assertGreater(len(ui), 0)
    #     # Assert that ds is unchanged
    #     self.assertEqual(ds_save, self.ds)
    #     # Assert index updated
    #     self.assertEqual(self.instruments, self.test_mod.index.keys())
    #     # Assert ui is dict
    #     self.assertIsInstance(ui, dict)
    #     # Assert ui keys are a subset of index keys
    #     self.assertLessEqual(ui.keys(), self.test_mod.index.keys())
    #     # Assert all ui values are DictStreams
    #     self.assertTrue(all(isinstance(_v, DictStream) for _v in ui.values()))
    #     # Assert all ui keys correspond to ready entries
    #     self.assertTrue(all(self.test_mod.index[_k]['ready'] for _k in ui.keys()))
    #     # Assert that unit_input is empty dict if nothing is ready
    #     self.assertTrue(self.test_mod._continue_pulsing)
    #     ui = self.test_mod.get_unit_input(DictStream())
    #     # Assert _continue_pulsing flag is flipped False
    #     self.assertFalse(self.test_mod._continue_pulsing)
    #     # Assert that ui is empty dict
    #     self.assertEqual(ui, {})

    # def test_run_unit_process(self):
    #     mod = WindMod(window_stats={'target_npts': 1000, 'target_sampling_rate': 100.}, overlap=200)
    #     ui = mod.get_unit_input(self.ds.copy())
    #     ui_copy = ui.copy()
    #     uo = mod.run_unit_process(ui)
    #     # Assert that each unit_input element produces one unit_output element
    #     self.assertEqual(len(ui), len(uo))