import unittest
from pathlib import Path
from collections import deque

from PULSE.mod.base import BaseMod
from PULSE.mod.window import WindMod
from PULSE.data.dictstream import DictStream
from PULSE.test.example_data import load_seattle_example

class TestWindMod(unittest.TestCase):
    st, _, _ = load_seattle_example(Path().cwd())
    def setUp(self):
        self.ds = DictStream(self.st.copy())
        self.instruments = self.ds.split_on().keys()
        self.test_mod = WindMod()
    
    def tearDown(self):
        del self.ds
        del self.test_mod

    def test___init__(self):
        self.assertIsInstance(self.test_mod, BaseMod)
        self.assertIsInstance(self.test_mod, WindMod)
        self.assertEqual(self.test_mod._input_types, [DictStream])
        self.assertIsInstance(self.test_mod.index, dict)
        self.assertIsInstance(self.test_mod.output, deque)
        self.assertEqual(self.test_mod.index, {})
        

    def test_check_input(self):
        self.assertIsNone(self.test_mod.check_input(self.ds))
        with self.assertRaises(TypeError):
            self.test_mod.check_input(deque([]))

    def test_get_unit_input(self):
        # Make an appropriately scaled WindMod for the seattle record-section example
        test_mod = WindMod(window_stats={'target_npts': 1000, 'target_sampling_rate': 100.}, overlap=200)
        ds_save = self.ds.copy()
        self.assertEqual(test_mod.index, {})
        ui = test_mod.get_unit_input(self.ds)
        # Setup check - make sure unit_input is non-empty
        self.assertGreater(len(ui), 0)
        # Assert that ds is unchanged
        self.assertEqual(ds_save, self.ds)
        # Assert index updated
        self.assertEqual(self.instruments, test_mod.index.keys())
        # Assert ui is dict
        self.assertIsInstance(ui, dict)
        # Assert ui keys are a subset of index keys
        self.assertLessEqual(ui.keys(), test_mod.index.keys())
        # Assert all ui values are DictStreams
        self.assertTrue(all(isinstance(_v, DictStream) for _v in ui.values()))
        # Assert all ui keys correspond to ready entries
        self.assertTrue(all(test_mod.index[_k]['ready'] for _k in ui.keys()))
        # Assert that unit_input is empty dict if nothing is ready
        self.assertTrue(test_mod._continue_pulsing)
        ui = test_mod.get_unit_input(DictStream())
        # Assert _continue_pulsing flag is flipped False
        self.assertFalse(test_mod._continue_pulsing)
        # Assert that ui is empty dict
        self.assertEqual(ui, {})



    def test_update_from_seisbench(self):
        self.assertTrue(False)
    
    # def test_get_unit_input(self):