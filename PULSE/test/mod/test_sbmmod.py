from unittest import TestCase
from pathlib import Path
from collections import deque

import torch
import seisbench.models as sbm
import numpy as np
from obspy import read

from PULSE.data.header import MLStats
from PULSE.data.window import Window
from PULSE.data.foldtrace import FoldTrace
from PULSE.data.dictstream import DictStream
from PULSE.mod.base import BaseMod
from PULSE.mod.sbm import SBMMod
# from PULSE.test.example_data import load_townsend_example

# Globally set number of threads for torch use
torch.set_num_threads(torch.get_num_threads()//2)

class Test_SBMMod(TestCase):
    # Get max thread count at top of test
    max_threads = torch.get_num_threads()
    # st = load_townsend_example(Path().cwd())
    def setUp(self):
        
        # Create a 6000 sample 3-C window
        st6000 = read().resample(200)
        for tr in st6000:
            tr.stats.sampling_rate = 100
        self.w6000 = Window(st6000)
        # Create a 3001 sample 3-C window
        st3001 = read()
        for tr in st3001:
            tr.data = np.r_[tr.data, tr.data[-1]]
        self.w3001 = Window(st3001)
        self.test_mod = SBMMod()

    def tearDown(self):
        del self.test_mod
        del self.w6000
        del self.w3001

    # def test___init__(self):
    #     # Test types from defaults in __init__
    #     self.assertIsInstance(self.test_mod, BaseMod)
    #     self.assertIsInstance(self.test_mod, SBMMod)
    #     self.assertIsInstance(self.test_mod.model, sbm.WaveformModel)
    #     self.assertIsInstance(self.test_mod.model, sbm.EQTransformer)
    #     self.assertIsInstance(self.test_mod.output, deque)
    #     self.assertIsInstance(self.test_mod.device, torch.device)
    #     for _mod in self.test_mod.cmods.values():
    #         self.assertIsInstance(_mod, sbm.WaveformModel)
    #     self.assertEqual(set(['pnw']), self.test_mod.cmods.keys())
    #     # Test equalities
    #     self.assertEqual(self.test_mod.device.type, 'cpu')
    #     # Test auto-assigned batch_sizes
    #     self.assertEqual(self.test_mod.batch_sizes, (1, 256))
    #     # Test auto-name adjustment
    #     self.assertEqual(self.test_mod.name, 'SBMMod_EQTransformer')

    # def test___init___model(self):
    #     mod = SBMMod(model=sbm.PhaseNet(), weight_names=['original', 'stead'])
    #     self.assertIsInstance(mod, SBMMod)
    #     self.assertIsInstance(mod.model, sbm.PhaseNet)
    #     self.assertEqual(mod.cmods.keys(), set(['original', 'stead']))
    #     # Test errors - WaveformModel
    #     with self.assertRaises(TypeError):
    #         SBMMod(model=sbm.WaveformModel())
    #     # Test errors - non-model
    #     with self.assertRaises(TypeError):
    #         SBMMod(model='foo')
    #     # Test not implemented error
    #     with self.assertRaises(NotImplementedError):
    #         SBMMod(model=sbm.BasicPhaseAE())

    # def test___init___weight_names(self):
    #     mod = SBMMod(weight_names=['pnw','stead','instance'])
    #     # Test that 3 models are generated
    #     self.assertEqual(len(mod.cmods), 3)
    #     # Test that the model keys are set
    #     self.assertEqual(mod.cmods.keys(), set(['stead','pnw','instance']))
    #     # Test string input
    #     mod = SBMMod(weight_names='pnw')
    #     self.assertEqual(mod.cmods.keys(), set(['pnw']))
    #     # Test mismatched name error
    #     with self.assertRaises(ValueError):
    #         SBMMod(model=sbm.PhaseNet(), weight_names=['pnw'])
    
    # def test___init___compiled(self):
    #     mod = SBMMod(weight_names=['pnw','instance'], compiled=True)
    #     self.assertIsInstance(mod, SBMMod)
    #     for _wt, _mod in mod.cmods.items():
    #         self.assertIsInstance(_mod, sbm.WaveformModel)
        
    # def test___init___device(self):
    #     with self.assertRaises(RuntimeError):
    #         mod = SBMMod(device='abc')
    
    # def test___init___batch_sizes(self):
    #     self.assertEqual((1,32), SBMMod(batch_sizes=(1,32)).batch_sizes)
    #     with self.assertRaises(TypeError):
    #         SBMMod(batch_sizes=[1,32])
    #     with self.assertRaises(ValueError):
    #         SBMMod(batch_sizes=(2))
    #     with self.assertRaises(TypeError):
    #         SBMMod(batch_sizes=(1., 32.))
    #     with self.assertRaises(ValueError):
    #         SBMMod(bathc_sizes=(-1, 32))
    
    def test_get_unit_input(self):
        input = deque([self.w6000 for _e in range(29)])
        mod = SBMMod(batch_sizes=(1,16))
        unit_input = mod.get_unit_input(input)
        self.assertIsInstance(unit_input, dict)
        self.assertEqual(unit_input.keys(), set(['data','fold','meta']))
        self.assertIsInstance(unit_input['data'], np.ndarray)
        self.assertIsInstance(unit_input['fold'], np.ndarray)
        self.assertIsInstance(unit_input['meta'], list)
        self.assertTrue(all(isinstance(_e, MLStats) for _e in unit_input['meta']))
        self.assertEqual(unit_input['data'].shape, (16,3,6000))
        self.assertEqual(unit_input['fold'].shape, (16,6000))
        self.assertEqual(len(unit_input['meta']), 16)


