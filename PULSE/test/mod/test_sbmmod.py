"""
:module: PULSE.test.mod.test_sbmmod
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose: :class:`~unittest.TestCase` for :class:`~PULSE.mod.sbm.SBMMod`
 Test suite for the SeisBenchModel Module class of PULSE
"""
from unittest import TestCase
from collections import deque
from sys import platform

import torch
import seisbench.models as sbm
import numpy as np
from obspy import read

from PULSE.util.header import MLStats
from PULSE.data.window import Window
from PULSE.data.dictstream import DictStream
from PULSE.mod.base import BaseMod
from PULSE.mod.sbm_predicter import SBMMod

# Globally set number of threads for torch use to half available
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

    def tearDown(self):
        del self.w6000
        del self.w3001

    def test___init__(self):
        """Test suite for :meth:`~.SBMMod.__init__`
        """        
        # Test types from defaults in __init__
        mod = SBMMod()
        self.assertIsInstance(mod, BaseMod)
        self.assertIsInstance(mod, SBMMod)
        self.assertIsInstance(mod.model, sbm.WaveformModel)
        self.assertIsInstance(mod.model, sbm.EQTransformer)
        self.assertIsInstance(mod.output, deque)
        self.assertIsInstance(mod.device, torch.device)
        for _mod in mod.cmods.values():
            self.assertIsInstance(_mod, sbm.WaveformModel)
        self.assertEqual(set(['pnw']), mod.cmods.keys())
        # Test equalities
        self.assertEqual(mod.device.type, 'cpu')
        # Test auto-assigned batch_sizes
        self.assertEqual(mod.batch_sizes, (1, 256))
        # Test auto-name adjustment
        self.assertEqual(mod.name, 'SBMMod_EQTransformer')

    def test___init___model(self):
        """Test suite for :meth:`~.SBMMod.__init__` model inputs
        """        
        mod = SBMMod(model=sbm.PhaseNet(), weight_names=['original', 'stead'])
        self.assertIsInstance(mod, SBMMod)
        self.assertIsInstance(mod.model, sbm.PhaseNet)
        self.assertEqual(mod.cmods.keys(), set(['original', 'stead']))
        # Test errors - WaveformModel
        with self.assertRaises(TypeError):
            SBMMod(model=sbm.WaveformModel())
        # Test errors - non-model
        with self.assertRaises(TypeError):
            SBMMod(model='foo')
        # Test not implemented error
        with self.assertRaises(NotImplementedError):
            SBMMod(model=sbm.BasicPhaseAE())

    def test___init___weight_names(self):
        """Test suite for :meth:`~.SBMMod.__init__` weight_name inputs
        """  
        mod = SBMMod(weight_names=['pnw','stead','instance'])
        # Test that 3 models are generated
        self.assertEqual(len(mod.cmods), 3)
        # Test that the model keys are set
        self.assertEqual(mod.cmods.keys(), set(['stead','pnw','instance']))
        # Test string input
        mod = SBMMod(weight_names='pnw')
        self.assertEqual(mod.cmods.keys(), set(['pnw']))
        # Test mismatched name error
        with self.assertRaises(ValueError):
            SBMMod(model=sbm.PhaseNet(), weight_names=['pnw'])
    
    def test___init___compiled(self):
        """Test suite for :meth:`~.SBMMod.__init__` compiled flag
        """          
        mod = SBMMod(weight_names=['pnw','instance'], compiled=True)
        self.assertIsInstance(mod, SBMMod)
        for _wt, _mod in mod.cmods.items():
            self.assertIsInstance(_mod, sbm.WaveformModel)
        
    def test___init___device(self):
        """Test suite for :meth:`~.SBMMod.__init__` device flag
        """ 
        # For M1+ Apple Silicon - metal performance shaders
        mod = SBMMod()
        self.assertEqual(mod.device.type, 'cpu')
        if platform == 'darwin':
            mod = SBMMod(device='mps')
            self.assertEqual(mod.device.type, 'mps')
        with self.assertRaises(RuntimeError):
            mod = SBMMod(device='abc')
    
    def test___init___batch_sizes(self):
        self.assertEqual((1,32), SBMMod(batch_sizes=(1,32)).batch_sizes)
        with self.assertRaises(TypeError):
            SBMMod(batch_sizes=[1,32])
        with self.assertRaises(ValueError):
            SBMMod(batch_sizes=tuple([2]))
        with self.assertRaises(TypeError):
            SBMMod(batch_sizes=(1., 32.))
        with self.assertRaises(ValueError):
            SBMMod(batch_sizes=(-1, 32))
    
    def test_get_unit_input(self):
        input = deque([self.w6000 for _e in range(29)])
        mod = SBMMod(batch_sizes=(1,16))
        mod._continue_pulsing = True
        # Run first get_unit_input
        for _e, _size in enumerate([16, 13, 0]):
            # Run get_unit_input
            unit_input = mod.get_unit_input(input)
            if _e < 2:
                # Conduct checks on structure of unit_output
                self.assertIsInstance(unit_input, dict)
                self.assertEqual(unit_input.keys(), set(['data','fold','meta']))
                self.assertIsInstance(unit_input['data'], np.ndarray)
                self.assertIsInstance(unit_input['fold'], np.ndarray)
                self.assertIsInstance(unit_input['meta'], list)
                self.assertTrue(all(isinstance(_e, MLStats) for _e in unit_input['meta']))
                self.assertEqual(unit_input['data'].shape, (_size,3,6000))
                self.assertEqual(unit_input['fold'].shape, (_size,6000))
                self.assertEqual(len(unit_input['meta']), _size)
                # Make sure _continue_pulsing flag wasn't flipped
                self.assertTrue(mod._continue_pulsing)
            # Test None return for empyt input
            else:
                self.assertIsNone(unit_input)

                self.assertFalse(mod._continue_pulsing)
        # Test too-small batch_sizes behavior
        input = deque([self.w6000 for _e in range(3)])
        mod = SBMMod(batch_sizes=(4, 16))
        unit_input = mod.get_unit_input(input)
        self.assertIsNone(unit_input)
        self.assertEqual(len(input), 3)
        
        # Test Incorrect Element Type Error
        input = deque(['a' for _e in range(20)])
        with self.assertRaises(TypeError):
            mod.get_unit_input(input)
        
        # Test Window target npts dont match model error
        input = deque([self.w3001 for _e in range(10)])
        with self.assertRaises(AttributeError):
            mod.get_unit_input(input)

        # Test Window target sampling_rate dont match model error
        self.w6000.stats.target_sampling_rate *= 2
        for ft in self.w6000:
            ft.stats.sampling_rate *= 2
        input = deque([self.w6000 for _e in range(10)])
        with self.assertRaises(AttributeError):
            mod.get_unit_input(input)



    def test_run_unit_process(self):
        mod = SBMMod(weight_names=['pnw','instance'])
        input = deque([self.w6000.copy() for _e in range(3)])
        unit_input = mod.get_unit_input(input)
        unit_output = mod.run_unit_process(unit_input)
        # Inspect type of unit_output
        self.assertIsInstance(unit_output, dict)
        # Inspect structure of unit_output
        self.assertEqual(set(['data','meta','fold','pred']), unit_output.keys())
        self.assertEqual(set(['pnw','instance']), unit_output['pred'].keys())
        for _k, _v in unit_output['pred'].items():
            # Assert prediction array type
            self.assertIsInstance(_v, np.ndarray)
            # Assert that prediction shape matches data shape
            self.assertEqual(_v.shape, unit_input['data'].shape)
            # Assert that predictions are in range [0, 1]
            self.assertLessEqual(_v.max(), 1)
            self.assertGreaterEqual(_v.min(), 0)
        
        # Assert that data fold and meta are identical to input
        for fld in ['data','fold']:
            np.testing.assert_array_equal(unit_input[fld], unit_output[fld])
        self.assertTrue(all(unit_input['meta'][_e] == unit_output['meta'][_e] for _e in range(3)))
    

    def test_put_unit_output(self):
        """Test suite for :meth:`~.SBMMod.put_unit_output`
        """        
        # SetUp
        mod = SBMMod(weight_names=['pnw','instance'])
        self.assertEqual(len(mod.output), 0)
        input = deque([self.w6000.copy()]*2)
        # Build expected output names
        out_names = []
        ft = self.w6000[0]
        for _lbl in 'PSD':
            for wn in ['pnw','instance']:
                out_names.append(f'{ft.id[:-1]}{_lbl}.EQTransformer.{wn}')
        out_names = set(out_names)
        unit_input = mod.get_unit_input(input)
        unit_output = mod.run_unit_process(unit_input)
        
        # Run Method
        out = mod.put_unit_output(unit_output)

        # Assert that returns None
        self.assertIsNone(out)
        # Assert mod.output now has entries
        self.assertEqual(len(mod.output), 2)
        # Check properties of each entry
        for _e in mod.output:
            # Assert output element type
            self.assertIsInstance(_e, DictStream)
            # Assert number of prediction traces in DictStream (#weights x #labels)
            self.assertEqual(len(_e), 6)
            # Assert correct updated names
            self.assertEqual(out_names, set(_e.keys))
    
    def test_pulse(self):
        """Test suite for :meth:`~.SBMMod.pulse`
        """        
        mod = SBMMod(batch_sizes=(2,8))
        input = deque([self.w6000.copy() for _e in range(9)])
        # Run pulse
        output = mod.pulse(input)
        self.assertIsInstance(output, deque)
        self.assertEqual(output, mod.output)
        status = mod.stats
        # Assert that pulse stopped early following get_unit_input
        self.assertEqual(status['stop'], 'early-get')
        # Assert that initial input size was reported correctly
        self.assertEqual(status['in_init'], 9)
        # Assert final input size was reported correctly
        self.assertEqual(status['in_final'], len(input))
        # Assert that input size matches expectations after pulse
        self.assertEqual(len(input), 1)
        # Assert that output length matches reporting in stats
        self.assertEqual(len(output), status['out_final'])
        # Assert that number of expected iterations is reported in stats
        self.assertEqual(status['niter'], 1)
        # Assert the continue pulsing flag got flipped
        self.assertFalse(mod._continue_pulsing)
        

