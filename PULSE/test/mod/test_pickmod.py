from unittest import TestCase
from collections import deque

import numpy as np
from obspy.core.event import Pick
from obspy import read
from seisbench.models import PhaseNet

from PULSE.data.foldtrace import FoldTrace
from PULSE.data.dictstream import DictStream
from PULSE.mod.base import BaseMod
from PULSE.mod.picking import PickMod

class TestPickMod(TestCase):
    # Load 3C data once
    st = read()
    # Trim to add 1 sample to front
    st.trim(starttime = st[0].stats.starttime - 0.01, pad=True, fill_value=0.)

    # Load SeisBench model once
    mod = PhaseNet().from_pretrained('instance')
    # Set blinding to 0
    # Run Prediction
    _pred = mod.annotate(st)
    # Ensure _pred timeframe matches st
    _pred.trim(starttime=st[0].stats.starttime, endtime=st[0].stats.endtime, pad=True, fill_value=0.)

    def setUp(self):
        self.ds_wave = DictStream(self.st.copy())
        self.ds_pred = DictStream()
        for tr in self._pred.copy():
            ft = FoldTrace(tr)
            if ft.stats.component == 'N':
                ft.stats.channel = 'EHX'
            else:
                ft.stats.channel = 'EH'+ft.stats.component
            ft.stats.model='PhaseNet'
            ft.stats.weight='instance'
            ft.fold = ft.fold*3
            self.assertTrue(all(ft.fold == 3))
            self.ds_pred.extend(ft)
        
    def tearDown(self):
        del self.ds_wave
        del self.ds_pred

    def test___init___bare(self):
        # Initialize bare
        test_mod = PickMod()
        # Assert Types
        self.assertIsInstance(test_mod, BaseMod)
        self.assertIsInstance(test_mod, PickMod)
        self.assertIsInstance(test_mod.output, deque)
        self.assertIsInstance(test_mod.threshold, float)
        self.assertIsInstance(test_mod.min_fold, float)
        self.assertIsInstance(test_mod.max_trig_len, int)
        self.assertIsInstance(test_mod.mtl_delete, bool)
        # Assert Values
        self.assertEqual(test_mod._input_types, [deque])
        self.assertEqual(test_mod.min_fold, 1.)
        self.assertEqual(test_mod.max_trig_len, 1000)
        self.assertTrue(test_mod.mtl_delete)
        self.assertIsNone(test_mod.output.maxlen)
        self.assertEqual(test_mod.name,'PickMod')
    
    def test___init___min_fold(self):
        # Assert works for non-negative float and int
        for _arg in [0, 0.3, 1e-5, 200]:
            self.assertEqual(PickMod(min_fold=_arg).min_fold, _arg)
        # ERRORS
        for _arg in ['a', [], DictStream()]:
            with self.assertRaises(TypeError):
                PickMod(min_fold=_arg)
        for _arg in [-1e-9, -1]:
            with self.assertRaises(ValueError):
                PickMod(min_fold=_arg)
    
    def test___init___threshold(self):
        for _arg in [1e-9, 1, 5000, 1.]:
            self.assertEqual(PickMod(threshold=_arg).threshold, _arg)
        # Errors
        for _arg in ['a', [], DictStream()]:
            with self.assertRaises(TypeError):
                PickMod(threshold=_arg)
        for _arg in [0, -1e-9, -1.1]:
            with self.assertRaises(ValueError):
                PickMod(threshold=_arg)

    def test___init___max_trig_len(self):
        for _arg in [1, 10, 300000]:
            self.assertEqual(PickMod(max_trig_len=_arg).max_trig_len, _arg)
        # Errors
        for _arg in ['a', [], DictStream(), 1.1]:
            with self.assertRaises(TypeError):
                PickMod(max_trig_len=_arg)
        for _arg in [-100, -1, 0]:
            with self.assertRaises(ValueError):
                PickMod(max_trig_len=_arg)
    
    def test___init___max_trig_len_delete(self):
        for _arg in [True, False]:
            self.assertEqual(PickMod(max_trig_len_delete=_arg).mtl_delete, _arg)
        for _arg in ['a',1, 1.1, DictStream()]:
            with self.assertRaises(TypeError):
                PickMod(max_trig_len_delete=_arg)


    def test_get_unit_input(self):
        inpt = deque([self.ds_pred])
        mod = PickMod()
        unit_input = mod.get_unit_input(inpt)
        # Assert unit_pred was retrieved
        self.assertIsInstance(unit_input, DictStream)
        self.assertEqual(unit_input, self.ds_pred)

    def test_run_unit_process(self):
        unit_input = self.ds_pred
        mod = PickMod()
        unit_output = mod.run_unit_process(unit_input)
        # Assert Type
        self.assertIsInstance(unit_output, deque)
        # Assert all resource_id's are unique
        id_set = set([_e.resource_id for _e in unit_output])
        self.assertEqual(len(id_set), len(unit_output))

        # Inspect each element
        for _e in unit_output:
            # Assert Type
            self.assertIsInstance(_e, Pick)
            # Assert resource_id formatting
            self.assertIn('PULSE', _e.resource_id.id)
            # Assert waveform ID formatting
            self.assertIn('?', _e.waveform_id.id)
            # Test that max value is roughly the same
            tp = _e.time
            ti = _e.time - _e.time_errors.lower_uncertainty
            tf = _e.time + _e.time_errors.upper_uncertainty
            # Reconstitute peak value
            mag = (mod.threshold/(1 - _e.time_errors.confidence_level/100))
            # Reconstitude relative peak sample position
            mpos = int((tp - ti)*100)
            _ift = unit_input.select(component=_e.phase_hint)[0].view(starttime=ti, endtime=tf)
            # Assert that max values are within rounding
            self.assertAlmostEqual(_ift.max(), mag, places=6)
            # Assert that max position is accurate to 1 sample
            self.assertAlmostEqual(mpos, np.argmax(_ift.data), delta=1)
        
        # Error check
        for _arg in [self.ds_pred[0].copy(), 'a', 1.1, 1]:
            with self.assertRaises(TypeError):
                mod.run_unit_process(_arg)
        
    def test_put_unit_output(self):
        # Setup
        mod = PickMod()
        self.assertEqual(len(mod.output), 0)
        unit_output = mod.run_unit_process(self.ds_pred)
        uol = len(unit_output)
        uoc = unit_output.copy()
        # Run test process
        mod.put_unit_output(unit_output)

        # Assert shifts in unit_output and mod.output element counts
        self.assertEqual(len(mod.output), uol)
        self.assertEqual(len(unit_output), 0)

        # Assert that ordering is preserved
        for _e in range(uol):
            if mod.output[_e] != uoc[_e]:
                breakpoint()
            self.assertEqual(mod.output[_e], uoc[_e])
        