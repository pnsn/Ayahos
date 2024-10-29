import unittest, pytest

from collections import deque
from obspy import read

from PULSE.data.foldtrace import FoldTrace
from PULSE.data.dictstream import DictStream
from PULSE.mod.process import ProcMod, BaseMod

class TestProcMod(unittest.TestCase):

    def setUp(self):
        self.test_modi = ProcMod(
            'PULSE.data.foldtrace.FoldTrace',
            'filter',
            pkwargs={'type': 'bandpass',
                    'freqmin': 1.,
                    'freqmax': 40.},
            mode='inplace',
            name=None)
        self.test_modo = ProcMod(
            'PULSE.data.dictstream.DictStream',
            'fnsearch',
            pkwargs={'idstring':"*.??[NZ]"},
            mode='output',
            name=None
        )
        self.st = read()
        self.test_inputi = deque([FoldTrace(self.st[0].copy()) for _ in range(6)])
        self.test_unit_outputi = FoldTrace(self.st[0].copy()).filter('bandpass', freqmin=1., freqmax=40.)
        self.test_outputi = deque([self.test_unit_outputi.copy() for _ in range(6)])
        self.test_inputo = deque([DictStream(self.st.copy()) for _ in range(6)])
        self.test_unit_outputo = {'BW.RJOB..EHN','BW.RJOB..EHZ'}
        self.test_outputo = deque([self.test_unit_outputo]*6)

    def tearDown(self):
        del self.test_modi
        del self.test_modo
        del self.test_inputi
        del self.test_inputo
        del self.test_outputi
        del self.test_outputo
        del self.test_unit_outputi
        del self.test_unit_outputo
        del self.st

    def test_init(self):
        self.assertIsInstance(self.test_modi, BaseMod)
        self.assertIsInstance(self.test_modo, BaseMod)
        self.assertIsInstance(self.test_modi, ProcMod)
        self.assertIsInstance(self.test_modo, ProcMod)
        self.assertEqual(self.test_modi.name, 'ProcMod_inplace_filter')
        self.assertEqual(self.test_modo.name, 'ProcMod_output_fnsearch')
    
    def test_init_errors(self):
        with self.assertRaises(SystemExit):
            ProcMod('PULSE.data.foldtrace.FoldTrace','baz')
        with self.assertRaises(SystemExit):
            ProcMod('PULSE.data.foldtrace.FoldTrace','filter',
                    pkwargs=['type','bandpass'])
        with self.assertRaises(SystemExit):
            ProcMod('PULSE.data.foldtrace.FoldTrace','copy',mode='OUTPUT')

    def test_get_unit_input(self):
        unit_input = self.test_modi.get_unit_input(self.test_inputi)
        self.assertIsInstance(unit_input, self.test_modi.pclass)
        unit_input = self.test_modo.get_unit_input(self.test_inputo)
        self.assertIsInstance(unit_input, self.test_modo.pclass)
        # Rases SystemExit
        with self.assertRaises(SystemExit):
            self.test_modi.get_unit_input(self.test_inputo)

    def test_run_unit_process(self):
        unit_input = self.test_modi.get_unit_input(self.test_inputi)
        self.assertEqual(len(unit_input.stats.processing), 0)
        unit_output = self.test_modi.run_unit_process(unit_input)
        # Assert that processing showed up (for this case)
        self.assertTrue(self.test_modi.pmethod in unit_output.stats.processing[-1])
        # Assert that the input and the output are identical (inplace modification)
        self.assertEqual(unit_input, unit_output)
        self.assertEqual(unit_output, self.test_unit_outputi)

        unit_input = self.test_modo.get_unit_input(self.test_inputo)
        self.assertEqual(len(unit_input), 3)
        unit_output = self.test_modo.run_unit_process(unit_input)
        # Assert expected output
        self.assertEqual(unit_output, self.test_unit_outputo)


    def test_pulse(self):
        self.setUp()
        output = self.test_modi.pulse(self.test_inputi)
        self.assertEqual(output, self.test_outputi)
        output = self.test_modo.pulse(self.test_inputo)
        self.assertEqual(output, self.test_outputo)

    