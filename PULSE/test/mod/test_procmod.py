import unittest, logging

from collections import deque
from obspy import read

from PULSE.data.foldtrace import FoldTrace
from PULSE.data.dictstream import DictStream
from PULSE.mod.processer import ProcMod, BaseMod

class TestProcMod(unittest.TestCase):

    def setUp(self):
        """SetUp procedure for each test in this TestCase
        
        Creates a pair of :class:`~.ProcMod` objects,
        one for 'inplace' and one for 'output' and a set of
        input traces loaded using :meth:`~obspy.read` loaded
        into :class:`~collections.deque` objects.

        """        
        self.test_modi = ProcMod(
            'PULSE.data.foldtrace.FoldTrace',
            'filter',
            pkwargs={'type': 'bandpass',
                    'freqmin': 1.,
                    'freqmax': 40.},
            mode='inplace',
            name=None)
        self.test_modo = ProcMod(
            'obspy.core.stream.Stream',
            'select',
            pkwargs={'channel':"[NZ]"},
            mode='output',
            name=None
        )
        # Read waveforms
        self.st = read()
        # Make an input for 'inplace' mod
        self.test_inputi = deque([FoldTrace(self.st[0].copy()) for _ in range(6)])
        # Make an unit output for 'inplace' mod
        self.test_unit_outputi = FoldTrace(self.st[0].copy()).filter('bandpass', freqmin=1., freqmax=40.)
        # Make an output for 'inplace' mod
        self.test_outputi = deque([self.test_unit_outputi.copy() for _ in range(6)])
        # Make an input for 'output' mod
        self.test_inputo = deque([self.st.copy() for _ in range(6)])
        # Generate unit output for 'output' mod
        self.test_unit_outputo = self.st.copy().select(channel='[NZ]')
        # Generate output for 'output' mod
        self.test_outputo = deque([self.test_unit_outputo]*6)

    def tearDown(self):
        """tearDown procedure for each test in this TestCase

        Deletes all objects generated using :meth:`~.TestProcMod.setUp`
        """        
        del self.test_modi
        del self.test_modo
        del self.test_inputi
        del self.test_inputo
        del self.test_outputi
        del self.test_outputo
        del self.test_unit_outputi
        del self.test_unit_outputo
        del self.st

    def test___init__(self):
        """Test suite for :meth:`~.ProcMod.__init__`
        """        
        self.assertIsInstance(self.test_modi, BaseMod)
        self.assertIsInstance(self.test_modo, BaseMod)
        self.assertIsInstance(self.test_modi, ProcMod)
        self.assertIsInstance(self.test_modo, ProcMod)
        self.assertEqual(self.test_modi.name, 'ProcMod_inplace_filter')
        self.assertEqual(self.test_modo.name, 'ProcMod_output_select')
    
    def test___init___critical(self):
        """Test suite for system exits raised by critical logging
        messages for :meth:`~.ProcMod.__init__`
        """        
        with self.assertRaises(AttributeError):
            ProcMod('PULSE.data.foldtrace.FoldTrace','baz')
        with self.assertRaises(TypeError):
            ProcMod('PULSE.data.foldtrace.FoldTrace','filter',
                    pkwargs=['type','bandpass'])
        with self.assertRaises(ValueError):
            ProcMod('PULSE.data.foldtrace.FoldTrace','copy',mode='OUTPUT')

    def test_get_unit_input(self):
        """Test suite for :meth:`~.ProcMod.get_unit_input`
        """        
        unit_input = self.test_modi.get_unit_input(self.test_inputi)
        self.assertIsInstance(unit_input, self.test_modi.pclass)
        unit_input = self.test_modo.get_unit_input(self.test_inputo)
        self.assertIsInstance(unit_input, self.test_modo.pclass)
        # Rases SystemExit
        with self.assertRaises(SystemExit):
            self.test_modi.get_unit_input(self.test_inputo)

    def test_run_unit_process(self):
        """Test suite for :meth:`~.ProcMod.run_unit_process`
        """        
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
        """test suite for :meth:`~.ProcMod.pulse`
        """
        # Run inplace pulse    
        output = self.test_modi.pulse(self.test_inputi)
        self.assertEqual(output, self.test_outputi)
        # Run output pulse
        output = self.test_modo.pulse(self.test_inputo)
        self.assertEqual(output, self.test_outputo)

    