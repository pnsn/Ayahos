import unittest, logging
from collections import deque

import numpy as np
from obspy import read

from PULSE.data.foldtrace import FoldTrace
from PULSE.data.ftbuffer import FTBuffer
from PULSE.data.dictstream import DictStream
from PULSE.mod.base import BaseMod
from PULSE.mod.buffering import BufferMod


class TestBufferMod(unittest.TestCase):
    st = read()
        
    def setUp(self):
        self.test_mod = BufferMod()
        self.test_input = deque()
        for tr in self.st:
            for itr in tr/4:
                self.test_input.append(itr)

    def tearDown(self):
        del self.test_mod
        del self.test_input

    def test___init__(self):
        # Assert type identities
        self.assertIsInstance(self.test_mod, BaseMod)
        self.assertIsInstance(self.test_mod, BufferMod)
        self.assertIsInstance(self.test_mod.output, DictStream)
    
    def test___init__method(self):
        for _e in range(4):
            if _e != 1:
                self.assertIsInstance(BufferMod(method=_e), BufferMod)
            else:
                with self.assertRaises(ValueError):
                    BufferMod(method=_e)
        for _e in ['a', None, str]:
            with self.assertRaises(ValueError):
                BufferMod(method=_e)
    
    def test___init__fill_value(self):
        """Test suite for :meth:`~.BufferMod.__init__` for
        (non)standard inputs to **fill_value**
        """        
        for _e in [1, 1., None]:
            self.assertIsInstance(BufferMod(fill_value=_e), BufferMod)
        for _e in ['a', type]:
            with self.assertRaises(TypeError):
                BufferMod(fill_value=_e)

    def test_run_unit_process(self):
        """Test suite for :meth:`~.BufferMod.run_unit_process`
        """        
        # Test with Trace input
        unit_input = self.test_input[-1].copy()
        unit_output = self.test_mod.run_unit_process(unit_input)
        # Assert unit_output is FoldTrace
        self.assertIsInstance(unit_output,FoldTrace)
        # Test with FoldTrace input
        unit_input = FoldTrace(self.test_input[-1].copy())
        unit_output = self.test_mod.run_unit_process(unit_input)
        self.assertIsInstance(unit_output, FoldTrace)
        # Raise critical message with mismatched type
        with self.assertLogs(level=logging.CRITICAL):
            self.test_mod.run_unit_process('a')

    def test_put_unit_output(self):
        """Test suite for :meth:`~.BufferMod.put_unit_output`
        """        
        ft = FoldTrace(self.test_input[0])
        self.assertEqual(self.test_mod.measure_output(), 0)
        self.test_mod.put_unit_output(ft)
        # Assert appended trace ID is in DictStream keys
        self.assertIn(ft.id, self.test_mod.output.keys)
        # Assert that the buffer has the correct length
        self.assertEqual(self.test_mod.output[ft.id].maxlen, self.test_mod.maxlen)
        # incorrect type raises critical logging message
        with self.assertLogs(level=logging.CRITICAL):
            self.test_mod.put_unit_output('a')
    
    def test_pulse(self):
        """Test suite for :meth:`~.BufferMod.pulse`
        """        
        output = self.test_mod.pulse(self.test_input)
        self.assertIsInstance(output, DictStream)
        for ftb in output:
            # Assert type
            self.assertIsInstance(ftb, FTBuffer)
            # Assert ID in input ID's
            self.assertIn(ftb.id, [tr.id for tr in self.st])
            # Get matching trace
            tr = self.st.select(id=ftb.id)[0]
            # Assert that the re-assembled data are in the right spot
            # in the buffer and match the original data
            np.testing.assert_array_equal(tr.data,
                                          ftb.data[-1*tr.count():].data)
        

    