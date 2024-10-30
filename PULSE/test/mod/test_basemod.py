"""
:module: PULSE.test.mod.test_basemod
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:AI Attribution: Unit test suite developed with assistance from ChatGPT with
and input of the PULSE.mod.base.BaseMod source code. Subsequent testing and
verification conducted by the author.

The archived conversation can be found here:
https://chatgpt.com/share/67194525-575c-8002-a6cc-80b1873de09f
"""

import unittest
from collections import deque
from obspy import UTCDateTime

from PULSE.util.header import ModStats
from PULSE.mod.base import BaseMod  # Replace with the actual module import


class TestBaseMod(unittest.TestCase):

    #############################################
    ## setUp and tearDown methods for TestCase ##
    #############################################

    def setUp(self):
        """Set up test environment."""
        self.test_mod = BaseMod(max_pulse_size=2, maxlen=5, name='Test')
        self.test_input = deque(range(9))
        self.test_units = [1, 'a', int, None]
        self.test_outputs = range(20)
    
    def tearDown(self) -> None:
        del self.test_mod
        del self.test_input
        del self.test_units
        del self.test_outputs

    #########################
    ## __init__ test suite ##
    #########################
    def test_init(self):
        """Test proper initialization of BaseMod."""
        self.assertEqual(self.test_mod.stats.mps, 2)
        self.assertEqual(self.test_mod.stats.maxlen, 5)
        self.assertEqual(self.test_mod.stats.name, f'{self.test_mod.__class__.__name__}_Test')
        self.assertEqual(len(self.test_mod.output), 0)
        self.assertIsInstance(self.test_mod.stats, ModStats)

    def test_invalid_max_pulse_size(self):
        """Test if invalid max_pulse_size raises errors."""
        with self.assertRaises(ValueError):
            self.test_mod.__class__(max_pulse_size=0)
        with self.assertRaises(TypeError):
            self.test_mod.__class__(max_pulse_size='invalid')

    def test_invalid_name_type(self):
        """Test if invalid name types raise errors."""
        with self.assertRaises(TypeError):
            self.test_mod.__class__(name=123)

    ###############################
    ## utility method test suite ##
    ###############################
            
    def test_setname(self):
        self.test_mod.setname('Test')
        self.assertEqual(self.test_mod.name, self.test_mod.stats.name)
        self.assertEqual(self.test_mod.name, f'{self.test_mod.__class__.__name__}_Test')
        self.test_mod.setname(None)
        self.assertEqual(self.test_mod.name, self.test_mod.__class__.__name__)

    def test_repr(self):
        repr_str = self.test_mod.__repr__()
        self.assertIn(f'{self.test_mod.__class__.__name__}_Test', repr_str)
        self.assertIn(f'mps: {self.test_mod.stats.mps}', repr_str)

    def test_copy(self):
        basemod2 = self.test_mod.copy()
        self.assertIsInstance(basemod2, BaseMod)
        self.assertEqual(basemod2.stats.name, f'{self.test_mod.__class__.__name__}_Test')
    
    def test_import_class(self):
        """Test the import_class method."""
        cls = self.test_mod.import_class('obspy.core.utcdatetime.UTCDateTime')
        self.assertEqual(cls, UTCDateTime)

        with self.assertRaises(ValueError):
            self.test_mod.import_class('invalid_class_name')

    ####################################
    ## pulse & subroutines test suite ##
    ####################################    

    def test_check_input(self):
        # Test with incorrect input type
        with self.assertRaises(SystemExit):
            self.test_mod.check_input([])  # Should be deque

    def test_measure_input(self):
        self.assertEqual(self.test_mod.measure_input(self.test_input),
                         len(self.test_input))
    
    def test_measure_output(self):
        self.assertEqual(self.test_mod.measure_output(), 0)
        self.test_mod.pulse(self.test_input)
        self.assertEqual(self.test_mod.measure_output(), 2)

    def test_pulse_startup(self):
        """Test the pulse_startup method."""
        self.test_mod.pulse_startup(self.test_input)
        self.assertEqual(self.test_mod.stats.in_init, len(self.test_input))
        self.assertEqual(self.test_mod.stats.out_init, 0)
        self.assertTrue(self.test_mod._continue_pulsing)

    def test_pulse_shutdown(self):
        """Test the pulse_shutdown method."""
        test_input = self.test_input
        self.test_mod.pulse_shutdown(test_input, niter=self.test_mod.stats.mps - 1, exit_type='max')
        self.assertEqual(self.test_mod.stats.niter, self.test_mod.stats.mps)

    def test_pulse_shutdown_exit_types(self):
        """Test the pulse_shutdown exit_type variable
        """        
        self.test_mod.pulse_shutdown(self.test_input, niter=0, exit_type='nodata')
        self.assertEqual(self.test_mod.stats.niter, self.test_mod.stats.mps - 2)
        self.test_mod.pulse_shutdown(self.test_input, niter=1, exit_type='early-get')
        self.assertEqual(self.test_mod.stats.niter, self.test_mod.stats.mps - 1)
        self.test_mod.pulse_shutdown(self.test_input, niter=1, exit_type='early-run')
        self.assertEqual(self.test_mod.stats.niter, self.test_mod.stats.mps - 1)
        self.test_mod.pulse_shutdown(self.test_input, niter=1, exit_type='early-put')
        self.assertEqual(self.test_mod.stats.niter, self.test_mod.stats.mps)
        # Test with incorrect exit_type
        with self.assertRaises(SystemExit):
            self.test_mod.pulse_shutdown(self.test_input, niter=2, exit_type='invalid')

    def test_get_unit_input(self):
        """Test the get_unit_input method."""
        test_input = self.test_input.copy()
        unit_input = self.test_mod.get_unit_input(test_input)
        self.assertEqual(unit_input, self.test_input[-1])

        # Test early stopping when deque is empty
        empty_input = deque()
        unit_input = self.test_mod.get_unit_input(empty_input)
        self.assertIsNone(unit_input)
        self.assertFalse(self.test_mod._continue_pulsing)
    
    def test_run_unit_process(self):
        """Test the run_unit_process method of BaseMod
        """        
        for unit_input in self.test_units:
            unit_output = self.test_mod.run_unit_process(unit_input)
            self.assertEqual(unit_input, unit_output)
        
    def test_put_unit_output(self):
        """Test the put_unit_output method of BaseMod
        """        
        for _e, unit_output in enumerate(self.test_units):
            self.test_mod.put_unit_output(unit_output)
            self.assertEqual(self.test_mod.output[0], unit_output)
            self.assertEqual(len(self.test_mod.output), _e+1)

    def test_put_unit_output_maxlen(self):
        """Test the put_unit_output method behavior surrounding
        maxlen setting for BaseMod
        """        
        for _e, unit_output in enumerate(self.test_outputs):
            self.test_mod.put_unit_output(unit_output)
            self.assertEqual(self.test_mod.output[0], self.test_outputs[_e])
            if _e+1 <= self.test_mod.stats.maxlen:
                self.assertEqual(len(self.test_mod.output), _e + 1)
            else:
                self.assertEqual(len(self.test_mod.output), self.test_mod.stats.maxlen)

    def test_pulse(self):
        """Test the pulse method
        """        
        inputs = self.test_input
        for _n in range(5):
            out = self.test_mod.pulse(inputs)
            # First call
            if _n == 0:
                self.assertEqual(len(self.test_mod.output), 2)
                self.assertEqual(self.test_mod.output, deque([7,8]))
            # Second call
            elif _n == 1:
                self.assertEqual(len(self.test_mod.output), 4)
                self.assertEqual(self.test_mod.output, deque([5,6,7,8]))
            # Third and onward (hit maxlen)
            else:
                self.assertEqual(len(self.test_mod.output), 5)
            # For all but the last, end on 'max'
            if _n < 4:
                self.assertEqual(self.test_mod.stats.stop, 'max')
            else:
                self.assertEqual(self.test_mod.stats.stop, 'early-get')



