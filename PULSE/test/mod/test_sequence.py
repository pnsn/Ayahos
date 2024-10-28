import pytest
from collections import deque
from unittest import TestCase

import pandas as pd

from PULSE.mod.base import BaseMod
from PULSE.mod.sequence import Sequence

class TestSequence(TestCase):

    def setUp(self):
        self.test_input = [BaseMod(name='0'), BaseMod(name='1')]
        self.test_seq = Sequence(self.test_input)
        self.test_mod = BaseMod(name='0', max_pulse_size=2)


    def tearDown(self):
        del self.test_input

    def test_init(self):
        self.assertIsInstance(self.test_seq, dict)
        self.assertIsInstance(self.test_seq, Sequence)

    def test_init_modules(self):
        self.assertIsInstance(self.test_seq, Sequence)
        self.assertEqual(len(self.test_seq), 2)
        self.assertEqual(list(self.test_seq.keys()), ['BaseMod_0','BaseMod_1'] )
        for _k, _v in self.test_seq.items():
            self.assertIsInstance(_v, BaseMod)
            self.assertEqual(_v.name, _k)
        # Test that input order is preserved
        seq1 = Sequence([BaseMod(name='j'), BaseMod(name='a'), BaseMod(name='1')])
        self.assertEqual(list(seq1.keys()), ['BaseMod_j','BaseMod_a','BaseMod_1'])
    
    def test_get_first(self):
        """Test the **get_first** method of :class:`~.Sequence`
        """  
        self.assertEqual(self.test_seq.first, self.test_input[0])

    def test_get_last(self):
        """Test the **get_last** method of :class:`~.Sequence`
        """  
        self.assertEqual(self.test_seq.last, self.test_input[-1])

    def test_copy(self):
        """Test the **update** method of :class:`~.Sequence`
        """  
        seq1 = self.test_seq.copy()
        self.assertIsInstance(seq1, Sequence)

    def test_update(self):
        """Test the **update** method of :class:`~.Sequence`
        """        
        seq1 = self.test_seq.copy()
        self.assertEqual(seq1.first.stats.mps, 1)
        # Test BaseMod input
        seq1.update(self.test_mod)
        self.assertEqual(seq1.first.stats.mps, 2)
        # Test Iterables
        for inpt in [[self.test_mod], (self.test_mod), {'BaseMod_0': self.test_mod}, {self.test_mod}]:
            seq1 = self.test_seq.copy()
            seq1.update(inpt)
        # Test errors
        with pytest.raises(TypeError):
            seq1.update(int)
        with pytest.raises(TypeError):
            seq1.update([1])
        with pytest.raises(KeyError):
            seq1.update({'BaseMod_1': BaseMod(name='0')})
        with pytest.raises(TypeError):
            seq1.update({'BaseMod_0': 'abc'})

    def test_validate(self):
        """Test the **validate** method of :class:`~.Sequence`
        """        
        self.test_seq.validate()
        # Precheck
        self.assertEqual(self.test_seq.last._input_types, [deque])
        # Setup - test mismatched _input_types
        self.test_seq.last._input_types = [int]
        self.assertEqual(self.test_seq.last._input_types, [int])
        with pytest.raises(SyntaxError):
            self.test_seq.validate()
        self.test_seq['BaseMod_0'] = 1
        self.assertEqual(self.test_seq['BaseMod_0'], 1)
        with pytest.raises(TypeError):
            self.test_seq.validate()

    def test_get_current_stats(self):
        """Test suite for the **get_current_stats** method of 
        :class:`~.Sequence and it's property **current_stats**
        """        
        # Test sequence with 2 items
        self.assertIsInstance(self.test_seq.get_current_stats(), pd.DataFrame)
        self.assertEqual(len(self.test_seq), len(self.test_seq.get_current_stats()))
        # Test empty sequence
        seq0 = Sequence()
        self.assertIsInstance(seq0.current_stats, pd.DataFrame)
        self.assertEqual(len(seq0.current_stats), 0)