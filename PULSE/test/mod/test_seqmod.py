
import unittest
from collections import deque
from PULSE.mod.base import BaseMod
from PULSE.mod.sequence import Sequence, SeqMod
from PULSE.test.mod.test_basemod import TestBaseMod


class TestSeqMod(TestBaseMod):

    def setUp(self):
        super().setUp()
        self.test_seq = Sequence([BaseMod(name=str(_e), max_pulse_size=4-_e) for _e in range(3)])
        self.test_mod = SeqMod(self.test_seq, name='Test')

    def tearDown(self) -> None:
        del self.test_seq
        del self.test_mod


    def test_init(self):
        # Check types
        self.assertIsInstance(self.test_mod, BaseMod)
        self.assertIsInstance(self.test_mod, SeqMod)
        self.assertIsInstance(self.test_mod.sequence, Sequence)
        # Check empty output match
        self.assertEqual(self.test_mod.output, self.test_seq['BaseMod_2'].output)
        # Check input_types match
        self.assertEqual(self.test_mod._input_types, self.test_seq['BaseMod_0']._input_types)
        # Check other reasonable inputs
        for modules in [BaseMod(), {BaseMod()}, {'BaseMod': BaseMod()}, (BaseMod())]:
            self.assertIsInstance(SeqMod(modules), SeqMod)

        # Other types raises SystemExit from maxlen
        for maxlen in [0, None, 'a']:
            with self.assertRaises(SystemExit):
                SeqMod(self.test_seq, maxlen=maxlen)
        # Other types raises SystemExit from modules
        for module in [int , [1], {'BaseMod_1': BaseMod(name='0')}, {'BaseMod_0': 'abc'}]:
            with self.assertRaises(SystemExit):
                SeqMod(module)

    def test_get_unit_input(self):
        self.assertEqual(self.test_mod.get_unit_input(self.test_input), self.test_input)


    def test_unit_output(self):
        output = self.test_mod.pulse(self.test_input)
        breakpoint()
        self.assertIsInstance(self.test)

    # def test_    
    


    # def test_initialization(self):
    #     self.assertIsInstance(self.test_mod.sequence, Sequence)
    #     # self.assertEqual(self.test_mod.sequence, {'BaseMod_0': BaseMod(name='0')})
    #     self.assertEqual(self.test_mod.stats.mps, 1)
    # #     # Locked to always None - set by last item in sequence
    # #     self.assertIsNone(self.test_mod.stats.maxlen)
    # #     self.assertEqual(self.test_mod.stats.name, 'SeqMod_Test')
    # #     self.assertIsInstance(self.test_mod.stats)
    # #     self.assertEqual(self.test_mod._max_age, 60.)

    # # def test_invalid_name_type(self):
    #     TestBaseMod(self)

        