
from PULSE.mod.base import BaseMod
from PULSE.mod.sequence import SeqMod
from PULSE.test.mod.test_basemod import TestBaseMod



class TestSeqMod(TestBaseMod):

    def setUp(self):
        super().setUp()
    
        self.test_seq = {f'BaseMod_{_e}': BaseMod(name=str(_e)) for _e in range(3)}
        self.test_mod = SeqMod(name='Test')
    
    def test_initialization(self):
        self.assertIsInstance(self.test_mod.sequence, dict)
        self.assertEqual(self.test_mod.sequence, {'Base': BaseMod()})
        self.assertEqual(self.test_mod.stats.mps, 1)
        # Locked to always None - set by last item in sequence
        self.assertIsNone(self.test_mod.stats.maxlen)
        self.assertEqual(self.test_mod.stats.name, 'SeqMod_Test')
        self.assertIsInstance(self.test_mod.stats)
        self.assertEqual(self.test_mod._max_age, 60.)

        