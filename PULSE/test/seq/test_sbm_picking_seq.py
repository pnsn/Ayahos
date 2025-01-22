import unittest
from pathlib import Path
from collections import deque

import seisbench.models as sbm

from PULSE.mod.base import BaseMod
from PULSE.mod.sequencing import SeqMod
from PULSE.seq.sbm_picking import SBM_Picking_Sequence
from PULSE.data.dictstream import DictStream
from PULSE.test.example_data import load_seattle_example

class TestWindMod(unittest.TestCase):
    st, _, _ = load_seattle_example(Path().cwd())
    model = sbm.EQTransformer()

    def setUp(self):
        self.ds = DictStream(self.st.copy())
        self.instruments = self.ds.split_on().keys()
        self.test_mod = SBM_Picking_Sequence(model=self.model)

    
    def tearDown(self):
        del self.ds
        del self.test_mod
        del self.instruments

    def test__init__(self):
        seq = self.test_mod
        self.assertIsInstance(seq, BaseMod)
        self.assertIsInstance(seq, SeqMod)
        self.assertIsInstance(seq, SBM_Picking_Sequence)