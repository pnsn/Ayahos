import unittest
from pathlib import Path
from collections import deque

import seisbench.models as sbm

from PULSE.mod.base import BaseMod
from PULSE.mod.sequencing import SeqMod
from PULSE.seq.sbm_picking import SBM_Picking_Sequence
from PULSE.data.dictstream import DictStream
from PULSE.test.example_data import load_townsend_example

class TestWindMod(unittest.TestCase):
    st, _, _ = load_townsend_example(Path().cwd())
    module = SBM_Picking_Sequence(model=sbm.EQTransformer())
    def setUp(self):
        self.ds = DictStream(self.st.copy())
    
    def tearDown(self):
        del self.ds


    def test__init__(self):
        mod = self.module
        self.assertIsInstance(mod, BaseMod)
        self.assertIsInstance(mod, SeqMod)
        self.assertIsInstance(mod, SBM_Picking_Sequence)
    
    def test_pulse(self):
        self.assertGreater(len(self.ds), 3)
        output = self.module.pulse(self.ds)
        breakpoint()