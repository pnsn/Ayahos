import unittest
from pathlib import Path

from PULSE.mod.window import WindMod
from PULSE.data.dictstream import DictStream
from PULSE.test.example_data import load_seattle_example

class TestWindMod(unittest.TestCase):
    st, _, _ = load_seattle_example(Path().cwd())
    def setUp(self):
        self.ds = DictStream(self.st.copy())
        self.test_mod = WindMod()
    
    def tearDown(self):
        del self.ds
        del self.test_mod

    def test___init__(self):
        self.assert