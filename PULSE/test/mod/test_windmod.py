import unittest
from pathlib import Path

from PULSE.mod.window import WindMod
from PULSE.data.dictstream import DictStream
from PULSE.test.example_data import load_seattle_example

class TestWindMod(unittest.TestCase):
    def setUp(self):
        st, _, _ = load_seattle_example(Path().cwd())
        self.st = st
        self.test_mod = WindMod()
