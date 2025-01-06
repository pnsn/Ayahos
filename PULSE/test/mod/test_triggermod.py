from unittest import TestCase

import numpy as np
from obspy import read, UTCDateTime
from seisbench.models import PhaseNet


from PULSE.data.dictstream import DictStream
from PULSE.data.pick import Trigger
from PULSE.mod.base import BaseMod
from PULSE.mod.trigger import TriggerMod


class TestTriggerMod(TestCase):
    # Create a basic prediction output
    _model = PhaseNet().from_pretrained('instance')
    _st = read()
    for tr in _st:
        tr.data = np.r_[tr.data[0], tr.data]
    _ast = _model.annotate(_st)
    _ast = _ast.trim(starttime=_st[0].stats.starttime,
                        endtime=_st[0].stats.endtime,
                        pad=True,
                        fill_value=None)
    for _tr in _ast:
        _tr.stats.channel= 'EH'+_tr.stats.component

    def setUp(self):
        self.data = DictStream(self._st.copy())
        self.pred = DictStream(self._ast.copy())

    
    def tearDown(self):
        del self.data
        del self.pred

    def test___init__(self):
        