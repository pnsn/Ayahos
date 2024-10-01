import os
import pytest
from obspy import Trace, UTCDateTime, Stream
from obspy.core.tests.test_trace import TestTrace
from PULSE.test.data.util import *
from PULSE.data.foldtrace import FoldTrace


class TestFoldTrace(TestTrace):

    def test_init(self):
        """Tests the __init__ method of the FoldTrace class.
        """
        data = load_logo_vector()
        tr = load_logo_trace()
        # NumPy ndarray input
        ftr = FoldTrace(data = data)
        # Check length
        assert len(ftr) == 30
        assert len(ftr.data) == 30
        assert len(ftr.fold) == 30

        # NumPy masked array input
        data = np.ma.array(data=data,
                           mask=[False]*len(data),
                           fill_value=-999)
        data.mask[15:25] = True
        mftr = FoldTrace(data = data)
        # Check length
        assert len(mftr) == 30
        assert len(mftr.data) == 30
        assert len(mftr.fold) == 30
        
        # Trace input
        ftr = FoldTrace(tr)
        # Check length
        assert len(ftr) == 30
        assert len(ftr.data) == 30
        assert len(ftr.fold) == 30
        # Check inherited metadata
        for _k in tr.stats.defaults.keys():
            assert tr.stats[_k] == ftr.stats[_k]

        # Check assign different dtype
        ftr = FoldTrace(data=data,
                        dtype=np.float32)
        # Check dtype matches assigned
        assert ftr.data.dtype == np.float32

        # other data types will raise
        with pytest.raises(ValueError):
            FoldTrace(data=list(data))
        with pytest.raises(ValueError):
            FoldTrace(data=tuple(data))
        with pytest.raises(ValueError):
            FoldTrace(data='1234')

        # other fold types will raise
        with pytest.raises(ValueError):
            FoldTrace(data=np.ones(4),fold=[1,1,1,1])
        with pytest.raises(ValueError):
            FoldTrace(data=np.ones(4), fold=(1,1,1,1))
        with pytest.raises(ValueError):
            FoldTrace(data=np.ones(4), fold='1234')

    def test_setattr(self):
        """Tests the __setattr__ method for FoldTrace
        """
        # NumPy ndaray
        ftr = FoldTrace()
        ftr.data = np.arange(4)
        assert len(ftr) == 4
        assert len(ftr.data) == 4
        assert len(ftr.fold) == 4
        assert all(ftr.fold == np.ones(4))

        # NumPy ndarray with defined dtype
        ftr.data = np.arange(4, dtype=np.float32)
        assert len(ftr) == 4
        assert ftr.data.dtype == np.float32
        assert ftr.fold.dtype == np.float32
        
        # NumPy masked array
        ftr.data = np.ma.array([0,1,2,3,4],
                               mask=[True, False, False, True, False])
        # Check lengths
        assert len(ftr) == 5
        assert len(ftr.fold) == 5
        # Check that masked = fold -> 0 is enforced
        assert ftr.fold == np.array([0,1,1,0,1], dtype=ftr.data.dtype)

        # Other types will raise
        tr = FoldTrace()
        ftr = FoldTrace(data=np.arange(4))
        with pytest.raises(ValueError):
            tr.__setattr__('data', [0,1,2,3])
        with pytest.raises(ValueError):
            ftr.__setattr__('fold', [0,1,1,0])
        with pytest.raises(ValueError):
            tr.__setattr__('data', (0,1,2,3))
        with pytest.raises(ValueError):
            ftr.__setattr__('fold', (0, 1, 1, 0))
        with pytest.raises(ValueError):
            tr.__setattr__('data', '1234')
        with pytest.raises(ValueError):
            ftr.__setattr__('fold', '1234')

    def test_len(self):
        foldtr = FoldTrace(data=np.arange(1000))
        assert len(foldtr) == 1000
        assert foldtr.count() == 1000

    def test_get_fold_view_trace(self):
        

