import os
import pytest
from obspy import Trace, UTCDateTime, Stream
from obspy.core.tests.test_trace import TestTrace
from PULSE.test.data.util import *
from PULSE.data.foldtrace import FoldTrace
from PULSE.data.header import MLStats


class TestFoldTrace(TestTrace):

    def test_init_data(self):
        """Tests the __init__ method of the FoldTrace class
        for inputs to the **data** argument
        """
        # DATA INPUT TESTS
        data = load_logo_vector()

        # NumPy ndarray input
        ftr = FoldTrace(data = data)
        # Check length
        assert len(ftr) == 30
        # Check that class dtype matches input dtype
        assert ftr.dtype == data.dtype

        # NumPy masked array input
        data = np.ma.array(data=data,
                           mask=[False]*len(data),
                           fill_value=-999)
        data.mask[15:25] = True
        mftr = FoldTrace(data = data)
        # Check length
        assert len(mftr) == 30
        # Check that data is masked array
        assert isinstance(mftr.data, np.ma.MaskedArray)
        # Check that data dtype matches class dtype
        assert mftr.dtype == data.dtype
        
        # Trace input
        tr = load_logo_trace()
        ftr = FoldTrace(tr)
        # Check length
        assert len(ftr) == len(tr)
        # Check that FoldTrace.dtype is input Trace.data dtype
        assert ftr.dtype == tr.data.dtype

        # other data types will raise
        with pytest.raises(TypeError):
            FoldTrace(data=list(data))
        with pytest.raises(TypeError):
            FoldTrace(data=tuple(data))
        with pytest.raises(TypeError):
            FoldTrace(data='1234')

    def test_init_fold(self):
        # FOLD INPUT TESTS
        data = load_logo_vector()
        ftr = FoldTrace(data=data,
                        fold=np.ones(data.shape))
        # Check that fold dtype inherited from input data dtype
        assert ftr.fold.dtype == data.dtype
        
        # other fold types will raise
        with pytest.raises(TypeError):
            FoldTrace(data=np.ones(4),fold=[1,1,1,1])
        with pytest.raises(TypeError):
            FoldTrace(data=np.ones(4), fold=(1,1,1,1))
        with pytest.raises(TypeError):
            FoldTrace(data=np.ones(4), fold='1234')
        # other fold length will raise
        with pytest.raises(ValueError):
            FoldTrace(data=np.arange(4), fold=np.ones(5))

        ftr = FoldTrace(data.astype(np.float32))

    def test_init_header(self):
        # HEADER INPUT TESTS  
        tr = load_logo_trace()  
        header = tr.stats
        # Header from explicit input
        ftr = FoldTrace(header=header)
        assert isinstance(ftr.stats, MLStats)
        for _k in header.defaults.keys():
            assert header[_k] == ftr.stats[_k]

        # Header from implicit input via Trace
        ftr = FoldTrace(tr)
        # Check inherited metadata
        for _k in tr.stats.defaults.keys():
            assert tr.stats[_k] == ftr.stats[_k]

        # other types raises
        # TODO: Most of this should be handled in the PULSE.data.header tests
        with pytest.raises(ValueError):
            ftr = FoldTrace(header=['a'])

    def test_init_dtype(self):
        # DTYPE INPUT TESTS
        data = load_logo_trace()
        # Check explicit, conflicting assigned dtype
        data = data.astype(np.float64)
        # dtype overrides data dtype
        ftr = FoldTrace(data=data,
                        dtype=np.float32)
        # Check dtype matches assigned
        assert ftr.dtype == np.float32
        assert ftr.data.dtype == np.float32

        ftr = FoldTrace(dtype=int)
        assert ftr.dtype == int
        assert ftr.data.dtype == int
        assert ftr.fold.dtype == int

        # dtype overrides data and fold dtype
        ftr = FoldTrace(data=data,
                        fold=np.ones(shape=data.shape, dtype=np.int16),
                        dtype=np.float32)
        assert ftr.dtype == np.float32
        assert ftr.data.dtype == np.float32
        assert ftr.fold.dtype == np.float32

        # Check implicit assignment of dtype via data
        ftr = FoldTrace(data=data,
                        fold=np.ones(shape=data.shape, dtype=np.int16))
        assert ftr.dtype == data.dtype
        assert ftr.data.dtype == ftr.dtype
        assert ftr.fold.dtype == ftr.dtype

        # other dtype type raises
        with pytest.raises(TypeError):
            FoldTrace(dtype='abc')

        
        

    # def test_setattr(self):
    #     """Tests the __setattr__ method for FoldTrace
    #     """
    #     # NumPy ndaray
    #     ftr = FoldTrace()
    #     ftr.data = np.arange(4, dtype=np.float32)
    #     assert len(ftr) == 4
    #     assert len(ftr.data) == 4
    #     # Check that assigning data does not update fold values
    #     assert len(ftr.fold) == 0
    #     # Check that assigning fold must conform to fold_rules
    #     ftr.fold = np.ones(4, dtype=np.float64)
    #     assert len(ftr.fold) == 4
    #     assert ftr.fold.dtype == np.float32

    #     # NumPy ndarray with defined dtype
    #     ftr.data = np.arange(4, dtype=np.float32)
    #     assert len(ftr) == 4
    #     assert ftr.data.dtype == np.float32
    #     assert ftr.fold.dtype == np.float32
        
    #     # NumPy masked array
    #     ftr.data = np.ma.array([0,1,2,3,4],
    #                            mask=[True, False, False, True, False])
    #     ftr.fold = np.ones(5)
    #     # Check lengths
    #     assert len(ftr) == 5
    #     assert len(ftr.fold) == 5
    #     # Check that masked = fold -> 0 is enforced
    #     assert all(ftr.fold == np.array([0,1,1,0,1]))
    #     # Check that dtype is enforced
    #     assert ftr.fold.dtype == ftr.data.dtype

    #     # Other types will raise
    #     tr = FoldTrace()
    #     ftr = FoldTrace(data=np.arange(4))
    #     with pytest.raises(TypeError):
    #         tr.__setattr__('data', [0,1,2,3])
    #     with pytest.raises(TypeError):
    #         ftr.__setattr__('fold', [0,1,1,0])
    #     with pytest.raises(TypeError):
    #         tr.__setattr__('data', (0,1,2,3))
    #     with pytest.raises(TypeError):
    #         ftr.__setattr__('fold', (0, 1, 1, 0))
    #     with pytest.raises(TypeError):
    #         tr.__setattr__('data', '1234')
    #     with pytest.raises(TypeError):
    #         ftr.__setattr__('fold', '1234')
        
    #     # Incorrect shape of fold will raise
    #     with pytest.raises(ValueError):
    #         ftr.__setattr__('fold', np.ones(5))

    # def test_enforce_fold_rules(self):
    #     """Test suite for the _enforce_fold_rules method
    #     of FoldTrace
    #     """ 
    #     ftr = FoldTrace(load_logo_trace(), dtype=np.float64)
    #     # Test dtype adjustment
    #     assert ftr.fold.dtype == np.float64
    #     ftr.fold = ftr.fold.astype(dtype=np.float32)
    #     assert ftr.fold.dtype == np.float64


    # # def test_add_with_gap(self):
    # #     ftr = FoldTrace()


    # # def test_get_fold_view_trace(self):


