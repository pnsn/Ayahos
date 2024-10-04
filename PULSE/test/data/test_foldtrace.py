import os, pytest, warnings
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
        assert mftr.data.dtype == data.dtype
        
        # Trace input
        tr = load_logo_trace()
        ftr = FoldTrace(tr)
        # Check length
        assert len(ftr) == len(tr)
        # Check that FoldTrace.dtype is input Trace.data dtype
        assert ftr.data.dtype == tr.data.dtype

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
        ftr = FoldTrace(data=tr.data, header=header)
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


    def test_setattr_data(self):
        """Tests the __setattr__ method for FoldTrace
        """
        # NumPy ndaray
        ftr = FoldTrace()
        # Check default dtype
        assert ftr.data.dtype == np.float64
        # Set to different data
        ftr.data = np.arange(4, dtype=np.int64)
        assert len(ftr) == 4
        assert len(ftr.data) == 4
        assert ftr.dtype == np.int64
        # Check that assigning data does not update fold values
        assert len(ftr.fold) == 0
        # Check that assigning data updates fold dtype
        assert ftr.fold.dtype == np.int64
        # other type raisese
        with pytest.raises(TypeError):
            ftr.data = [1,2,3,4]
        with pytest.raises(TypeError):
            ftr.data = (1,2,3,4)
        with pytest.raises(TypeError):
            ftr.data = '1234'

    def test_setattr_fold(self):
        
        ftr = FoldTrace(data=np.arange(4, dtype=np.float64))
        # Explicit fold set
        ftr.fold = np.ones(4, dtype=np.int64)
        # Assert length
        assert len(ftr.fold) == 4
        # Assert reference dtype
        assert ftr.fold.dtype == np.float64
        # Assert still all ones
        assert all(ftr.fold==1)

    def test_add_trace_with_gap(self):
        # set up
        ftr1 = FoldTrace(data=np.arange(1000))
        ftr1.stats.sampling_rate = 200
        start = ftr1.stats.starttime
        ftr1.verify()

        ftr2 = FoldTrace(data=np.arange(0, 1000)[::-1])
        ftr2.stats.sampling_rate = 200
        ftr2.stats.starttime = start + 10   
        ftr2.verify()
        # Assemble output suite
        options = [ftr1 + ftr2,
                   ftr1.__add__(ftr2, fill_value=0),
                   ftr1.__add__(ftr2, method=2),
                   ftr1.__add__(ftr2, method=3)]
        with pytest.raises(NotImplementedError):
            ftr1.__add__(ftr2, method=1)

        for _e, ftr in enumerate(options):
            ftr.verify()
            # stats
            assert ftr.stats.starttime == start
            assert ftr.stats.endtime == start + 14.995
            assert ftr.stats.sampling_rate == 200
            assert ftr.stats.npts == 3000
            # dtype
            assert ftr.dtype == ftr1.data.dtype
            assert np.ma.is_masked(ftr.data)
            # fill value
            if _e == 1:
                assert ftr.data.fill_value == 0
            else:
                assert ftr.data.fill_value == 999999
            # data
            assert len(ftr) == 3000
            assert ftr[0] == 0
            assert ftr[999] == 999
            assert ftr[2000] == 999
            assert ftr[2999] == 0   
            # fold
            assert len(ftr.fold) == 3000
            assert ftr.fold[0] == 1
            assert ftr.fold[999] == 1
            assert ftr.fold[1001] == 0
            assert ftr.fold[1999] == 0
            assert ftr.fold[2000] == 1


    def test_add_with_overlap(self):
        # set up
        ftr1 = FoldTrace(data=np.arange(1000))
        ftr1.stats.sampling_rate = 200
        start = ftr1.stats.starttime
        assert all(ftr1.fold == 1)
        ftr2 = FoldTrace(data=np.arange(0, 1000)[::-1])
        ftr2.stats.sampling_rate = 200
        ftr2.stats.starttime = start + 4
        assert all(ftr2.fold == 1)
        ftr3 = ftr2.copy()
        ftr3.fold = ftr3.fold*2.
        assert all(ftr3.fold == 2)

        # Assemble output suite
        options = [ftr1 + ftr2,
                   ftr1.__add__(ftr2, fill_value=0),
                   ftr1.__add__(ftr2, method=2),
                   ftr1.__add__(ftr2, method=3),
                   ftr1.__add__(ftr3, method=3)]
        with pytest.raises(NotImplementedError):
            ftr1.__add__(ftr2, method=1)

        for _e, ftr in enumerate(options):
            ftr.verify()
        # stats
        assert ftr.stats.starttime == start
        assert ftr.stats.endtime == start + 8.995
        assert ftr.stats.sampling_rate == 200
        assert ftr.stats.npts == 1800
        # data
        assert len(ftr) == 1800
        assert ftr[0] == 0
        assert ftr[799] == 799
        if _e in [0,1]:
            assert ftr[800].mask
            assert ftr[999].mask
            if _e == 0:
                ftr.data.fill_value = 999999
            elif _e == 1:
                ftr.data.fill_value == 0
        elif _e == 2:
            ftr[800] == 1000
            ftr[999] == 1000
        elif _e == 3:
            ftr[800] == np.mean([1000,801])
            ftr[999] == np.mean([801,1000])
        elif _e == 4:
            ftr[800] == np.mean([801, 2000])
            ftr[999] == np.mean([801, 2000])
        assert ftr[1000] == 799
        assert ftr[1799] == 0
        # verify
        ftr.verify()           

    def test_add_same_trace(self):
        ftr1 = FoldTrace(data=np.arange(1001))
        options = [ftr1 + ftr1,
                   ftr1.__add__(ftr1, method=0),
                   ftr1.__add__(ftr1, method=2),
                   ftr1.__add__(ftr1, method=3)]
        for ftr in options:
            assert ftr.stats == ftr1.stats
            np.testing.assert_array_equal(ftr.data, ftr1.data)
            np.testing.assert_array_equal(ftr.fold, ftr1.fold*2)

    def test_add_within_trace(self):
        # set up
        ftr1 = FoldTrace(data=np.arange(1001))
        ftr1.stats.sampling_rate = 200
        start = ftr1.stats.starttime
        ftr2 = FoldTrace(data=np.arange(201))
        ftr2.stats.sampling_rate = 200
        ftr2.stats.starttime = start + 2
        ftr2.fold *= 2
        
        ftr3 = ftr1.copy()
        ftr3.data = ftr3.data.astype(np.float32)
        ftr4 = ftr2.copy()
        ftr4.data = ftr4.data.astype(np.float32)
        options = [ftr1 + ftr2,
                   ftr2 + ftr1,
                   ftr1.__add__(ftr2, method=0, fill_value=0),
                   ftr1.__add__(ftr2, method=2),
                   ftr1.__add__(ftr2, method=3),
                   ftr3.__add__(ftr4, method=3)]
        for _e, ftr in enumerate(options):
            assert ftr.stats == ftr1.stats
            # Dropout adding
            if _e in [0,1,2]:
                mask = np.zeros(len(ftr1)).astype(np.bool_)
                mask[400:601] = True
                np.testing.assert_array_equal(ftr.data.mask, mask)
                np.testing.assert_array_equal(ftr.data.data[:400], ftr1.data[:400])
                np.testing.assert_array_equal(ftr.data.data[601:], ftr1.data[601:])
                if _e == 2:
                    ftr.data.fill_value == 0
                else:
                    ftr.data.fill_value == 999999
            else:
                assert not isinstance(ftr.data, np.ma.MaskedArray)
                # Max stacking
                if _e == 3:
                    # assert that overlapping 
                    np.testing.assert_array_equal(ftr.data[400:601],ftr1.data[400:601])
                # Avg stacking
                if _e in [4,5]:
                    foldweighted=np.sum(np.c_[ftr2.data*2, ftr1.data[400:601]],axis=1)
                    foldweighted = foldweighted/3.
                    foldweighted = foldweighted.astype(ftr.dtype)
                    np.testing.assert_array_equal(ftr.data[400:601], foldweighted)

    def test_add_gap_and_overlap(self):
        # set up
        ftr1 = FoldTrace(data=np.arange(1000))
        ftr1.stats.sampling_rate = 200
        start = ftr1.stats.starttime
        ftr2 = FoldTrace(data=np.arange(1000)[::-1])
        ftr2.stats.sampling_rate = 200
        ftr2.stats.starttime = start + 4
        ftr3 = FoldTrace(data=np.arange(1000)[::-1])
        ftr3.stats.sampling_rate = 200
        ftr3.stats.starttime = start + 12
        # overlap
        overlap = ftr1 + ftr2
        assert len(overlap) == 1800
        mask = np.zeros(1800).astype(np.bool_)
        mask[800:1000] = True
        np.testing.assert_array_equal(overlap.data.mask, mask)
        # Check that all masked samples have fold = 0
        assert all(overlap.fold[overlap.data.mask] == 0)
        np.testing.assert_array_equal(overlap.data.data[:800], ftr1.data[:800])
        np.testing.assert_array_equal(overlap.data.data[1000:], ftr2.data[200:])
        # overlap + gap
        overlap_gap = overlap + ftr3
        assert len(overlap_gap) == 3400
        mask = np.zeros(3400).astype(np.bool_)
        mask[800:1000] = True
        mask[1800:2400] = True
        np.testing.assert_array_equal(overlap_gap.data.mask, mask)
        assert all(overlap_gap.fold[overlap_gap.data.mask]==0)
        np.testing.assert_array_equal(overlap_gap.data.data[:800],
                                      ftr1.data[:800])
        np.testing.assert_array_equal(overlap_gap.data.data[1000:1800],
                                      ftr2.data[200:])
        np.testing.assert_array_equal(overlap_gap.data.data[2400:], ftr3.data)
        # gap
        gap = ftr2 + ftr3
        assert len(gap) == 2600
        mask = np.zeros(2600).astype(np.bool_)
        mask[1000:1600] = True
        np.testing.assert_array_equal(gap.data.mask, mask)
        assert all(gap.fold[gap.data.mask]==0)
        np.testing.assert_array_equal(gap.data.data[:1000], ftr2.data)
        np.testing.assert_array_equal(gap.data.data[1600:], ftr3.data)

    def test_add_into_gap(self):
        """
        Test __add__ method of the Trace class
        Adding a trace that fits perfectly into gap in a trace
        """
        my_array = np.arange(6, dtype=np.int32)
        stats = load_logo_trace().stats
        start = stats['starttime']
        bigtrace = FoldTrace(data=np.array([], dtype=np.int32), header=stats)
        bigtrace_sort = bigtrace.copy()
        stats['npts'] = len(my_array)
        my_trace = FoldTrace(data=my_array, header=stats)

        stats['npts'] = 2
        ftr1 = FoldTrace(data=my_array[0:2].copy(), header=stats)
        stats['starttime'] = start + 2
        ftr2 = FoldTrace(data=my_array[2:4].copy(), header=stats)
        stats['starttime'] = start + 4
        ftr3 = FoldTrace(data=my_array[4:6].copy(), header=stats)

        bftr1 = bigtrace
        bftr2 = bigtrace_sort
        for method in [0, 2, 3]:
            # Random
            bigtrace = bftr1.copy()
            bigtrace = bigtrace.__add__(ftr1, method=method)
            bigtrace = bigtrace.__add__(ftr3, method=method)
            bigtrace = bigtrace.__add__(ftr2, method=method)

            # Sorted
            bigtrace_sort = bftr2.copy()
            bigtrace_sort = bigtrace_sort.__add__(ftr1, method=method)
            bigtrace_sort = bigtrace_sort.__add__(ftr2, method=method)
            bigtrace_sort = bigtrace_sort.__add__(ftr3, method=method)

            for ftr in (bigtrace, bigtrace_sort):
                assert isinstance(ftr, FoldTrace)
                assert not isinstance(ftr.data, np.ma.masked_array)
            # Assert all data are the same
            assert (bigtrace_sort.data == my_array).all()
            # Assert everything is the same via __eq__
            assert bigtrace_sort == my_trace
            # Assert all data are the same, despite different add order
            assert (bigtrace.data == my_array).all()
            # Assert everything is the same via __eq__ despite different __add__ order
            assert bigtrace == my_trace
            # Assert that all data types are the same
            for array_ in (bigtrace.data, bigtrace_sort.data):
                assert my_array.dtype == array_.dtype

    def test_iadd(self):
        # set up
        ftr0 = FoldTrace(data=np.arange(1000))
        ftr0.stats.sampling_rate = 200
        start = ftr0.stats.starttime
        assert all(ftr0.fold == 1)
        ftr1 = ftr0.copy()
        assert ftr0 == ftr1
        ftr2 = FoldTrace(data=np.arange(0, 1000)[::-1])
        ftr2.stats.sampling_rate = 200
        ftr2.stats.starttime = start + 2
        assert all(ftr2.fold == 1)

        ftr1 += ftr2
        # assert ftr0 != ftr1
        assert ftr1 == ftr0 + ftr2
        for method in [2,3]:
            ftr = ftr0.copy()
            ftr.__iadd__(ftr2, method=method)
            assert ftr == ftr0.__add__(ftr2, method=method)
    
    def test_eq(self):
        tr = load_logo_trace()
        ftr0 = FoldTrace(tr)
        # Assert mismatch type
        assert tr != ftr0
        # Assert identical
        ftr1 = ftr0.copy()
        assert ftr0 == ftr1
        # Assert mismatch stats
        ftr1.stats.network='OU'
        assert ftr0 != ftr1
        # Assert mismatch data
        ftr1 = ftr0.copy()
        ftr1.data = np.zeros(ftr0.data.shape)
        assert ftr0 != ftr1
        # Assert mismatch fold
        ftr1 = ftr0.copy()
        ftr1.fold = ftr1.fold*2
        assert ftr0 != ftr1
        # Assert mismatch dtype
        ftr1 = ftr0.copy()
        ftr1.data = ftr1.data.astype(np.float32)
        assert ftr0 != ftr1

    def test_detrend(self):
        """
        Test detrend method of trace
        """
        t = np.arange(10)
        data = 0.1 * t + 1.
        tr = FoldTrace(data=data.copy())

        tr.detrend(type='simple')
        # Assert data change
        np.testing.assert_array_almost_equal(tr.data, np.zeros(10))
        # Assert no fold change
        np.testing.assert_array_equal(tr.fold, np.ones(tr.count(), dtype=tr.dtype))
        tr.data = data.copy()
        tr.detrend(type='linear')
        np.testing.assert_array_almost_equal(tr.data, np.zeros(10))
        # Assert no fold change
        np.testing.assert_array_equal(tr.fold, np.ones(tr.count(), dtype=tr.dtype))

        data = np.zeros(10)
        data[3:7] = 1.

        tr.data = data.copy()
        tr.detrend(type='simple')
        np.testing.assert_almost_equal(tr.data[0], 0.)
        np.testing.assert_almost_equal(tr.data[-1], 0.)
        # Assert no fold change
        np.testing.assert_array_equal(tr.fold, np.ones(tr.count(), dtype=tr.dtype))
        tr.data = data.copy()
        tr.detrend(type='linear')
        np.testing.assert_almost_equal(tr.data[0], -0.4)
        np.testing.assert_almost_equal(tr.data[-1], -0.4)
        # Assert no fold change
        np.testing.assert_array_equal(tr.fold, np.ones(tr.count(), dtype=tr.dtype))

    def test_differentiate(self):
        """
        Test differentiation method of trace
        """
        t = np.linspace(0., 1., 11)
        data = 0.1 * t + 1.
        tr = FoldTrace(data=data)
        tr.stats.delta = 0.1
        tr.differentiate(method='gradient')
        np.testing.assert_array_almost_equal(tr.data, np.ones(11) * 0.1)
        # Assert no fold change
        np.testing.assert_array_equal(tr.fold, np.ones(tr.count(), dtype=tr.dtype))

    def test_integrate(self):
        """
        Test integration method of trace
        """
        data = np.ones(101) * 0.01
        tr = FoldTrace(data=data)
        tr.stats.delta = 0.1
        tr.integrate()
        # Assert time and length of resulting array.
        assert tr.stats.starttime == UTCDateTime(0)
        assert tr.stats.npts == 101
        np.testing.assert_array_almost_equal(
            tr.data, np.concatenate([[0.0], np.cumsum(data)[:-1] * 0.1]))        
        # Assert no fold change
        np.testing.assert_array_equal(tr.fold, np.ones(tr.count(), dtype=tr.dtype))

    def test_verify(self):
        tr = FoldTrace()
        tr.verify()
        tr = FoldTrace(data=np.array([1]))
        tr.verify()
        tr = load_townsend_example()[0]
        tr.verify()

    def test_get_fold_trace(self):
        data = np.arange(5, dtype=np.float64)
        tr = FoldTrace(data=data)
        assert tr.fold.dtype == np.float64
        trf = tr.get_fold_trace()
        # Assert trf data is expected fold
        np.testing.assert_array_equal(trf.data, np.ones(5, dtype=np.float64))
        # Assert trf data is tr fold
        np.testing.assert_array_equal(trf.data, tr.fold)
        # Update one value in trf.data
        trf.data[2] = 3
        trf.fold[2] = 0
        # Assert that update to view data is applied to source fold
        assert tr.fold[2] == 3
        # Assert that update to view fold does not affect source data
        assert tr.data[2] == 2
        # Assert that changing view stats does not affect source stats
        trf.stats.network = 'UO'
        assert tr.stats.network == ''

    def test_taper(self):
        data = np.ones(101, dtype=np.float64)*3
        tr = FoldTrace(data=data)
        tr.taper(max_percentage=0.05, type='cosine')
        # Assert all data values are at or below original values
        for _e in range(tr.count()):
            # Assert that taper does not accentuate data
            assert 0 <= tr.data[_e] <= 3
            # Assert that taper does not accentuate fold
            assert 0 <= tr.fold[_e] <= 1
            # Assert that taper is applied in-kind to data and fold
            if _e not in [0, tr.count() - 1]:
                assert np.abs(tr.data[_e]/tr.fold[_e] - 3) <= 1e-12
    
    def test_taper_onesided(self):
        # setup
        data = np.ones(11, dtype=np.float32)
        tr = FoldTrace(data=data)
        # overlong taper - raises UserWarning for both appications - ignoring
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", UserWarning)
            tr.taper(max_percentage=None, side="left")
        assert len(w) == 2
        for _w in w:
            assert _w.category == UserWarning

        assert tr.data[:5].sum() < 5.
        assert tr.fold[:5].sum() < 5.
        assert tr.data[6:].sum() == 5.
        assert tr.fold[6:].sum() == 5.

        data = np.ones(11, dtype=np.float32)
        tr = FoldTrace(data=data)

        # overlong taper - raises UserWarning for both applications - ignoring
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", UserWarning)
            tr.taper(max_percentage=None, side="right")
        assert len(w) == 2
        for _w in w:
            assert _w.category == UserWarning

        assert tr.data[:5].sum() == 5.
        assert tr.fold[:5].sum() == 5.
        assert tr.data[6:].sum() < 5.
        assert tr.fold[6:].sum() < 5.

    def test_taper_length(self):
        npts = 11
        type_ = "hann"

        data = np.ones(npts, dtype=np.float32)
        tr = FoldTrace(data=data, header={'sampling': 1.})

        # test an overlong taper request, still works but raises UserWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", UserWarning)
            tr.taper(max_percentage=0.7, max_length=int(npts / 2) + 1)
        assert len(w) == 2
        for _w in w:
            assert _w.category == UserWarning

        data = np.ones(npts)
        tr = FoldTrace(data=data, header={'sampling': 1.})
        # first 3 samples get tapered
        tr.taper(max_percentage=None, type=type_, side="left", max_length=3)
        # last 5 samples get tapered
        tr.taper(max_percentage=0.5, type=type_, side="right", max_length=None)
        assert np.all(tr.data[:3] < 1.)
        assert np.all(tr.data[3:6] == 1.)
        assert np.all(tr.data[6:] < 1.)

        assert np.all(tr.fold[:3] < 1.)
        assert np.all(tr.fold[3:6] == 1.)
        assert np.all(tr.fold[6:] < 1.)

        data = np.ones(npts, dtype=np.float32)
        tr = FoldTrace(data=data, header={'sampling': 1.})
        # first 3 samples get tapered
        tr.taper(max_percentage=0.5, type=type_, side="left", max_length=3)
        # last 3 samples get tapered
        tr.taper(max_percentage=0.3, type=type_, side="right", max_length=5)
        assert np.all(tr.data[:3] < 1.)
        assert np.all(tr.data[3:8] == 1.)
        assert np.all(tr.data[8:] < 1.)
        assert np.all(tr.fold[:3] < 1.)
        assert np.all(tr.fold[3:8] == 1.)
        assert np.all(tr.fold[8:] < 1.)

    def test_resample(self):
        data = np.arange(101, dtype=np.float32)
        tr = FoldTrace(data=data)
        tr2 = tr.copy()
        tr2.resample(2)
        breakpoint()