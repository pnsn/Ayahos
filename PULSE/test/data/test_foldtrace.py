"""
:module: PULSE.test.data.test_foldtrace
:auth: Nathan T. Stevens
:org: Pacific Northwest Seismic Network
:email: ntsteven (at) uw.edu
:license: AGPL-3.0
:purpose: This module contains unit tests for the :class:`~PULSE.data.foldtrace.FoldTrace` class.
    It builds on the ObsPy :mod:`~obspy.core.test.test_trace` module.
"""

import pytest, warnings
from obspy import UTCDateTime, Stream
from obspy.core.tests.test_trace import TestTrace
from PULSE.test.data.util import *
from PULSE.data.foldtrace import FoldTrace
from PULSE.data.header import MLStats

class TestFoldTrace(TestTrace):
    #########################
    ## __init__ TEST SUITE ##
    #########################
    def test_init_data(self):
        """Tests the __init__ method of the FoldTrace class
        for inputs to the **data** argument
        """
        # DATA INPUT TESTS
        data = load_logo_vector()

        # NumPy ndarray input
        tr = FoldTrace(data = data)
        # Check length
        assert len(tr) == 30
        # Check that class dtype matches input dtype
        assert tr.dtype == data.dtype

        # NumPy masked array input
        data = np.ma.array(data=data,
                           mask=[False]*len(data),
                           fill_value=-999)
        data.mask[15:25] = True
        mtr = FoldTrace(data = data)
        # Check length
        assert len(mtr) == 30
        # Check that data is masked array
        assert isinstance(mtr.data, np.ma.MaskedArray)
        # Check that data dtype matches class dtype
        assert mtr.data.dtype == data.dtype
        
        # Trace input
        tr = load_logo_trace()
        tr = FoldTrace(tr)
        # Check length
        assert len(tr) == len(tr)
        # Check that FoldTrace.dtype is input Trace.data dtype
        assert tr.data.dtype == tr.data.dtype

        # other data types will raise
        with pytest.raises(TypeError):
            FoldTrace(data=list(data))
        with pytest.raises(TypeError):
            FoldTrace(data=tuple(data))
        with pytest.raises(TypeError):
            FoldTrace(data='1234')

    def test_init_fold(self):
        """Test suite for the __init__ method of FoldTrace
        related to kwarg **fold**
        """        
        # FOLD INPUT TESTS
        data = load_logo_vector()
        tr = FoldTrace(data=data,
                        fold=np.ones(data.shape))
        # Check that fold dtype inherited from input data dtype
        assert tr.fold.dtype == data.dtype
        
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
        with pytest.raises(ValueError):
            FoldTrace(fold=np.arange(5))
        tr = FoldTrace(data.astype(np.float32))

    def test_init_header(self):
        """Test suite for the __init__ method of FoldTrace
        related to kwarg **header**
        """        
        # HEADER INPUT TESTS  
        tr = load_logo_trace()  
        header = tr.stats
        # Header from explicit input of Stats
        ftr = FoldTrace(header=header)
        assert isinstance(ftr.stats, MLStats)
        for _k in header.defaults.keys():
            if _k not in ['npts','endtime']:
                assert header[_k] == ftr.stats[_k]
            else:
                assert header[_k] != ftr.stats[_k]
        header = {'network': 'UW', 'station': 'GNW', 'location': '--', 'channel':'HHN'}
        ftr = FoldTrace(header=header)
        for _k in header.keys():
            assert header[_k] == ftr.stats[_k]
        # Header from implicit input via Trace
        ftr = FoldTrace(data=tr)
        # Check inherited metadata
        for _k in tr.stats.defaults.keys():
            assert ftr.stats[_k] == tr.stats[_k]

        # other types raises
        # TODO: Most of this should be handled in the PULSE.data.header tests
        with pytest.raises(TypeError):
            ftr = FoldTrace(header=['a'])
    
    ############################
    ## __setattr__ TEST SUITE ##
    ############################   
    def test_setattr_data(self):
        """Tests the __setattr__ method for FoldTrace
        related to key **data**
        """
        # NumPy ndaray
        tr = FoldTrace()
        # Check default dtype
        assert tr.data.dtype == np.float64
        # Set to different data
        tr.data = np.arange(4, dtype=np.int64)
        assert len(tr) == 4
        assert len(tr.data) == 4
        assert tr.dtype == np.int64
        # Check that assigning data does not update fold values
        assert len(tr.fold) == 0
        # Check that assigning data updates fold dtype
        assert tr.fold.dtype == np.int64
        # other type raisese
        with pytest.raises(TypeError):
            tr.data = [1,2,3,4]
        with pytest.raises(TypeError):
            tr.data = (1,2,3,4)
        with pytest.raises(TypeError):
            tr.data = '1234'

    def test_setattr_fold(self):
        """Test suite for the __setattr__ method of FoldTrace
        related to key **fold**
        """        
        tr = FoldTrace(data=np.arange(4, dtype=np.float64))
        # Explicit fold set
        tr.fold = np.ones(4, dtype=np.int64)
        # Assert length
        assert len(tr.fold) == 4
        # Assert reference dtype
        assert tr.fold.dtype == np.float64
        # Assert still all ones
        assert all(tr.fold==1)

    def test_astype(self):
        """Test suite for the astype method of FoldTrace
        """        
        tr = FoldTrace(data=np.arange(4), dtype=np.float32)
        assert tr.dtype == np.float32
        assert tr.data.dtype == np.float32
        assert tr.fold.dtype == np.float32
        # Test selection of dtypes
        for _dt in [None, 'f8', int, np.float32]:
            tr2 = tr.copy().astype(_dt)
            if _dt is None:
                assert tr2.dtype == tr.dtype
                assert tr2.data.dtype == tr.data.dtype
                assert tr2.fold.dtype == tr.fold.dtype
            else:
                assert tr2.dtype == _dt
                assert tr2.data.dtype == _dt
                assert tr2.fold.dtype == _dt

    ########################
    ## __add__ TEST SUITE ##
    ########################
    def test_add_trace_with_gap(self):

        # set up
        tr1 = FoldTrace(data=np.arange(1000, dtype=np.float64))
        tr1.stats.sampling_rate = 200
        start = tr1.stats.starttime
        tr1.verify()

        tr2 = FoldTrace(data=np.arange(0, 1000, dtype=np.float64)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 10   
        tr2.verify()
        # Assemble output suite
        options = [tr1 + tr2,
                   tr1.__add__(tr2, fill_value=0),
                   tr1.__add__(tr2, method=2),
                   tr1.__add__(tr2, method=3)]
        with pytest.raises(NotImplementedError):
            tr1.__add__(tr2, method=1)

        for _e, tr in enumerate(options):
            tr.verify()
            # stats
            assert tr.stats.starttime == start
            assert tr.stats.endtime == start + 14.995
            assert tr.stats.sampling_rate == 200
            assert tr.stats.npts == 3000
            # dtype
            assert tr.dtype == tr1.data.dtype
            assert np.ma.is_masked(tr.data)
            # fill value
            if _e == 1:
                assert tr.data.fill_value == 0
            # data
            assert len(tr) == 3000
            assert tr[0] == 0
            assert tr[999] == 999
            assert tr[2000] == 999
            assert tr[2999] == 0   
            # fold
            assert len(tr.fold) == 3000
            assert tr.fold[0] == 1
            assert tr.fold[999] == 1
            assert tr.fold[1001] == 0
            assert tr.fold[1999] == 0
            assert tr.fold[2000] == 1

    def test_add_with_overlap(self):
        # set up
        tr1 = FoldTrace(data=np.arange(1000))
        tr1.stats.sampling_rate = 200
        start = tr1.stats.starttime
        assert all(tr1.fold == 1)
        tr2 = FoldTrace(data=np.arange(0, 1000)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 4
        assert all(tr2.fold == 1)
        tr3 = tr2.copy()
        tr3.fold = tr3.fold*2.
        assert all(tr3.fold == 2)

        # Assemble output suite
        options = [tr1 + tr2,
                   tr1.__add__(tr2, fill_value=0),
                   tr1.__add__(tr2, method=2),
                   tr1.__add__(tr2, method=3),
                   tr1.__add__(tr3, method=3)]
        with pytest.raises(NotImplementedError):
            tr1.__add__(tr2, method=1)

        for _e, tr in enumerate(options):
            tr.verify()
        # stats
        assert tr.stats.starttime == start
        assert tr.stats.endtime == start + 8.995
        assert tr.stats.sampling_rate == 200
        assert tr.stats.npts == 1800
        # data
        assert len(tr) == 1800
        assert tr[0] == 0
        assert tr[799] == 799
        if _e in [0,1]:
            assert tr[800].mask
            assert tr[999].mask
            if _e == 0:
                tr.data.fill_value = 999999
            elif _e == 1:
                tr.data.fill_value == 0
        elif _e == 2:
            tr[800] == 1000
            tr[999] == 1000
        elif _e == 3:
            tr[800] == np.mean([1000,801])
            tr[999] == np.mean([801,1000])
        elif _e == 4:
            tr[800] == np.mean([801, 2000])
            tr[999] == np.mean([801, 2000])
        assert tr[1000] == 799
        assert tr[1799] == 0
        # verify
        tr.verify()           

    def test_add_same_trace(self):
        tr1 = FoldTrace(data=np.arange(1001))
        options = [tr1 + tr1,
                   tr1.__add__(tr1, method=0),
                   tr1.__add__(tr1, method=2),
                   tr1.__add__(tr1, method=3)]
        for tr in options:
            assert tr.stats == tr1.stats
            np.testing.assert_array_equal(tr.data, tr1.data)
            np.testing.assert_array_equal(tr.fold, tr1.fold*2)

    def test_add_within_trace(self):
        # set up
        tr1 = FoldTrace(data=np.arange(1001))
        tr1.stats.sampling_rate = 200
        start = tr1.stats.starttime
        tr2 = FoldTrace(data=np.arange(201))
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 2
        tr2.fold *= 2
        
        tr3 = tr1.copy()
        tr3.data = tr3.data.astype(np.float32)
        tr4 = tr2.copy()
        tr4.data = tr4.data.astype(np.float32)
        options = [tr1 + tr2,
                   tr2 + tr1,
                   tr1.__add__(tr2, method=0, fill_value=0),
                   tr1.__add__(tr2, method=2),
                   tr1.__add__(tr2, method=3),
                   tr3.__add__(tr4, method=3)]
        for _e, tr in enumerate(options):
            assert tr.stats == tr1.stats
            # Dropout adding
            if _e in [0,1,2]:
                mask = np.zeros(len(tr1)).astype(np.bool_)
                mask[400:601] = True
                np.testing.assert_array_equal(tr.data.mask, mask)
                np.testing.assert_array_equal(tr.data.data[:400], tr1.data[:400])
                np.testing.assert_array_equal(tr.data.data[601:], tr1.data[601:])
                if _e == 2:
                    tr.data.fill_value == 0
                else:
                    tr.data.fill_value == 999999
            else:
                assert not isinstance(tr.data, np.ma.MaskedArray)
                # Max stacking
                if _e == 3:
                    # assert that overlapping 
                    np.testing.assert_array_equal(tr.data[400:601],tr1.data[400:601])
                # Avg stacking
                if _e in [4,5]:
                    foldweighted=np.sum(np.c_[tr2.data*2, tr1.data[400:601]],axis=1)
                    foldweighted = foldweighted/3.
                    foldweighted = foldweighted.astype(tr.dtype)
                    np.testing.assert_array_equal(tr.data[400:601], foldweighted)

    def test_add_gap_and_overlap(self):
        # set up
        tr1 = FoldTrace(data=np.arange(1000))
        tr1.stats.sampling_rate = 200
        start = tr1.stats.starttime
        tr2 = FoldTrace(data=np.arange(1000)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 4
        tr3 = FoldTrace(data=np.arange(1000)[::-1])
        tr3.stats.sampling_rate = 200
        tr3.stats.starttime = start + 12
        # overlap
        overlap = tr1 + tr2
        assert len(overlap) == 1800
        mask = np.zeros(1800).astype(np.bool_)
        mask[800:1000] = True
        np.testing.assert_array_equal(overlap.data.mask, mask)
        # Check that all masked samples have fold = 0
        assert all(overlap.fold[overlap.data.mask] == 0)
        np.testing.assert_array_equal(overlap.data.data[:800], tr1.data[:800])
        np.testing.assert_array_equal(overlap.data.data[1000:], tr2.data[200:])
        # overlap + gap
        overlap_gap = overlap + tr3
        assert len(overlap_gap) == 3400
        mask = np.zeros(3400).astype(np.bool_)
        mask[800:1000] = True
        mask[1800:2400] = True
        np.testing.assert_array_equal(overlap_gap.data.mask, mask)
        assert all(overlap_gap.fold[overlap_gap.data.mask]==0)
        np.testing.assert_array_equal(overlap_gap.data.data[:800],
                                      tr1.data[:800])
        np.testing.assert_array_equal(overlap_gap.data.data[1000:1800],
                                      tr2.data[200:])
        np.testing.assert_array_equal(overlap_gap.data.data[2400:], tr3.data)
        # gap
        gap = tr2 + tr3
        assert len(gap) == 2600
        mask = np.zeros(2600).astype(np.bool_)
        mask[1000:1600] = True
        np.testing.assert_array_equal(gap.data.mask, mask)
        assert all(gap.fold[gap.data.mask]==0)
        np.testing.assert_array_equal(gap.data.data[:1000], tr2.data)
        np.testing.assert_array_equal(gap.data.data[1600:], tr3.data)

    def test_add_into_gap(self):
        """
        Test __add__ method of the Trace class
        Adding a trace that fits perfectly into gap in a trace
        """
        my_array = np.arange(6, dtype=np.float64)
        stats = dict(load_logo_trace().stats)
        start = stats['starttime']
        bigtrace = FoldTrace(data=np.array([]), header=stats)
        bigtrace_sort = bigtrace.copy()
        stats['npts'] = len(my_array)
        my_trace = FoldTrace(data=my_array, header=stats)

        stats['npts'] = 2
        tr1 = FoldTrace(data=my_array[0:2].copy(), header=stats)
        stats['starttime'] = start + 2
        tr2 = FoldTrace(data=my_array[2:4].copy(), header=stats)
        stats['starttime'] = start + 4
        tr3 = FoldTrace(data=my_array[4:6].copy(), header=stats)

        btr1 = bigtrace
        btr2 = bigtrace_sort
        for method in [0, 2, 3]:
            # Random
            bigtrace = btr1.copy()
            bigtrace = bigtrace.__add__(tr1, method=method)
            bigtrace = bigtrace.__add__(tr3, method=method)
            bigtrace = bigtrace.__add__(tr2, method=method)

            # Sorted
            bigtrace_sort = btr2.copy()
            bigtrace_sort = bigtrace_sort.__add__(tr1, method=method)
            bigtrace_sort = bigtrace_sort.__add__(tr2, method=method)
            bigtrace_sort = bigtrace_sort.__add__(tr3, method=method)

            for tr in (bigtrace, bigtrace_sort):
                assert isinstance(tr, FoldTrace)
                assert not isinstance(tr.data, np.ma.masked_array)
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

    ########################################
    ## SELF CHECK/COMPARITOR METHOD TESTS ##
    ########################################
    def test_eq(self):
        tr = load_logo_trace()
        tr0 = FoldTrace(data=tr)
        # Assert mismatch type
        assert tr != tr0
        # Assert identical
        tr1 = tr0.copy()
        assert tr0 == tr1
        # Assert mismatch stats
        tr1.stats.network='OU'
        assert tr0 != tr1
        # Assert mismatch data
        tr1 = tr0.copy()
        tr1.data = np.zeros(tr0.data.shape)
        assert tr0 != tr1
        # Assert mismatch fold
        tr1 = tr0.copy()
        tr1.fold = tr1.fold*2
        assert tr0 != tr1
        # Assert mismatch dtype
        tr1 = tr0.copy()
        tr1.data = tr1.data.astype(np.float32)
        assert tr0 != tr1
    
    def test_verify(self):
        tr = FoldTrace()
        tr.verify()
        tr = FoldTrace(data=np.array([1]))
        tr.verify()
        tr = FoldTrace(load_townsend_example()[0])
        tr.verify()

    #############################
    ## VIEW-BASED METHOD TESTS ##
    #############################
    def test_get_view(self):
        tr = FoldTrace(data=np.arange(10))
        # Test None inputs
        view = tr.get_view()
        assert tr == view
        # Test starttime within source time domain
        view = tr.get_view(starttime = tr.stats.starttime + 2)
        assert view.count() == 8
        assert view.stats.starttime == tr.stats.starttime + 2
        assert view.stats.endtime == tr.stats.endtime
        # Test starttime at start
        view = tr.get_view(starttime = tr.stats.starttime)
        assert tr == view
        # Test starttime before start
        view = tr.get_view(starttime = tr.stats.starttime - 2)
        assert tr == view
        # Test specified endtime within domain
        view = tr.get_view(endtime = tr.stats.endtime - 1)
        assert view.count() == 9
        assert view.stats.starttime == tr.stats.starttime
        assert view.stats.endtime == tr.stats.endtime - 1
        # Test endtime at end
        view = tr.get_view(endtime = tr.stats.endtime)
        assert tr == view
        # Test endtime after end
        view = tr.get_view(endtime = tr.stats.endtime + 1)
        assert tr == view
        # Specify both start and endtime
        view = tr.get_view(starttime = tr.stats.starttime + 2,
                           endtime = tr.stats.endtime - 2)
        assert view.count() == 6
        assert view.stats.starttime == tr.stats.starttime + 2
        assert view.stats.endtime == tr.stats.endtime - 2
        assert all(view.data == tr.copy().trim(starttime = view.stats.starttime,
                                           endtime = view.stats.endtime))
        # Assert that modifying data in view modifies source data
        tr_bu = tr.copy()
        view.data[0] += 1
        assert view.data[0] == tr_bu.data[2] + 1
        assert view.data[0] == tr.data[2]
        view.fold[0] += 1
        assert view.fold[0] == tr_bu.fold[2] + 1
        assert view.fold[0] == tr.fold[2]

    def test_get_fold_trace(self):
        data = np.arange(5, dtype=np.float64)
        tr = FoldTrace(data=data)
        assert tr.fold.dtype == np.float64
        trf = tr._get_fold_trace()
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

    #################################
    ## DATA MODIFYING METHOD TESTS ##
    #################################
    def test_detrend(self):
        """
        Test detrend method of trace
        """
        t = np.arange(10)
        data = 0.1 * t + 1.
        tr = FoldTrace(data=data.copy())
        tr2 = tr.copy().detrend(type='simple')
        # Assert data change
        np.testing.assert_array_almost_equal(tr2.data, np.zeros(10))
        assert len(tr2.stats.processing) == 1
        # Assert no fold change
        np.testing.assert_array_equal(tr2.fold, np.ones(tr2.count(), dtype=tr2.dtype))
        tr2 = tr.copy().detrend(type='linear')
        np.testing.assert_array_almost_equal(tr2.data, np.zeros(10))
        assert len(tr2.stats.processing) == 1
        # Assert no fold change
        np.testing.assert_array_equal(tr2.fold, np.ones(tr2.count(), dtype=tr2.dtype))

        data = np.zeros(10)
        data[3:7] = 1.

        tr.data = data.copy()
        tr2 = tr.copy().detrend(type='simple')
        np.testing.assert_almost_equal(tr2.data[0], 0.)
        np.testing.assert_almost_equal(tr2.data[-1], 0.)
        # Assert no fold change
        np.testing.assert_array_equal(tr2.fold, np.ones(tr2.count(), dtype=tr2.dtype))
        tr2 = tr.copy().detrend(type='linear')
        assert len(tr2.stats.processing) == 1
        np.testing.assert_almost_equal(tr2.data[0], -0.4)
        np.testing.assert_almost_equal(tr2.data[-1], -0.4)
        # Assert no fold change
        np.testing.assert_array_equal(tr2.fold, np.ones(tr2.count(), dtype=tr2.dtype))

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

    ##################################
    ## TRIMING/PADDING METHOD TESTS ##
    ##################################
    def test_ltrim(self):
        # Setup
        tr = FoldTrace(data=np.arange(101))
        tr.fold[50:] = 2
        start = tr.stats.starttime
        # shortening trim
        tr2 = tr.copy()._ltrim(start + 1)
        assert tr2.count() == 100
        assert all(tr2.data == tr.data[1:])
        assert all(tr2.fold[:49] == 1)
        assert all(tr2.fold[49:] == 2)
        # shortening to not nearest sample
        tr2 = tr.copy()._ltrim(start + 1.25, nearest_sample=False)
        assert tr2.count() == 99
        assert all(tr2.data == tr.data[2:])
        assert all(tr2.fold[:48] == 1)
        assert all(tr2.fold[48:] == 2)
        # padding trim without padding enabled
        tr2 = tr.copy()._ltrim(start - 2)
        np.testing.assert_array_equal(tr2.data, tr.data)
        # padding trim with padding enabled
        tr2 = tr.copy()._ltrim(start - 2, pad=True)
        assert tr2.count() == 103
        assert all(tr2.data[2:] == tr.data)
        assert np.ma.is_masked(tr2.data)
        assert all(tr2.fold[:2] == 0)
        assert all(tr2.fold[2:52] == 1)
        assert all(tr2.fold[52:] == 2)
        # padding trim with specified fill_value
        tr2 = tr.copy()._ltrim(start - 2, pad=True, fill_value=80)
        assert not np.ma.is_masked(tr2.data)
        assert all(tr2.data[:2] == 80)
        assert all(tr2.data[2:] == tr.data)

    def test_rtrim(self):
        # Setup
        tr = FoldTrace(data=np.arange(101))
        tr.fold[50:] = 2
        end = tr.stats.endtime
        # shortening trim
        tr2 = tr.copy()._rtrim(end - 1)
        assert tr2.count() == 100
        assert all(tr2.data == tr.data[:-1])
        assert all(tr2.fold[:50] == 1)
        assert all(tr2.fold[50:] == 2)
        # shortening to not nearest sample
        tr2 = tr.copy()._rtrim(end - 1.25, nearest_sample=False)
        assert tr2.count() == 99
        assert all(tr2.data == tr.data[:-2])
        assert all(tr2.fold[:50] == 1)
        assert all(tr2.fold[50:] == 2)
        # padding trim without padding enabled
        tr2 = tr.copy()._rtrim(end + 2)
        np.testing.assert_array_equal(tr2.data, tr.data)
        # padding trim with padding enabled
        tr2 = tr.copy()._rtrim(end + 2, pad=True)
        assert tr2.count() == 103
        assert all(tr2.data[:-2] == tr.data)
        assert np.ma.is_masked(tr2.data)
        assert all(tr2.fold[-2:] == 0)
        assert all(tr2.fold[:50] == 1)
        assert all(tr2.fold[50:-2] == 2)
        # padding trim with specified fill_value
        tr2 = tr.copy()._rtrim(end + 2, pad=True, fill_value=80)
        assert not np.ma.is_masked(tr2.data)
        assert all(tr2.data[-2:] == 80)
        assert all(tr2.data[:-2] == tr.data)    

    def test_trim(self):
        tr = FoldTrace(data=np.arange(101))
        tr.fold[50:] = 2
        start = tr.stats.starttime
        end = tr.stats.endtime
        # Non-trim
        tr2 = tr.copy().trim()
        assert all(tr2.data == tr.data)
        assert all(tr2.fold == tr.fold)
        # ltrim imbedded call
        tr2 = tr.copy().trim(starttime=start + 2)
        assert all(tr2.data == tr.data[2:])
        assert all(tr2.fold == tr.fold[2:])
        # rtrim imbedded call
        tr2 = tr.copy().trim(endtime=end - 2)
        assert all(tr2.data == tr.data[:-2])
        assert all(tr2.fold == tr.fold[:-2])
        # combined call
        tr2 = tr.copy().trim(starttime = start + 2,
                             endtime = end - 2)
        assert all(tr2.data == tr.data[2:-2])
        assert all(tr2.fold == tr.fold[2:-2])
        # combined call with nearest_sample
        tr2 = tr.copy().trim(starttime = start + 1.25,
                             endtime = end - 0.01)
        assert all(tr2.data == tr.data[1:])
        assert all(tr2.fold == tr.fold[1:])
        # combined call without nearest_sample
        tr2 = tr.copy().trim(starttime = start + 1.25,
                             endtime = end - 0.01,
                             nearest_sample=False)
        assert all(tr2.data == tr.data[2:-1])
        assert all(tr2.fold == tr.fold[2:-1])
        # combined call with padding
        tr2 = tr.copy().trim(starttime = start - 2,
                             endtime = end + 3,
                             pad=True)
        assert np.ma.is_masked(tr2.data)
        assert tr2.count() == 101+5
        # combined call with padding and fill_value
        tr2 = tr.copy().trim(starttime = start - 2,
                             endtime = end + 3,
                             pad=True,
                             fill_value = 999)
        assert not np.ma.is_masked(tr2.data)
        assert tr2.count() == 101+5
        assert all(tr2.data[:2] == 999)
        assert all(tr2.data[-3:] == 999)
        assert all(tr2.data[2:-3] == tr.data)
        assert all(tr2.fold[:2] == 0)
        assert all(tr2.fold[-2:] == 0)
        assert all(tr2.fold[2:-3] == tr.fold)

    def test_split(self):
        # Setup
        tr1 = FoldTrace(data=np.arange(10))
        tr2 = tr1.copy()
        tr3 = tr1.copy()
        tr2.stats.starttime += 20
        tr3.stats.starttime += 40
        gappy_tr = tr1 + tr2 + tr3
        # Test split with internal gaps
        split_st = gappy_tr.copy().split()
        assert isinstance(split_st, Stream)
        assert len(split_st) == 3
        assert split_st[0] == tr1
        assert split_st[1] == tr2
        assert split_st[2] == tr3
        # Test split with contiguous data
        split1 = tr1.copy().split()
        assert isinstance(split1, Stream)
        assert split1[0] == tr1
        # Test split with trailing masked values
        tr1m = tr1.copy()
        tr1m.data = np.ma.MaskedArray(data=tr1m.data,
                                      mask=[False]*10)
        tr1m.data.mask[8:] = True
        split1 = tr1m.split()
        assert isinstance(split1, Stream)
        assert len(split1) == 1
        assert not isinstance(split1[0], np.ma.MaskedArray)
        assert not np.ma.is_masked(split1[0].data)
        assert split1[0].count() == 8

        # Test ascopy
        gtr = gappy_tr.copy()
        # Test ascopy = False (the split stream contains views)
        gst1 = gtr.split(ascopy=False)
        for _tr in gst1:
            _tr.data[0] += 1
        for _e in [0,20,40]:
            assert gtr.data[_e] == gappy_tr.data[_e] + 1
        # Test ascopy = True for completeness
        gtr = gappy_tr.copy()
        gst2 = gtr.split(ascopy=True)
        for _tr in gst2:
            _tr.data[0] += 1
        for _e in [0, 20, 40]:
            assert gtr.data[_e] == gappy_tr.data[_e]

    ######################
    ## TAPER TEST SUITE ##
    ######################
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
        
    ###########################
    ## RESAMPLING TEST SUITE ##
    ###########################
    def test_interp_fold(self):
        """Test suite for the _interp_fold private method of FoldTrace
        used to augment inherited data resampling methods from Trace
        """        
        # Setup
        tr = FoldTrace(data=np.arange(6), fold=np.array([1,1,1,2,2,2]), dtype=np.float32)
        tr2 = tr.copy()
        tr2.data = np.linspace(0,6,11)
        tr2.stats.sampling_rate = 2
        assert tr.stats.starttime == tr2.stats.starttime
        assert tr.stats.endtime == tr2.stats.endtime
        # Check that verify fails on Exception (data and fold shape don't match)
        with pytest.raises(Exception):
            tr2.verify()
        # Apply method
        tr3 = tr2.copy()
        tr3._interp_fold(tr.stats.starttime, 1)
        assert tr3.data.shape == tr3.fold.shape
        # Test with slight non-aligned samples
        tr3 = tr2.copy()
        tr3.stats.starttime -= 0.1
        tr3._interp_fold(tr.stats.starttime, 1)
        assert tr3.verify()
        # Test with large non-aligned samples
        tr3 = tr2.copy()
        tr3.stats.starttime -= 1.37
        tr3.data = np.arange(7, dtype=tr3.dtype)
        tr3._interp_fold(tr.stats.starttime, 1)
        assert tr3.verify()


    def test_enforce_time_domain(self):
        # Setup
        ft = FoldTrace(data=np.arange(6), fold=np.array([1,1,1,2,2,2]), dtype=np.float32)
        ft2 = ft.copy()
        ft2.data = np.linspace(0,6,11)
        ft2.fold = np.array([1,1,1,1,1,1.5,2,2,2,2,2])
        # Test rtrim effect
        ft3 = ft2.copy()._enforce_time_domain(ft.stats)
        assert ft3.stats.endtime <= ft.stats.endtime
        assert ft3.stats.starttime >= ft.stats.starttime
        # Test ltrim effect
        ft4 = ft.copy()
        ft4.stats.starttime -= 2
        ft4._enforce_time_domain(ft.stats)
        assert ft4.stats.endtime <= ft.stats.endtime
        assert ft4.stats.starttime >= ft.stats.starttime



    def test_resample(self):
        """Test suite for the resample method of FoldTrace
        TODO: complete this test suite
        """        
        ft = FoldTrace(load_logo_trace())
        ft.fold[15:] = 2
        assert ft.stats.sampling_rate == 1.
        assert ft.stats.npts == 30
        # downsample by factor of 2
        ft2 = ft.copy().resample(sampling_rate=0.5)
        assert ft2.stats.endtime == ft.stats.endtime - 1.0
        assert ft2.stats.sampling_rate == 0.5
        assert ft2.stats.starttime == ft.stats.starttime
        assert ft2.fold.shape == ft2.data.shape
        # downsample by factor of 10
        ft3 = ft.copy().resample(sampling_rate=0.1)
        assert ft3.stats.endtime == ft.stats.endtime - 9.0
        assert ft3.stats.starttime == ft.stats.starttime
        assert ft3.stats.sampling_rate == 0.1
        assert ft3.fold.shape == ft3.data.shape

        # upsample by factor of 2 and don't enforce time domain
        ft4 = ft.copy().resample(sampling_rate=2, enforce_time_domain=False)
        # Updated sampling rate
        assert ft4.stats.sampling_rate == 2.
        # Unchanged starttime
        assert ft4.stats.starttime == ft.stats.starttime
        # Changed endtime
        assert ft4.stats.endtime != ft.stats.endtime

        # upsample by factor of 2
        ft5 = ft.copy().resample(sampling_rate=2)
        assert ft4.stats.starttime == ft.stats.starttime
        # enforced time domain check
        assert ft4.stats.endtime >= ft.stats.endtime >= ft5.stats.endtime
        # changed sampling rate
        assert ft5.stats.sampling_rate == 2.
        # shape of fold and data are consistent
        assert ft5.fold.shape == ft5.data.shape

        # exception raised with gappy data
        ft.data = np.ma.MaskedArray(data=ft.data,
                                     mask=[False]*30)
        ft.mask[10:15] = True
        with pytest.raises(Exception):
            ft.resample(3)


    def test_interpolate(self):
        """Test suite for interpolate method of FoldTrace
        TODO: Add other tests from the test_trace.py suite for interpolate?
        """        
        ft = FoldTrace(load_logo_trace())
        ft.fold[15:] = 2
        assert ft.stats.sampling_rate == 1.
        assert ft.stats.npts == 30
        # upsample by an integer factor
        ft2 = ft.copy().interpolate(sampling_rate=2.)
        assert ft2.stats.endtime == ft.stats.endtime
        assert ft2.stats.starttime == ft.stats.starttime
        assert ft2.stats.sampling_rate == 2.
        assert ft2.data.shape == ft2.fold.shape
        # upsample by a float factor
        ft3 = ft.copy().interpolate(sampling_rate=11.5)
        assert ft3.stats.starttime == ft.stats.starttime
        assert ft3.stats.endtime <= ft.stats.endtime
        assert ft3.stats.sampling_rate == 11.5
        assert ft3.data.shape == ft3.fold.shape

        # exception raised with gappy data
        ft.data = np.ma.MaskedArray(data=ft.data,
                                     mask=[False]*30)
        ft.mask[10:15] = True
        with pytest.raises(Exception):
            ft.interpolate(3)

    def test_decimate(self):
        """_summary_
        """        



    # def test_interpolate(self):
    #     """Test suite for the interpolate method of FoldTrace
    #     TODO: complete this test suite
    #     """
    #     ft = FoldTrace(load_logo_trace())
    #     assert ft.stats.sampling_rate == 1.
    #     assert ft.stats.npts == 30

    #     breakpoint()
    #     tr = FoldTrace(data = np.arange(100), dtype=np.float64)
    #     tr.fold[50:] = 2
    #     test_target_data = np.linspace(0,99,200)
    #     tr2 = tr.copy().interpolate(2)
    #     assert tr2.data.shape == tr2.fold.shape
    #     # np.testing.assert_array_equal(tr2.data, test_target)

    ############################
    ## TEST INTERPOLATE SUITE ##
    ############################
    # def test_interpolate_fold(self):
    #     # Populate FoldTrace
    #     tr = FoldTrace(load_townsend_example()[0])
    #     tr.fold[10:] = 2

    #     tr2 = tr.copy()
    #     # Test upsampling effects on 
    #     tr2= tr.copy().interpolate(2.3)
    #     assert all(tr2.fold == 1)
    #     # Test upsampling degrading fold density
    #     tr2 = tr.copy().interpolate(2.3)
    #     assert all(tr2.fold == 1/2.3)
    #     # Test downsampling not degrading fold density
    #     tr2 = tr.copy().interpolate(0.5)
    #     assert all(tr2.fold == 1)
    #     # Test upsampling resulting in intermediate values
    #     tr.fold[51:] = 2
    #     tr2 = tr.copy().interpolate(10)
    #     assert all(tr2.fold[:500] == 1)
    #     assert all(tr2.fold[501:510] < 2) and all(tr2.fold[501:510] > 1)
    #     assert all(tr2.fold[510:] == 2)
    #     # Raises NotImplemented for masked data
    #     tr = FoldTrace(data=np.ma.MaskedArray(data=np.arange(101),
    #                                           mask=[False]*101))
    #     tr.data.mask[20:30] = True
    #     with pytest.raises(NotImplementedError):
    #         tr2 = tr.copy().interpolate(10)

    # def test_interpolate(self):
    #     """
    #     Tests the interpolate function.

    #     This also tests the interpolation in obspy.signal. No need to repeat
    #     the same test twice I guess.
    #     """
    #     # Load the prepared data. The data has been created using SAC.
    #     file_ = "interpolation_test_random_waveform_delta_0.01_npts_50.sac"
    #     # Load as FoldTrace
    #     org_tr = FoldTrace(read("/path/to/%s" % file_, round_sampling_interval=False)[0])
    #     # Set half of fold to 2
    #     org_tr.fold[25:] = 2
    #     file_ = "interpolation_test_interpolated_delta_0.003.sac"
    #     interp_delta_0_003 = FoldTrace(read(
    #         "/path/to/%s" % file_, round_sampling_interval=False)[0])
    #     file_ = "interpolation_test_interpolated_delta_0.077.sac"
    #     interp_delta_0_077 = FoldTrace(read(
    #         "/path/to/%s" % file_, round_sampling_interval=False)[0])
    #     # Perform the same interpolation as in Python with ObsPy.
    #     int_tr = org_tr.copy().interpolate(sampling_rate=1.0 / 0.003,
    #                                        method="weighted_average_slopes")
    #     # Assert that the sampling rate has been set correctly.
    #     assert int_tr.stats.delta == 0.003
    #     # Assert that the new end time is smaller than the old one. SAC at
    #     # times performs some extrapolation which we do not want to do here.
    #     assert int_tr.stats.endtime <= org_tr.stats.endtime
    #     # SAC extrapolates a bit which we don't want here. The deviations
    #     # to SAC are likely due to the fact that we use double precision
    #     # math while SAC uses single precision math.
    #     assert np.allclose(
    #         int_tr.data,
    #         interp_delta_0_003.data[:int_tr.stats.npts],
    #         rtol=1e-3)
    #     # Assert that fold is interpolated when upsampled
    #     assert all(int_tr.fold[:81] == 1)
    #     assert all(int_tr.fold[81:84] < 2)
    #     assert all(int_tr.fold[81:84] > 1)
    #     assert all(int_tr.fold[84:] == 2)

    #     int_tr = org_tr.copy().interpolate(sampling_rate=1.0 / 0.077,
    #                                        method="weighted_average_slopes")
    #     # Assert that the sampling rate has been set correctly.
    #     assert int_tr.stats.delta == 0.077
    #     # Assert that the new end time is smaller than the old one. SAC
    #     # calculates one sample less in this case.
    #     assert int_tr.stats.endtime <= org_tr.stats.endtime
    #     assert np.allclose(
    #         int_tr.data[:interp_delta_0_077.stats.npts],
    #         interp_delta_0_077.data,
    #         rtol=1E-5)
    #     # Assert that fold is trimmed when downsampled
    #     assert all(int_tr.fold[:4] == 1)
    #     assert all(int_tr.fold[4:] == 2)

        
    #     # Also test the other interpolation methods mainly by assuring the
    #     # correct SciPy function is called and everything stays internally
    #     # consistent. SciPy's functions are tested enough to be sure that
    #     # they work.
    #     for inter_type in ["linear", "nearest", "zero"]:
    #         int_tr = org_tr.copy().interpolate(sampling_rate=10, method=inter_type)
    #         assert int_tr.stats.delta == 0.1
    #         assert int_tr.stats.endtime <= org_tr.stats.endtime
    #         np.testing.assert_array_equal(int_tr.fold, np.array([1,1,1,2,2]))

    #     for inter_type in ['slinear','quadratic','cubic',1,2,3]:
    #         int_tr = org_tr.copy().interpolate(sampling_rate=10, method=inter_type)
    #         assert int_tr.stats.delta == 0.1
    #         assert int_tr.stats.endtime <= org_tr.stats.endtime
    #         np.testing.assert_array_equal(int_tr.fold, np.array([1,1,1,2,2]))
            

    # # # def test_interpolation_time_shift(self):
    # # #     """
    # # #     Tests the time shift of the interpolation.
    # # #     TODO: Update this test suite
    # # #     """
    # # #     tr = FoldTrace(read()[0])
    # # #     tr.stats.sampling_rate = 1.0
    # # #     tr.data = tr.data[:500]
    # # #     tr.interpolate(method="lanczos", sampling_rate=10.0, a=20)
    # # #     tr.stats.sampling_rate = 1.0
    # # #     tr.data = tr.data[:500]
    # # #     tr.fold = tr.fold[:500]

    # # #     org_tr = tr.copy()

    # # #     # Now this does not do much for now but actually just shifts the
    # # #     # samples.
    # # #     tr.interpolate(method="lanczos", sampling_rate=1.0, a=1,
    # # #                    time_shift=0.2)
    # # #     assert tr.stats.starttime == org_tr.stats.starttime + 0.2
    # # #     assert tr.stats.endtime == org_tr.stats.endtime + 0.2
    # # #     np.testing.assert_allclose(tr.data, org_tr.data, atol=1E-9)
    # # #     np.testing.assert_allclose(tr.fold, org_tr.fold, atol=1E-9)

    # # #     tr.interpolate(method="lanczos", sampling_rate=1.0, a=1,
    # # #                    time_shift=0.4)
    # # #     assert tr.stats.starttime == org_tr.stats.starttime + 0.6
    # # #     assert tr.stats.endtime == org_tr.stats.endtime + 0.6
    # # #     np.testing.assert_allclose(tr.data, org_tr.data, atol=1E-9)

    # # #     tr.interpolate(method="lanczos", sampling_rate=1.0, a=1,
    # # #                    time_shift=-0.6)
    # # #     assert tr.stats.starttime == org_tr.stats.starttime
    # # #     assert tr.stats.endtime == org_tr.stats.endtime
    # # #     np.testing.assert_allclose(tr.data, org_tr.data, atol=1E-9)

    # # #     # This becomes more interesting when also fixing the sample
    # # #     # positions. Then one can shift by subsample accuracy while leaving
    # # #     # the sample positions intact. Note that there naturally are some
    # # #     # boundary effects and as the interpolation method does not deal
    # # #     # with any kind of extrapolation you will lose the first or last
    # # #     # samples.
    # # #     # This is a fairly extreme example but of course there are errors
    # # #     # when doing an interpolation - a shift using an FFT is more accurate.
    # # #     tr.interpolate(method="lanczos", sampling_rate=1.0, a=50,
    # # #                    starttime=tr.stats.starttime + tr.stats.delta,
    # # #                    time_shift=0.2)
    # # #     # The sample point did not change but we lost the first sample,
    # # #     # as we shifted towards the future.
    # # #     assert tr.stats.starttime == org_tr.stats.starttime + 1.0
    # # #     assert tr.stats.endtime == org_tr.stats.endtime
    # # #     # The data naturally also changed.
    # # #     with pytest.raises(AssertionError):
    # # #         np.testing.assert_allclose(tr.data, org_tr.data[1:], atol=1E-9)
    # # #     # Shift back. This time we will lose the last sample.
    # # #     tr.interpolate(method="lanczos", sampling_rate=1.0, a=50,
    # # #                    starttime=tr.stats.starttime,
    # # #                    time_shift=-0.2)
    # # #     assert tr.stats.starttime == org_tr.stats.starttime + 1.0
    # # #     assert tr.stats.endtime == org_tr.stats.endtime - 1.0
    # # #     # But the data (aside from edge effects - we are going forward and
    # # #     # backwards again so they go twice as far!) should now again be the
    # # #     # same as we started out with.
    # # #     np.testing.assert_allclose(
    # # #         tr.data[100:-100], org_tr.data[101:-101], atol=1e-9, rtol=1e-4)

    # # # def test_interpolation_arguments(self):
    # # #     """
    # # #     Test case for the interpolation arguments.
    # # #     """
    # # #     tr = FoldTrace(read()[0])
    # # #     tr.stats.sampling_rate = 1.0
    # # #     tr.data = tr.data[:50]
    # # #     tr.fold = tr.fold[:50]
    # # #     tr.fold[25:] = 2

    # # #     # TODO: Add fold tests

    # # #     for inter_type in ["linear", "nearest", "zero", "slinear",
    # # #                        "quadratic", "cubic", 1, 2, 3,
    # # #                        "weighted_average_slopes"]:
    # # #         # If only the sampling rate is specified, the end time will be very
    # # #         # close to the original end time but never bigger.
    # # #         interp_tr = tr.copy().interpolate(sampling_rate=0.3,
    # # #                                           method=inter_type)
    # # #         assert tr.stats.starttime == interp_tr.stats.starttime
    # # #         assert tr.stats.endtime >= interp_tr.stats.endtime >= \
    # # #                tr.stats.endtime - (1.0 / 0.3)

    # # #         # If the starttime is modified the new starttime will be used but
    # # #         # the end time will again be modified as little as possible.
    # # #         interp_tr = tr.copy().interpolate(sampling_rate=0.3,
    # # #                                           method=inter_type,
    # # #                                           starttime=tr.stats.starttime +
    # # #                                           5.0)
    # # #         assert tr.stats.starttime + 5.0 == interp_tr.stats.starttime
    # # #         assert tr.stats.endtime >= interp_tr.stats.endtime >= \
    # # #                tr.stats.endtime - (1.0 / 0.3)

    # # #         # If npts is given it will be used to modify the end time.
    # # #         interp_tr = tr.copy().interpolate(sampling_rate=0.3,
    # # #                                           method=inter_type, npts=10)
    # # #         assert tr.stats.starttime == interp_tr.stats.starttime
    # # #         assert interp_tr.stats.npts == 10

    # # #         # If npts and starttime are given, both will be modified.
    # # #         interp_tr = tr.copy().interpolate(sampling_rate=0.3,
    # # #                                           method=inter_type,
    # # #                                           starttime=tr.stats.starttime +
    # # #                                           5.0, npts=10)
    # # #         assert tr.stats.starttime + 5.0 == interp_tr.stats.starttime
    # # #         assert interp_tr.stats.npts == 10

    # # #         # An earlier starttime will raise an exception. No extrapolation
    # # #         # is supported
    # # #         with pytest.raises(ValueError):
    # # #             tr.copy().interpolate(sampling_rate=1.0,
    # # #                                   starttime=tr.stats.starttime - 10.0)
    # # #         # As will too many samples that would overstep the end time bound.
    # # #         with pytest.raises(ValueError):
    # # #             tr.copy().interpolate(sampling_rate=1.0,
    # # #                                   npts=tr.stats.npts * 1E6)

    # # #         # A negative or zero desired sampling rate should raise.
    # # #         with pytest.raises(ValueError):
    # # #             tr.copy().interpolate(sampling_rate=0.0)
    # # #         with pytest.raises(ValueError):
    # # #             tr.copy().interpolate(sampling_rate=-1.0)

    # # # def test_resample_new(self):
    # # #     """
    # # #     Tests if Trace.resample works as expected and test that issue #857 is
    # # #     resolved.
    # # #     """
    # # #     starttime = UTC("1970-01-01T00:00:00.000000Z")
    # # #     tr0 = Trace(np.sin(np.linspace(0, 2 * np.pi, 10)),
    # # #                 {'sampling_rate': 1.0,
    # # #                  'starttime': starttime})
    # # #     # downsample
    # # #     tr = tr0.copy()
    # # #     tr.resample(0.5, window='hann', no_filter=True)
    # # #     assert len(tr.data) == 5
    # # #     expected = np.array([0.19478735, 0.83618307, 0.32200221,
    # # #                          -0.7794053, -0.57356732])
    # # #     assert np.all(np.abs(tr.data - expected) < 1e-7)
    # # #     assert tr.stats.sampling_rate == 0.5
    # # #     assert tr.stats.delta == 2.0
    # # #     assert tr.stats.npts == 5
    # # #     assert tr.stats.starttime == starttime
    # # #     assert tr.stats.endtime == \
    # # #            starttime + tr.stats.delta * (tr.stats.npts - 1)

    # # #     # upsample
    # # #     tr = tr0.copy()
    # # #     tr.resample(2.0, window='hann', no_filter=True)
    # # #     assert len(tr.data) == 20
    # # #     assert tr.stats.sampling_rate == 2.0
    # # #     assert tr.stats.delta == 0.5
    # # #     assert tr.stats.npts == 20
    # # #     assert tr.stats.starttime == starttime
    # # #     assert tr.stats.endtime == \
    # # #            starttime + tr.stats.delta * (tr.stats.npts - 1)

    # # #     # downsample with non integer ratio
    # # #     tr = tr0.copy()
    # # #     tr.resample(0.75, window='hann', no_filter=True)
    # # #     assert len(tr.data) == int(10 * .75)
    # # #     expected = np.array([0.15425413, 0.66991128, 0.74610418, 0.11960477,
    # # #                          -0.60644662, -0.77403839, -0.30938935])
    # # #     assert np.all(np.abs(tr.data - expected) < 1e-7)
    # # #     assert tr.stats.sampling_rate == 0.75
    # # #     assert tr.stats.delta == 1 / 0.75
    # # #     assert tr.stats.npts == int(10 * .75)
    # # #     assert tr.stats.starttime == starttime
    # # #     assert tr.stats.endtime == \
    # # #            starttime + tr.stats.delta * (tr.stats.npts - 1)

    # # #     # downsample without window
    # # #     tr = tr0.copy()
    # # #     tr.resample(0.5, window=None, no_filter=True)
    # # #     assert len(tr.data) == 5
    # # #     assert tr.stats.sampling_rate == 0.5
    # # #     assert tr.stats.delta == 2.0
    # # #     assert tr.stats.npts == 5
    # # #     assert tr.stats.starttime == starttime
    # # #     assert tr.stats.endtime == \
    # # #            starttime + tr.stats.delta * (tr.stats.npts - 1)

    # # #     # downsample with window and automatic filtering
    # # #     tr = tr0.copy()
    # # #     tr.resample(0.5, window='hann', no_filter=False)
    # # #     assert len(tr.data) == 5
    # # #     assert tr.stats.sampling_rate == 0.5
    # # #     assert tr.stats.delta == 2.0
    # # #     assert tr.stats.npts == 5
    # # #     assert tr.stats.starttime == starttime
    # # #     assert tr.stats.endtime == \
    # # #            starttime + tr.stats.delta * (tr.stats.npts - 1)

    # # #     # downsample with custom window
    # # #     tr = tr0.copy()
    # # #     window = np.ones((tr.stats.npts))
    # # #     tr.resample(0.5, window=window, no_filter=True)

    # # #     # downsample with bad window
    # # #     tr = tr0.copy()
    # # #     window = np.array([0, 1, 2, 3])
    # # #     with pytest.raises(ValueError):
    # # #         tr.resample(sampling_rate=0.5, window=window, no_filter=True)

    # # # def test_resample(self):
    # # #     data = np.arange(101, dtype=np.float32)
    # # #     tr = FoldTrace(data=data)
    # # #     tr2 = tr.copy()
    # # #     tr2.resample(2.3)
    # # #     breakpoint()