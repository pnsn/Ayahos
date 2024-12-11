"""
:module: PULSE.test.data.test_foldtrace
:auth: Nathan T. Stevens
:org: Pacific Northwest Seismic Network
:email: ntsteven (at) uw.edu
:license: AGPL-3.0
:purpose: This module contains unit tests for the :class:`~PULSE.data.foldtrace.FoldTrace` class.
    It builds on the ObsPy :mod:`~obspy.core.test.test_trace` module.
"""

import pytest
from obspy import UTCDateTime, Stream
from obspy.core.tests.test_trace import TestTrace
from PULSE.test.data.util import *
from PULSE.data.foldtrace import FoldTrace
from PULSE.util.header import MLStats



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
        # give options on what masking value is applied
        if _e in [0,1]:
            assert tr[800].mask
            assert tr[999].mask
            # Default masking for integer dtypes
            if _e == 0:
                tr.data.fill_value = 999999
            # Prescribed masking value (must be integer in this test)
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
        ft = FoldTrace(data=tr)
        # Assert mismatch type
        assert tr != ft
        # Assert identical
        ft1 = ft.copy()
        assert ft == ft1
        # Assert mismatch stats
        ft1.stats.network='OU'
        assert ft != ft1
        # Assert mismatch data
        ft1 = ft.copy()
        ft1.data = np.zeros(ft.data.shape)
        assert ft != ft1
        # Assert mismatch fold
        ft1 = ft.copy()
        ft1.fold = ft1.fold*2
        assert ft != ft1
        # Assert mismatch dtype
        ft1 = ft.copy()
        ft1.data = ft1.data.astype(np.float32)
        assert ft != ft1
    
    def test_verify(self):
        tr = FoldTrace()
        tr.verify()
        tr = FoldTrace(data=np.array([1]))
        tr.verify()
        tr = FoldTrace(load_townsend_example()[0][0])
        tr.verify()

    #############################
    ## VIEW-BASED METHOD TESTS ##
    #############################
    def test_view(self):
        ft = FoldTrace(data=np.arange(10), header={'sampling_rate': 2.})
        # Test None inputs
        view = ft.view()
        assert ft == view
        # Test starttime within source time domain
        view = ft.view(starttime = ft.stats.starttime + 2)
        assert view.count() == 6
        assert view.stats.starttime == ft.stats.starttime + 2
        assert view.stats.endtime == ft.stats.endtime
        # Test starttime at start
        view = ft.view(starttime = ft.stats.starttime)
        assert ft == view
        # Test starttime before start
        view = ft.view(starttime = ft.stats.starttime - 2)
        assert ft == view
        # Test specified endtime within domain
        view = ft.view(endtime = ft.stats.endtime - 1)
        assert view.count() == 8
        assert view.stats.starttime == ft.stats.starttime
        assert view.stats.endtime == ft.stats.endtime - 1
        # Test endtime at end
        view = ft.view(endtime = ft.stats.endtime)
        assert ft == view
        # Test endtime after end
        view = ft.view(endtime = ft.stats.endtime + 1)
        assert ft == view
        # Specify both start and endtime
        view = ft.view(starttime = ft.stats.starttime + 2,
                           endtime = ft.stats.endtime - 2)
        assert view.count() == 2
        assert view.stats.starttime == ft.stats.starttime + 2
        assert view.stats.endtime == ft.stats.endtime - 2
        assert all(view.data == ft.copy().trim(starttime = view.stats.starttime,
                                           endtime = view.stats.endtime))
        # Assert that modifying data in view modifies source data
        ft_bu = ft.copy()
        view = ft.view()
        view.data[0] += 1
        assert view.data[0] == ft_bu.data[0] + 1
        assert view.data[0] == ft.data[0]
        view.fold[0] += 1
        assert view.fold[0] == ft_bu.fold[0] + 1
        assert view.fold[0] == ft.fold[0]

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
        # Setup
        t = np.linspace(0., 1., 11)
        data = 0.1 * t + 1.
        ft = FoldTrace(data=data)
        tr = Trace(data=data)
        ft.stats.delta = 0.1
        tr.stats.delta = 0.1
        # Apply method
        ftd = ft.copy().differentiate(method='gradient')
        trd = tr.copy().differentiate(method='gradient')
        # Assert identical result for obspy equivalent
        np.testing.assert_array_almost_equal(ftd.data, trd.data)
        # Assert no fold change
        np.testing.assert_array_equal(ftd.fold, ft.fold)


    def test_integrate(self):
        """
        Test integration method of trace
        """
        data = np.ones(101) * 0.01
        ft = FoldTrace(data=data)
        ft.stats.delta = 0.1
        ft.integrate()
        # Assert time and length of resulting array.
        assert ft.stats.starttime == UTCDateTime(0)
        assert ft.stats.npts == 101
        np.testing.assert_array_almost_equal(
            ft.data, np.concatenate([[0.0], np.cumsum(data)[:-1] * 0.1]))        
        # Assert no fold change
        np.testing.assert_array_equal(ft.fold, np.ones(ft.count(), dtype=ft.dtype))

    ##################################
    ## TRIMING/PADDING METHOD TESTS ##
    ##################################
    def test_ltrim(self):
        # Setup
        ft = FoldTrace(data=np.arange(101))
        ft.fold[50:] = 2
        start = ft.stats.starttime
        # shortening ftim
        ft2 = ft.copy()._ltrim(start + 1)
        assert ft2.count() == 100
        assert all(ft2.data == ft.data[1:])
        assert all(ft2.fold[:49] == 1)
        assert all(ft2.fold[49:] == 2)
        # shortening to not nearest sample
        ft2 = ft.copy()._ltrim(start + 1.25, nearest_sample=False)
        assert ft2.count() == 99
        assert all(ft2.data == ft.data[2:])
        assert all(ft2.fold[:48] == 1)
        assert all(ft2.fold[48:] == 2)
        # padding trim without padding enabled
        ft2 = ft.copy()._ltrim(start - 2)
        np.testing.assert_array_equal(ft2.data, ft.data)
        # padding trim with padding enabled
        ft2 = ft.copy()._ltrim(start - 2, pad=True)
        assert ft2.count() == 103
        assert all(ft2.data[2:] == ft.data)
        assert not np.ma.is_masked(ft2.data)
        # NOTE: This 999999 value is the default fill_value for integer dtypes
        assert all(ft2.data[:2]==999999)
        assert all(ft2.fold[:2] == 0)
        assert all(ft2.fold[2:52] == 1)
        assert all(ft2.fold[52:] == 2)
        # padding trim with specified fill_value
        ft2 = ft.copy()._ltrim(start - 2, pad=True, fill_value=80)
        assert not np.ma.is_masked(ft2.data)
        assert all(ft2.data[:2] == 80)
        assert all(ft2.data[2:] == ft.data)
        # Padding trim without apply_fill
        ft2 = ft.copy()._ltrim(start - 2, pad=True, fill_value=80, apply_fill=False)
        assert np.ma.is_masked(ft2.data)
        assert ft2.data.fill_value == 80
        assert all(ft2.data.mask[:2])
        assert not any(ft2.data.mask[2:])

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
        assert not np.ma.is_masked(tr2.data)
        assert all(tr2.data[-2:] == 999999)
        assert all(tr2.fold[-2:] == 0)
        assert all(tr2.fold[:50] == 1)
        assert all(tr2.fold[50:-2] == 2)
        # padding trim with specified fill_value
        tr2 = tr.copy()._rtrim(end + 2, pad=True, fill_value=80)
        assert not np.ma.is_masked(tr2.data)
        assert all(tr2.data[-2:] == 80)
        assert all(tr2.data[:-2] == tr.data)    
        # Padding trim without apply_fill
        tr2= tr.copy()._rtrim(end + 2, pad=True, fill_value=80, apply_fill=False)
        assert np.ma.is_masked(tr2.data)
        assert tr2.data.fill_value == 80
        assert all(tr2.data.mask[-2:])
        assert not any(tr2.data.mask[:-2])


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
        assert not np.ma.is_masked(tr2.data)
        assert all(tr2.data[:2] == 999999)
        assert all(tr2.data[-3:] == 999999)
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
        # Padding without apply_fill
        tr2 = tr.copy().trim(starttime = start - 2,
                             endtime = end + 3,
                             pad=True,
                             fill_value = 0.,
                             apply_fill=False)
        assert np.ma.is_masked(tr2.data)
        assert tr2.data.fill_value == 0.

    def test_trim_on_masked(self):
        tr = read()[0]
        tr.data = np.ma.MaskedArray(
            data = tr.data,
            mask = [False]*tr.count(),
            fill_value=-999.
        )
        tr.data.mask[1000:1500] = True
        ft = FoldTrace(tr)
        # Make sure masking -> 0 fold is enforced
        assert all(ft.fold[1000:1500] == 0)
        for fv in [0., 0, None, 999]:
            # applying fill values
            ft2 = ft.copy().trim(
                endtime = tr.stats.endtime + 10,
                fill_value = fv,
                pad=True)
            # not applying fill values
            ft3 = ft.copy().trim(
                endtime = tr.stats.endtime + 10,
                pad=True,
                fill_value = fv,
                apply_fill = False
            )
            # Assert that ft2 is not masked
            assert not np.ma.is_masked(ft2.data)
            # Assert that ft3 is masked
            assert np.ma.is_masked(ft3.data)
            # Assert that both foldtraces have the same fold vectors
            np.testing.assert_array_equal(ft2.fold, ft3.fold)
            # Assert that the fill values in ft2 match the fill_value for ft3
            if fv is not None:
                assert ft3.data.fill_value == fv
            else:
                # NOTE: This value is the default fill_value for floating dtypes
                assert ft3.data.fill_value == 1e20
            assert all(ft2.data[ft3.data.mask] == ft3.data.fill_value)

            
           



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
        ft = FoldTrace(data=data)
        ft.taper(max_percentage=0.05, type='cosine')
        # Assert all data values are at or below original values
        for _e in range(ft.count()):
            # Assert that taper does not accentuate data
            assert 0 <= ft.data[_e] <= 3
            # Assert that taper does not accentuate fold
            assert 0 <= ft.fold[_e] <= 1
            # Assert that taper is applied in-kind to data and fold
            if _e not in [0, ft.count() - 1]:
                assert np.abs(ft.data[_e]/ft.fold[_e] - 3) <= 1e-12
        
    def test_taper_onesided(self):
        # setup
        data = np.ones(11, dtype=np.float32)
        ft = FoldTrace(data=data)
        # Apply left taper
        ft.taper(max_percentage=None, side="left")
        assert ft.data[:5].sum() < 5.
        assert ft.fold[:5].sum() < 5.
        assert ft.data[6:].sum() == 5.
        assert ft.fold[6:].sum() == 5.

        # setup
        data = np.ones(11, dtype=np.float32)
        ft = FoldTrace(data=data)
        # Apply right taper
        ft.taper(max_percentage=None, side="right")
        assert ft.data[:5].sum() == 5.
        assert ft.fold[:5].sum() == 5.
        assert ft.data[6:].sum() < 5.
        assert ft.fold[6:].sum() < 5.
        
    def test_taper_length(self):
        npts = 11
        type_ = "hann"

        data = np.ones(npts, dtype=np.float32)
        ft = FoldTrace(data=data, header={'sampling': 1.})
        # Test warning on overlog taper
        with pytest.warns(UserWarning):
            ft.taper(max_percentage=0.7)
        # Test that tapering is still applied
        assert all(ft.data[:5] < 1)
        assert ft.data[5] == 1
        assert all(ft.data[6:] < 1)
        # Test max_length
        data = np.ones(npts)
        ft = FoldTrace(data=data, header={'sampling': 1.})
        # first 3 samples get tapered
        ft.taper(max_percentage=None, type=type_, side="left", max_length=3)
        # last 5 samples get tapered
        ft.taper(max_percentage=0.5, type=type_, side="right", max_length=None)
        assert np.all(ft.data[:3] < 1.)
        assert np.all(ft.data[3:6] == 1.)
        assert np.all(ft.data[6:] < 1.)

        assert np.all(ft.fold[:3] < 1.)
        assert np.all(ft.fold[3:6] == 1.)
        assert np.all(ft.fold[6:] < 1.)

        data = np.ones(npts, dtype=np.float32)
        ft = FoldTrace(data=data, header={'sampling': 1.})
        assert ft.stats.processing == []
        # first 3 samples get tapered
        ft.taper(max_percentage=0.5, type=type_, side="left", max_length=3)
        # last 3 samples get tapered
        ft.taper(max_percentage=0.3, type=type_, side="right", max_length=5)
        assert np.all(ft.data[:3] < 1.)
        assert np.all(ft.data[3:8] == 1.)
        assert np.all(ft.data[8:] < 1.)
        assert np.all(ft.fold[:3] < 1.)
        assert np.all(ft.fold[3:8] == 1.)
        assert np.all(ft.fold[8:] < 1.)
        
    ###########################
    ## RESAMPLING TEST SUITE ##
    ###########################
    def test_interp_fold(self):
        """Test suite for the _interp_fold private method of FoldTrace
        used to augment inherited data resampling methods from Trace
        """        
        # Setup
        ft = FoldTrace(data=np.arange(6), fold=np.array([1,1,1,2,2,2]), dtype=np.float32)
        ft2 = ft.copy()
        ft2.data = np.linspace(0,6,11)
        ft2.stats.sampling_rate = 2
        assert ft.stats.starttime == ft2.stats.starttime
        assert ft.stats.endtime == ft2.stats.endtime
        # Check that verify fails on Exception (data and fold shape don't match)
        with pytest.raises(Exception):
            ft2.verify()
        # Apply method
        ft3 = ft2.copy()
        ft3._interp_fold(ft.stats.starttime, 1)
        assert ft3.data.shape == ft3.fold.shape
        # Test with slight non-aligned samples
        ft3 = ft2.copy()
        ft3.stats.starttime -= 0.1
        ft3._interp_fold(ft.stats.starttime, 1)
        assert ft3.verify()
        # Test with large non-aligned samples
        ft3 = ft2.copy()
        ft3.stats.starttime -= 1.37
        ft3.data = np.arange(7, dtype=ft3.dtype)
        ft3._interp_fold(ft.stats.starttime, 1)
        assert ft3.verify()

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
        """
        # Setup
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
        ft2 = ft.copy().resample(sampling_rate=0.1)
        assert ft2.stats.endtime == ft.stats.endtime - 9.0
        assert ft2.stats.starttime == ft.stats.starttime
        assert ft2.stats.sampling_rate == 0.1
        assert ft2.fold.shape == ft2.data.shape

        # upsample by factor of 2 and don't enforce time domain
        ft2 = ft.copy().resample(sampling_rate=2, enforce_time_domain=False)
        # Updated sampling rate
        assert ft2.stats.sampling_rate == 2.
        # Unchanged starttime
        assert ft2.stats.starttime == ft.stats.starttime
        # Changed endtime
        assert ft2.stats.endtime != ft.stats.endtime

        # upsample by factor of 2, enforing time domain
        ft3 = ft.copy().resample(sampling_rate=2)
        assert ft3.stats.starttime == ft.stats.starttime
        # enforced time domain check
        assert ft.stats.endtime >= ft3.stats.endtime
        # changed sampling rate
        assert ft3.stats.sampling_rate == 2.
        # shape of fold and data are consistent
        assert ft3.fold.shape == ft3.data.shape

        # upsample by a non-integer factor
        ft4 = ft.copy().resample(sampling_rate=1.4, enforce_time_domain=True)
        assert ft4.stats.starttime == ft.stats.starttime
        # enforced time domain check
        assert ft.stats.endtime >= ft4.stats.endtime
        # Changed sampling rate
        assert ft4.stats.sampling_rate == 1.4
        # Shape of data and fold consistent
        assert ft4.fold.shape == ft4.data.shape

        # exception raised with gappy data
        ft.data = np.ma.MaskedArray(data=ft.data,
                                     mask=[False]*30)
        ft.data.mask[10:15] = True
        with pytest.raises(Exception):
            ft.resample(3)

    def test_interpolate(self):
        """Test suite for interpolate method of FoldTrace
        """
        tr = load_logo_trace()
        ft = FoldTrace(load_logo_trace())
        ft.fold[15:] = 2
        assert ft.stats.sampling_rate == 1.
        assert ft.stats.npts == 30

        # upsample by an integer factor
        tr2 = tr.copy().interpolate(sampling_rate=2.)
        ft2 = ft.copy().interpolate(sampling_rate=2.)
        assert ft2.stats.endtime == ft.stats.endtime
        assert ft2.stats.starttime == ft.stats.starttime
        assert ft2.stats.sampling_rate == 2.
        assert ft2.data.shape == ft2.fold.shape
        np.testing.assert_array_equal(ft2.data, tr2.data)

        # upsample by a float factor
        tr3 = tr.copy().interpolate(sampling_rate=11.5)
        ft3 = ft.copy().interpolate(sampling_rate=11.5)
        assert ft3.stats.starttime == ft.stats.starttime
        assert ft3.stats.endtime <= ft.stats.endtime
        assert ft3.stats.sampling_rate == 11.5
        assert ft3.data.shape == ft3.fold.shape
        np.testing.assert_array_equal(ft3.data, tr3.data)

        # downsampleby an integer factor
        tr4 = tr.copy().interpolate(sampling_rate=0.5)
        ft4 = ft.copy().interpolate(sampling_rate=0.5)
        assert ft4.stats.starttime == ft.stats.starttime
        assert ft4.stats.endtime <= ft.stats.endtime
        assert ft4.stats.sampling_rate == 0.5
        assert ft4.data.shape == ft4.fold.shape
        np.testing.assert_array_equal(ft4.data, tr4.data)

        # exception raised with gappy data
        ft.data = np.ma.MaskedArray(data=ft.data,
                                     mask=[False]*30)
        ft.data.mask[10:15] = True
        with pytest.raises(Exception):
            ft.interpolate(3)

    def test_decimate(self):
        """Test suite for FoldTrace.decimate
        """
        # Run extended tests
        tr = load_logo_trace()
        ft = FoldTrace(load_logo_trace())
        ft.fold[15:] = 2

        # decimate by an integer factor
        tr2 = tr.copy().decimate(2)
        ft2 = ft.copy().decimate(2)
        assert ft2.stats.starttime == tr2.stats.starttime == tr.stats.starttime
        assert ft2.stats.endtime == tr2.stats.endtime <= tr.stats.endtime
        assert ft2.stats.sampling_rate == 0.5
        assert tr2.stats.sampling_rate == 0.5
        np.testing.assert_array_equal(ft2.data, tr2.data)
        assert ft2.fold.shape == ft2.data.shape

        # positive int-like is acceptable
        ft3 = ft.copy().decimate(2.)
        assert ft3 == ft2
        # non-int-like decimation factor raises rror
        with pytest.raises(TypeError):
            ft.copy().decimate(1.1)
        # non-positive int-like factor raises error
        with pytest.raises(ValueError):
            ft.copy().decimate(-1)


    def test_filter(self):
        """Test suite for FoldTrace.filter
        """
        # Setup
        tr = read()[0]
        ft = FoldTrace(tr.copy())
        # Define filters
        hpf = {'type': 'highpass', 'freq': 10}
        lpf = {'type': 'lowpass', 'freq': 10}
        bpf = {'type': 'bandpass', 'freqmin': 1, 'freqmax': 20}
        # Iterate across filters
        for fkw in [hpf, lpf, bpf]:
            # Apply filtering to copies of Trace and FoldTrace
            trf = tr.copy().filter(**fkw)
            ftf = ft.copy().filter(**fkw)
            # Assert that data match for both filtered
            np.testing.assert_array_equal(trf.data, ftf.data)
            # Assert that fold matches the original (unfiltered) FoldTrace.fold
            np.testing.assert_array_equal(ftf.fold, ft.fold)
            # Assert that processing matches
            assert trf.stats.processing == ftf.stats.processing

    def test_normalize(self):
        """Test suite for normalize method of FoldTrace
        """
        super().test_normalize()
        tr = read()[0]
        ft = FoldTrace(tr.copy())
        # Test string norm inputs
        for _nrm in ['max','minmax','peak','std','standard','sigma']:
            if _nrm in ['max','minmax','peak']:
                tr2 = tr.copy().normalize(norm=None)
            else:
                tr2 = tr.copy().normalize(norm=tr.std())
            ft2 = ft.copy().normalize(norm=_nrm)
            # Assert same result as ObsPy
            np.testing.assert_array_equal(ft2.data, tr2.data)
            # Assert no change to fold
            np.testing.assert_array_equal(ft2.fold, ft.fold)
        # Test int and float norm inputs
        for _nrm in [2, 2.34]:
            tr2 = tr.copy().normalize(norm=_nrm)
            ft2 = ft.copy().normalize(norm=_nrm)
            # Assert same result as ObsPy
            np.testing.assert_array_equal(ft2.data, tr2.data)
            # Assert fold not changed
            np.testing.assert_array_equal(ft2.fold, ft.fold)

    def test_taper_double_processing_bugfix(self):
        # Setup
        tr = read()[0]
        ft = FoldTrace(tr)
        ftc = ft.copy()
        pcount = len(ft.stats.processing)
        assert pcount == 0
        # Apply Method
        ft.taper(0.05)
        # Assert that taper only adds one processing entry
        assert len(ft.stats.processing) == 1
        # Assert that entry is a taper entry
        assert 'taper' in ft.stats.processing[0]
        # Assert that all other stats are the same besides processing
        for _k, _v in ft.stats.items():
            if _k != 'processing':
                assert ftc.stats[_k] == _v

    def test_align_starttime(self):
        # Setup
        ft0 = FoldTrace(data=np.ones(10))
        ts = UTCDateTime(0)
        assert len(ft0.stats.processing) == 0

        # Test small positive adjustment
        ft = ft0.copy()
        ft.align_starttime(ts+0.01, 1)
        assert ft.stats.starttime == ts+0.01
        assert len(ft.stats.processing) == 1
        assert 'align_starttime' in ft.stats.processing[-1]

        # Test small negative adjustment
        ft = ft0.copy()
        ft.align_starttime(ts-0.01, 1)
        assert ft.stats.starttime == ts-0.01

        # Test large, but near-enough positive adjustment
        ft = ft0.copy()
        ft.align_starttime(ts+1.01, 1)
        assert ft.stats.starttime == ts + 0.01
        # Test large, but near-enough negative adjustment
        ft = ft0.copy()
        ft.align_starttime(ts-1.01, 1)
        assert ft.stats.starttime == ts - 0.01

        # Test small, but too far positive adjustment
        ft = ft0.copy()
        ft.align_starttime(ts + 0.04, 1)
        assert ft.stats.starttime == ts +0.04
        assert 'interpolate' in ft.stats.processing[-1]
        # Test small, but too far negative adjustment
        ft = ft0.copy()
        ft.align_starttime(ts-0.04, 1)
        assert ft.stats.starttime == ts + 1 - 0.04

        # Test large, too far, positive adjustment
        ft = ft0.copy()
        ft.align_starttime(ts + 1.04, 1)
        assert ft.stats.starttime == ts + 0.04

        # Test large, too far, negative adjustment
        ft = ft0.copy()
        ft.align_starttime(ts - 1.04, 1)
        assert ft.stats.starttime == ts + 1 - 0.04

        # Test different sampling rate that accurately aligns to current
        ft = ft0.copy()
        ft.align_starttime(ts + 0.01, 100)
        assert ft.stats.starttime == ts
        # Test different sampling rate that does not align
        ft = ft0.copy()
        ft.align_starttime(ts + 0.01, 10)
        assert ft.stats.starttime == ts + 0.01
