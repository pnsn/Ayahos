from wyrm.structures.rtbufftrace import RtBuffTrace
import pytest
from obspy import Trace, Stream
import numpy as np


class test_rtbufftrace:
    """
    Testing suite for the wyrm.structures.rtbufftrace.RtBuffTrace class
    """

    def generate_trace_examples(self):
        # Create trace with random data
        self.trace_abc = Trace(
            data=np.random.rand(3000, 1),
            header={
                "station": "ABC",
                "network": "UW",
                "channel": "BHZ",
                "sampling_rate": 50.0,
                "location.": "--",
            },
        )
        # Create copy of trace with gappy data
        self.trace_abc_gappy = self.trace_abc.copy()
        self.trace_abc_gappy.data = np.ma.masked_array(
            data=self.trace_abc_gappy.data,
            mask=[False] * 1000 + [True] * 500 + [False] * 1500,
            fill_value=None,
        )
        # Create next contiguous trace
        self.trace_abc_next = self.trace_abc.copy()
        self.trace_abc_next.stats.starttime = (
            self.trace_abc.stats.endtime + 1 / self.trace_abc.stats.sampling_rate
        )
        # Create next contiguous trace, but from horizontal channel code
        self.trace_abc_hztl_next = self.trace_abc_next.copy()
        self.trace_abc_hztl_next.stats.channel = self.trace_abc.stats.channel[:-1] + "N"

        # Create next contiguous trace with gaps
        self.trace_abc_next_gappy = self.trace_abc_gappy.copy()
        self.trace_abc_next_gappy.stats.starttime = self.trace_abc.stats.endtime

        # Create next data after a large-ish outage
        self.trace_abc_post_outage = self.trace_abc.copy()
        self.trace_abc_post_outage.stats.starttime += self._test_max_length * 1.01

        # TODO: Create overlapping sequential trace

        return self

    def __init__(self):
        self.generate_trace_examples()
        self._test_max_length = 90.0

    def test_init(self):
        # Test defaults with blank initialization
        rtbt = RtBuffTrace()
        assert rtbt.stats.sampling_rate == 1
        assert rtbt.stats.filled_fraction == 0
        assert rtbt.stats.valid_fraction == 1
        assert isinstance(RtBuffTrace, Trace)
        assert rtbt.fill_value is None
        assert rtbt.max_length == 1
        assert rtbt.method == 1
        assert rtbt.interpolation_samples == -1
        assert rtbt._have_appended_trace == False

        # Test max_length valid inputs
        for _i in [1e-9, 1, 1.0, 50]:
            assert RtBuffTrace(max_length=_i).max_length == _i
        # Test max_length ValueError exceptions
        for _i in [0, np.inf]:
            with pytest.raises(ValueError):
                RtBuffTrace(max_length=_i)
        # Test max_length TypeError exceptions
        for _i in ["a", int, False]:
            with pytest.raises(TypeError):
                RtBuffTrace(max_length=_i)

        # Test fill_value valid inputs
        for _i in [1.0, 0, -13, -0.1, None, np.inf]:
            assert RtBuffTrace(fill_value=_i).fill_value == _i
        # Test fill_value TypeError exceptions
        for _i in ["a", int, False]:
            with pytest.raises(TypeError):
                RtBuffTrace(fill_value=_i)

        # Test method valid inputs
        for _i in [-1, 0, 1]:
            assert RtBuffTrace(method=_i).method == _i
        # Test ValueError exceptions
        for _i in [-1.0, 0.0, 1.0, False, "a", np.inf]:
            with pytest.raises(ValueError):
                RtBuffTrace(method=_i)

        # Test interpolation_samples valid inputs
        for _i in [-1, 0, 1e6]:
            assert RtBuffTrace(interpolation_samples=_i).interpolation_samples == _i
        for _i in [-np.inf, -2, np.inf]:
            with pytest.raises(ValueError):
                RtBuffTrace(interpolation_samples=_i)
        for _i in [True, "a", int]:
            with pytest.raises(TypeError):
                RtBuffTrace(interpolation_samples=_i)

    def test__eq__(self):
        assert not RtBuffTrace().__eq__(Trace())
        assert RtBuffTrace().__eq__(RtBuffTrace())

    def test_copy(self):
        rtbt = RtBuffTrace()
        assert rtbt == rtbt.copy()

    def test_to_trace(self):
        rtbt = RtBuffTrace()
        trace = Trace()
        assert isinstance(rtbt.to_trace, Trace)
        assert rtbt.to_trace() == trace

    def test__first_append(self):
        rtbt = RtBuffTrace(max_length=self._test_max_length)
        assert not rtbt._have_appended_trace
        ff0 = rtbt.filled_fraction
        vf0 = rtbt.valid_fraction
        # Test _first_append()
        with pytest.raises(TypeError):
            RtBuffTrace(max_length=self._test_max_length)._first_append(3.0)
        rtbt._first_append(self.trace_abc)
        assert rtbt.data == self.trace_abc.data
        assert rtbt.stats == self.trace_abc.stats
        assert rtbt._have_appended_trace
        assert rtbt.filled_fraction != ff0
        assert (
            rtbt.filled_fraction
            == (self.trace_abc.stats.endtime - self.trace_abc.stats.starttime)
            / rtbt.max_length
        )
        assert rtbt.valid_fraction == vf0

    def test__check_attribute_compatability(self):
        rtbt = RtBuffTrace(max_length=self._test_max_length)._first_append(
            self.trace_abc
        )
        # Test _check_attribute_compatability
        assert rtbt._check_attribute_compatability(self.trace_abc_next, attr_str="id")
        # Test ValueError exception examples
        for _i in ["id", "stats.starttime", "stats.endtime", "stats.channel"]:
            with pytest.raises(ValueError):
                rtbt._check_attribute_compatability(
                    self.trace_abc_hztl_next, attr_str=_i
                )

    def test_check_trace_compatability(self):
        rtbt = RtBuffTrace(max_length=self._test_max_length)._first_append(
            self.trace_abc
        )
        # Test check_trace_compatability
        assert rtbt.check_trace_compatability(self.trace_abc_next)
        assert not rtbt.check_trace_compatability(self.trace_abc_hztl_next)

    def test__contiguous_append(self):
        rtbt = RtBuffTrace(max_length=self._test_max_length)._first_append(
            self.trace_abc
        )
        # Test _contiguous_append() on sequential traces
        with pytest.raises(TypeError):
            rtbt._contiguous_append(4.0)
        lt_merge_abc = Stream([self.trace_abc, self.trace_abc_next])
        lt_merge_abc.merge(method=1, interpolation_samples=-1, fill_value=None)
        lt_merge_abc._ltrim(self.trace_abc_next.stats.endtime - self._test_max_length)
        assert rtbt._contiguous_append(self.trace_abc_next).data == lt_merge_abc.data
        assert rtbt._contiguous_append(self.trace_abc_next).stats == lt_merge_abc.stats
        assert rtbt.filled_fraction == 1
        assert rtbt.valid_fraction == 1
        # Test multiple streams post merge exception
        rtbt = RtBuffTrace(max_length=self._test_max_length)._first_append(
            self.trace_abc
        )
        assert rtbt == rtbt.copy()._contiguous_append(self.trace_abc_hztl_next)
        # TODO: Test append of overlapping traces

    def test__gappy_append(self):
        rtbt = RtBuffTrace(max_length=self._test_max_length)._first_append(
            self.trace_abc
        )
        # Test _contiguous_append()
        with pytest.raises(TypeError):
            rtbt._gappy_append(4.0)

        # TODO: Test append of contiguous, complete traces
            
        # TODO: Test append of contiguous, gappy traces
            
        # TODO: Test append of nonsequential packets


    def test__assess_relatvive_timing(self):
        rtbt = RtBuffTrace(max_length=self._test_max_length)._first_append(
            self.trace_abc
        )
        # Test TypeError exception
        with pytest.raises(TypeError):
            rtbt._assess_relative_timing(4.0)
        # Test case where trace trails and overlaps rtbt
        status, gap = rtbt._assess_relatvive_timing(self.trace_abc_overlapping)
        assert status == 'overlapping'
        assert gap is None
        # Test case where trace traisl and does not overlap rtbt
        status, gap = rtbt._assess_relatvive_timing(self.trace_abc_post_outage)
        assert status == 'trailing'
        assert gap is not None
        assert gap >= 0
        assert gap == self._test_max_length*1.1
        # Test case where trace is leads and overlaps rtbt
        rtbt = RtBuffTrace(max_length=self._test_max_length)._first_append(
            self.trace_abc_overlapping
        )
        status, gap = rtbt._assess_relative_timing(self.trace_abc)
        assert status == 'overlapping'
        assert gap is None
        # Test case where trace lags and does not overlap rtbt
        rtbt = RtBuffTrace(max_length=self._test_max_length)._first_append(
            self.trace_abc_post_outage
        )
        status, gap = rtbt._assess_relative_timing(self.trace_abc)
        assert status == 'leading'
        assert gap is not None
        assert gap >= 0
        assert gap == self._test_max_length*1.1

    # TODO
    def test_get_unmasked_fraction(self):

    # TODO
    def test_get_filled_fraction(self):

    # TODO 
    def test_append(self):

