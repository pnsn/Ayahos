import pytest
from wyrm.structures.rtbufftrace import RtBuffTrace
from obspy import Trace, read, Stream
import numpy as np

# Create test set of traces
# NOTE: Get a better event/station combo for demo...
test_st = read('../../../example/uw61957912/UW.TOUCH.._EN_HH_ZNE.mseed')
st_EN_ZNE = test_st.select(channel='EN?')
st_HH_ZNE = test_st.select(channel='HH?')
st_seg = Stream()

for _i in np.arange(0,)

class Test_RtBuffTrace:
    """
    Test suite for wyrm.structures.RtBuffTrace
    """

    def __init__(self):
        self.type_list = [True, 1, -1, 0, 1.0, "a", RtBuffTrace(), str, None]

    def test_init(self):
        # test empty init
        rtbt = RtBuffTrace()
        # Assert inheritance
        assert type(rtbt) is Trace
        # Assert self type
        assert type(rtbt) is RtBuffTrace
        # Assert attributes
        assert np.ma.is_masked(rtbt.data)
        assert rtbt.dtype is None
        assert rtbt.have_appended_trace == False
        assert rtbt.stats == Trace().stats

        # test assigned max_length init
        for val in self.type_list:
            if val in [1, 1.0]:
                rtbt = RtBuffTrace(max_length=val)
                assert type(rtbt) is RtBuffTrace
                assert rtbt.max_length == val
            elif val == -1:
                with pytest.raises(ValueError):
                    RtBuffTrace(max_length=val)
            else:
                with pytest.raises(TypeError):
                    RtBuffTrace(max_length=val)
        # test assigned fill_value init
        for val in self.type_list:
            if val not in [True, "a", RtBuffTrace(), str]:
                rtbt = RtBuffTrace(fill_value=val)
                assert type(rtbt) is RtBuffTrace
                assert rtbt.fill_value == val
                assert rtbt.data.fill_value == val
            else:
                with pytest.raises(TypeError):
                    RtBuffTrace(val)

    def test_copy(self):
        rtbt = RtBuffTrace(max_length=10)
        assert rtbt.copy() == rtbt


    def test_append(self):
        rtbt = RtBuffTrace(max_length=10)
        # Test compatability checks
        for val in self.type_list:
            if val is not RtBuffTrace():
                with pytest.raises(TypeError):
                    rtbt.append(val)
            else:
                # assert that even appending an empty trace preserves max_length
                assert rtbt.append(val) == RtBuffTrace(max_length=10)
        assert rtbt.append(ST[0])
        with pytest.raises(TypeError)


    def test_display_buff_status(self):
        rtbt = 

    def test_str_(self):