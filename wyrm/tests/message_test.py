import pytest
from wyrm.core.message import *
from obspy.core.tests.test_trace import TestTrace

class Test__BaseMsg:
    def test_init(self):
        # Test default settings
        assert _BaseMsg() == _BaseMsg(mtype="TYPE_TRACEBUF2", mcode=19)

        # Test compatability check for message-type (mtype)
        with pytest.raises(TypeError):
            for x in [1, -1.0, True, None]:
                _BaseMsg(mtype=x)
        with pytest.raises(SyntaxError):
            _BaseMsg(mtype="TRACEBUF2")

        # Test compatability check for message-code (mcode)
        # TypeError tests
        with pytest.raises(TypeError):
            for x in ["a", -1.0, True, None]:
                _BaseMsg(mcode=x)
        # value out of range SyntaxError
        with pytest.raises(SyntaxError):
            for x in [-1, 256, 1000, np.inf, -np.inf, np.nan]:
                _BaseMsg(mcode=x)
        # Values not in earthworm_global.d
        for _i in range(256):
            # Raise SyntaxError for values 0-99
            if _i not in EW_GLOBAL_DICT.values():
                if _i < 100:
                    with pytest.raises(SyntaxError):
                        _BaseMsg(mcode=_i)
                # Raise Warning for values 100-255
                elif _i >= 100:
                    with pytest.raises(Warning):
                        _BaseMsg(mcode=_i)

        # Test matches for all type:code pairs from earthworm_global.d
        for _i, _k in enumerate(EW_GLOBAL_DICT.keys()):
            for _j, _v in enumerate(EW_GLOBAL_DICT.values()):
                # If coming from the same indices,
                # assert match yields valid _BaseMsg
                if _i == _j:
                    assert isinstance(_BaseMsg(mtype=_k, mcode=_v), _BaseMsg)
                    # Do SyntaxError test on lowercase mtype inputs
                    with pytest.raises(SyntaxError):
                        _BaseMsg(mtype=_k.lower(), mcode=_v)
                # Otherwise ensure that _BaseMsg returns SyntaxError
                else:
                    with pytest.raises(SyntaxError):
                        _BaseMsg(mtype=_k, mcode=_v)

    def test_repr(self):
        msg = _BaseMsg()
        assert msg.__repr__() == "MTYPE: TYPE_TRACEBUF2\nMCODE: 19\n"


class Test_TraceMsg:
    # Test Trace
    with pytest.test(TestTrace)
    