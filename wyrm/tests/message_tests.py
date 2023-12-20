"""
:module: wyrm.classes.tests.test_pyew_msg
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:attribution:
    This testing suite is based on the structure of the
    obspy.realtime.rttrace test suite.
"""

import pytest
from wyrm.core.message import _BaseMsg, _SNCLMsg, WaveMsg, EW_GLOBAL_DICT
import numpy as np


class Test__BaseMsg:
    """
    Test suite for wyrm.core.message._BaseMsg
    """

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


class Test__SNCLMsg:
    """
    Test suite for wyrm.core.message._SNCLMsg
    """

    def test_init(self):
        msg = _SNCLMsg()
        # Test None-args version
        assert msg == _SNCLMsg(
            station=None,
            network=None,
            channel=None,
            location=None,
            mtype="TYPE_TRACEBUF2",
            mcode=19,
        )
        assert msg.station == ""
        assert msg.location == "--"
        assert msg.network == ""
        assert msg.channel == ""
        assert msg.sncl == "...--"

        # Test Station behavior
        test_sta_codes = [
            "",
            "G",
            "GN",
            "GNW",
            "GNW2",
            "GNW20",
            1,
            12,
            123,
            1234,
            12345,
        ]
        for _x in test_sta_codes:
            if len(_x) > 4:
                _y = _x[:4]
            else:
                _y = _x
            assert _SNCLMsg(station=_x).station == _y
            assert _SNCLMsg(station=_x).sncl == f"{_y}...--"

        # Test Network behavior
        for _x in ["", "U", "UW", "UWM", 1, 12, 123]:
            if len(_x) > 2:
                _y = _x[:2]
            else:
                _y = _x
            assert _SNCLMsg(network=_x).network == _y
            assert _SNCLMsg(network=_x).sncl == f".{_y}..--"

        # Test Channel behavior
        for _x in ["", "B", "BH", "BHN", "BHN0", 1, 12, 123, 1234]:
            if len(_x) > 3:
                _y = _x[:3]
            else:
                _y = _x
            assert _SNCLMsg(channel=_x).channel == _y
            assert _SNCLMsg(channel=_x).sncl == f"..{_y}.--"

        # Test Location behavior
        _y = "01"
        for _x in ["1", 1, "01", "001", 1001, 1101.1]:
            assert _SNCLMsg(location=_x).location == _y
            assert _SNCLMsg(location=_x).sncl == f"...{_y}"

        for _x in ["", " ", "  ", "   "]:
            assert _SNCLMsg(station=_x).location == "--"
            assert _SNCLMsg(station=_x).sncl == "...--"

    def test_repr():
        rstr1 = _SNCLMsg().__repr__()
        rstr2 = _SNCLMsg(
            station="GNW", network="UW", channel="BHN", location=""
        ).__repr__()
        assert rstr1 == "MTYPE: TYPE_TRACEBUF2\nMCODE: 19\n...--"
        assert rstr2 == "MTYPE: TYPE_TRACEBUF2\nMCODE: 19\nGNW.UW.BHN.--"


class Test_WaveMsg:
    def test_init():
        # Test None input
        msg0 = WaveMsg()
        msg1 = WaveMsg(input=None)
        assert msg1 == msg0
        # Test wave input
        wave0 = {
            "station": 'GNW',
            "network": 'UW',
            "channel": 'BHN',
            "location": '--',
            "nsamp": 100,
            "samprate": 100,
            "startt": 0.0,
            "endt": 1.0,
            "datatype": 's4',
            "data": np.random.rand(100).astype(np.float32),
        }
        msg2 = WaveMsg(wave0)
        assert msg2.station == wave0['station']
        assert msg2.network == wave0['network']
        assert msg2.channel == wave0['channel']
        assert msg2.location == wave0['location']
        assert msg2.startt == wave0['startt']
        assert msg2.endt == wave0['endt']
        assert msg2.nsamp == wave0['nsamp']
        assert msg2.samprate == wave0['samprate']
        assert msg2.datatype == wave0['datatype']
        assert msg2.data == wave0['data']
        