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
from wyrm.core.message import PEWMsg


class Test_PEWMsg:
    """
    Test suite for wyrm.classes.pyew_msg.PEWMsg
    """
    def test_init(self):
        """
        Thest the __init__ method of the PyEW base class

        :: TODO ::
        Include max character limits on __init__ and in tests
        """
        # Test None argument initialization
        msg = PEWMsg()
        assert msg.code == '...--'
        assert msg.station == ''
        assert msg.network == ''
        assert msg.channel == ''
        assert msg.location == '--'

        # Test station approved input types
        for _x in ['abc', 1000]:
            # Initialize
            msg = PEWMsg(station=_x)
            # Check attribute Types
            assert isinstance(msg.station, str)
            assert isinstance(msg.code, str)
            # Check for expected attribute contents
            assert msg.station == str(_x)  
            assert msg.code == f'{str(_x)}...--'      
        # Test station unapproved input type error reporting
        for _x in [100., False, int]:
            with pytest.raises(TypeError):
                PEWMsg(station=_x)
        
        # Test network approved input types
        for _x in ['abc', 1000]:
            # Generate message
            msg = PEWMsg(network=_x)
            # Check attribute Types
            assert isinstance(msg.network, str)
            assert isinstance(msg.code, str)
            # Check for expected attribute contents
            assert msg.network == str(_x)
            assert msg.code == f'.{str(_x)}..--'   
        # Test network unapproved input type error reporting
        for _x in [100., False, int]:
            with pytest.raises(TypeError):
                PEWMsg(station=_x)
        
        # Test channel approved input types
        for _x in ['abc', 100]:
            msg = PEWMsg(channel=_x)
            assert isinstance(msg.channel, str)
            assert isinstance(msg.code, str)
            assert msg.channel == str(_x)
            assert msg.code == f'..{str(_x)}.--'
        # Test channel unapproved types
        for _x in [100., False, int]:
            with pytest.raises(TypeError):
                PEWMsg(channel=_x)
        
        # Test location approved types
        for _x in ['abc', 100]:
            # Initialize message object
            msg = PEWMsg(location=_x)
            # Check expected attribute types
            assert isinstance(msg.code, str)
            assert isinstance(msg.location, str)
            # Check expected attribute contents
            assert msg.code == f'...{str(_x)}'
            assert msg.location == str(_x)
        # Test location behaviors for arbitrary whitespace input
        for _x in ['', ' ', '        ']:
            msg = PEWMsg(location=_x)
            assert isinstance(msg.location, str)
            assert isinstance(msg.code, str)
            assert msg.location == '--'
            assert msg.code == '...--'
        # Test location unapproved types
        for _x in [100., False, int]:
            with pytest.raises(TypeError):
                PEWMsg(channel=_x)
        
    def test_dtype_conversions(self):
        msg = PEWMsg()
        # Assert approved argument inputs
        for _x, _y in [('i2', np.int16),('i4', np.int32), ('i8', np.int64), ('s4', np.float32)]:
            assert msg.ew2np_dtype(_x) == _y
            assert msg.np2ew_dtype(_y) == _x
        # Check KeyError assertions on ew2np_dtype
        with pytest.raises(ValueError):
            for _x in ['d','f', 1, True]:
                msg.ew2np_dtype(_x)
        # Check TypeError assertions on ew2np_dtype

            

class Test_WaveMsg:
    """
    Test suite for wyrm.classes.pyew_msg.WaveMsg
    """
    def test_init(self)
        msg = PyEW_WaveMsg()
        assert len(msg) == 0:
            