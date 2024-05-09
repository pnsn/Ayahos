import math, pickle, warnings, pytest
# Import obspy.core.trace.Trace test suite class for inheritance
from obspy.core.tests.test_trace import TestTrace
# Import wyrm.core.mltrace.MLTrace as Trace to supercede Trace calls in TestTrace
from wyrm.core.trace.mltrace import MLTrace as Trace

class MLTrace_Test(TestTrace):

    def test_init(self):
        """
        Test the __init__ method of the MLTrace class
        """
        # Run inherited tests
        super().test_init()
        # 