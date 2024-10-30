import unittest, pytest

import numpy as np
from obspy import UTCDateTime, Trace

from PULSE.data.foldtrace import FoldTrace
from PULSE.data.ftbuffer import FTBuffer

class TestFTBuffer(unittest.TestCase):

    def setUp(self):
        # Create 2 minute traces with 1 minute offsets
        self.ft = {'early': FoldTrace(data=np.random.rand(121),
                                          header={'starttime': UTCDateTime(-30)}),
                       'middle': FoldTrace(data=np.random.rand(121)),
                       'late': FoldTrace(data=np.random.rand(121),
                                         header={'starttime': UTCDateTime(30)})}
        self.test_buff = FTBuffer()

    def tearDown(self):
        del self.ft

    def test_init(self):
        # Assert inheritance chain
        self.assertIsInstance(self.test_buff, Trace)
        self.assertIsInstance(self.test_buff, FoldTrace)
        self.assertIsInstance(self.test_buff, FTBuffer)
        # Assert default values
        self.assertEqual(self.test_buff.maxlen, 60.)
        self.assertTrue(self.test_buff._empty)
        self.assertEqual(self.test_buff.method, 3)
        self.assertIsNone(self.test_buff.fill_value)
        self.assertEqual(self.test_buff.dtype, np.float64)

    def test_init_maxlen(self):
        # Test very small
        self.assertIsInstance(FTBuffer(maxlen=1e-10), FTBuffer)
        # Test at limit
        self.assertIsInstance(FTBuffer(maxlen=1200.), FTBuffer)
        # Raises ValueError
        for val in [1200. + 1e-10, 0, -1e-10]:
            with self.assertRaises(ValueError):
                FTBuffer(maxlen=val)
        # Raises TypeError
        for val in [True, 'a', int]:
            with self.assertRaises(TypeError):
                FTBuffer(maxlen=val)

    def test_init_method(self):
        # Approved values
        for val in [0,2,3]:
            self.assertIsInstance(FTBuffer(method=val), FTBuffer)
        # Raises ValueError for unapproved values
        for val in [1, True, int, 'a']:
            with self.assertRaises(ValueError):
                FTBuffer(method=val)
    
    def test_first_append(self):
        # Test 1 - assert that calling first append and append
        # with empty trace result in the same
        # Conduct appends on copies (non-destructive appends)
        buff1 = self.test_buff.copy()
        buff1.append(self.ft['early'].copy())
        buff2 = self.test_buff.copy()
        buff2._first_append(self.ft['early'].copy())
        self.assertEqual(buff1, buff2)

        # Test 2 - Check expected timing and data
        self.assertEqual(buff1.stats.endtime - buff1.stats.starttime,
                         buff1.maxlen)
        self.assertEqual(buff1.stats.endtime, self.ft['early'].stats.endtime)
        self.assertEqual(buff1.stats.starttime, 
                         self.ft['early'].stats.endtime-buff1.maxlen)
        ftr1 = self.ft['early'].copy().trim(starttime=buff1.stats.starttime)
        np.testing.assert_array_equal(buff1.data, ftr1.data)
        np.testing.assert_array_equal(buff1.fold, ftr1.fold)
        # Make sure empty is False
        self.assertFalse(buff1._empty)
    
    def test_internal_add_processing_info(self):
        # Conduct a destructive append
        self.test_buff.append(self.ft['early'])
        # Show that processing logging is turned off for FTBuffer
        self.assertEqual(len(self.ft['early'].stats.processing), 1)
        self.assertEqual(len(self.test_buff.stats.processing), 1)
        self.assertTrue('trim' in self.ft['early'].stats.processing[0])
        self.assertFalse('trim' in self.test_buff.stats.processing[0])
        self.assertTrue('_first_append' in self.test_buff.stats.processing[0])

    def test_shift(self):
        self.test_buff.append(self.ft['early'])
        # shift scope of buffer 5 seconds
        self.test_buff.shift(endtime=UTCDateTime(95))
        # Assert that shifted data have correct endtime
        self.assertEqual(self.test_buff.stats.endtime, UTCDateTime(95))
        # Asesrt that shifted data are masked
        self.assertTrue(np.ma.is_masked(self.test_buff.data))
        # Test safeguards
        for val in [UTCDateTime(30), self.test_buff.stats.endtime -  1]:
            with self.assertRaises(ValueError):
                self.test_buff.shift(endtime=val)
        for val in [True, int, 'a', 2]:
            with self.assertRaises(TypeError):
                self.test_buff.shift(endtime=val)
        for val in [None]:
            self.assertIsInstance(self.test_buff.shift(val), FTBuffer)

    def test_subsequent_append(self):
        # Test error with _empty
        # with self.assertRaises(ValueError):
        #     self.test_buff._subsequent_append(self.ft['early'].copy())
        self.test_buff.append(self.ft['early'].copy())
        self.test_buff._subsequent_append(self.ft['middle'].copy())
        self.assertEqual(self.test_buff.count(), 61)
        self.assertEqual(self.test_buff.stats.endtime, 
                         self.ft['middle'].stats.endtime)
        # Assert non-overlapping samples match source
        np.testing.assert_array_equal(self.test_buff.data[31:],
                                      self.ft['middle'].data[-30:])
        # Fold remains at 1 outside of merged samples
        np.testing.assert_array_equal(self.test_buff.fold[31:],
                                      np.ones(30))
        # Fold goes to 2 inside merged samples
        np.testing.assert_array_equal(self.test_buff.fold[:31],
                                      np.ones(31)*2)
    def test_

        # self.assertIsInstance(self.test_buff, FTBuffer)
        # breakpoint()
        # self.assertEqual(self.test_buff.stats.endtime,
        #                  self.ft['early'].stats.endtime)
        # breakpoint()
        # self.assertEqual(self.test_buff.stats.starttime,
        #                  self.ft['early'].stats.endtime - self.test_buff.maxlen)