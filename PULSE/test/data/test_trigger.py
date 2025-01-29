from unittest import TestCase

import numpy as np
from obspy import UTCDateTime
from obspy.signal.trigger import trigger_onset

from PULSE.data.foldtrace import FoldTrace
from PULSE.data.trigger import Trigger, scaled_gaussian

class TestTrigger(TestCase):
    # Pre-setup
    xv = np.arange(3001)
    gaussian = scaled_gaussian([0.9, 300, 20**2], xv)
    noise = np.abs(np.random.normal(loc=0., scale=0.05, size=3001))
    pred = gaussian + noise
    header = {'network':'UW','station':'GPW','channel':'EHP',
              'model':'PhaseNet','weight':'test','sampling_rate':100.,
              'starttime': UTCDateTime('1991-02-21T15:35:00')}
    thres = 0.3

    def setUp(self):
        self.ft = FoldTrace(data=self.pred.copy(), header=self.header.copy())
        triggers = trigger_onset(self.ft.data, self.thres, self.thres)
        self.triggers = []
        for trigger in triggers:
            if trigger[1] - trigger[0] > 5:
                self.triggers.append(trigger)


    def tearDown(self):
        del self.ft
        del self.triggers

    # def test_setup(self):
    #     self.ft.plot()
    #     breakpoint()
        
    def test_init(self):
        trigger = Trigger(
            source_trace=self.ft,
            samp_on=self.triggers[0][0],
            samp_off=self.triggers[0][1],
            thr_on=self.thres,
            thr_off=self.thres)
        
        self.assertIsInstance(trigger, FoldTrace)
        self.assertIsInstance(trigger, Trigger)
        # Assert that data views match
        np.testing.assert_array_equal(
            trigger.data,
            self.ft.view(starttime=trigger.stats.starttime,
                         endtime=trigger.stats.endtime).data)
        self.assertEqual(trigger.samp_pad, 0)
        self.assertEqual(trigger.samp_on, 0)
        self.assertEqual(trigger.samp_off, trigger.count() - 1)
        self.assertEqual(trigger.samp_off, self.triggers[0][1] - self.triggers[0][0])



    def test_init_padded(self):
        trigger = Trigger(
            source_trace=self.ft,
            samp_on=self.triggers[0][0],
            samp_off=self.triggers[0][1],
            thr_on=self.thres,
            thr_off=self.thres,
            samp_pad=10
        )
        self.assertIsInstance(trigger, Trigger)
        self.assertEqual(trigger.samp_pad, 10)
        # Assert pading samples are present
        self.assertEqual(trigger.count(), 10*2 + self.triggers[0][1] - self.triggers[0][0] + 1)
        breakpoint()