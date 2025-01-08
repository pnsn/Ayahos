"""
:module: PULSE.mod.operating
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:
"""
from collections import deque

from obspy import UTCDateTime
from obspy.clients.seedlink.easyseedlink import create_client

from PULSE.mod.base import BaseMod
from PULSE.mod.sequencing import SeqMod

class RunMod(SeqMod):

    def __init__(
            self,
            metadata_memory_length=60.,
            report_interval=5.,
            sequence=[BaseMod(maxlen=1024)],
            mincache=1,
            maxcache=1024,
            name=None):

        super().__init__(
            modules=sequence,
            maxlen=metadata_memory_length,
            max_pulse_size=1,
            name=name)
        
        self.input = deque(maxlen=maxcache)
        self.report_interval = report_interval

    
    def __setattr__(self, key, value):
        if key == 'report_interval':
            if not isinstance(value, (int, float)):
                raise TypeError
            else:
                value = float(value)
            if value <= 0:
                raise ValueError


    def run(self):
        pulse_no = 0
        t0 = UTCDateTime()
        while pulse_no < self.max_pulses:
            self.get_input()
            if self.measure_input(input) > self.mincache:
                self.pulse(input)
                pulse_no += self.increment
                ti = UTCDateTime()
                if ti - t0 > self.report_interval:
                    self.report()
                    t0 = ti

    def get_input(self):
        for _e in range(10):
            self.input.appendleft(_e)

    def report(self):
        print(self.sequence.current_stats)


class RunEasySEEDLinkMod(RunMod):
    
    def __init__(
            self,
            seedlink_url='rtserve.iris.washington.edu',
            streams=[('CC','REM','BH?'),
                     ('CC','SEP','BH?'),
                     ('UW','SHW','HH?'),
                     ('UW','HSR','HH?')],
            sequence=[BaseMod(maxlen=1024)],
            mincache=3,
            maxcache=1024,
            max_pulses=None,
            maxlen=60):
        # Inherit from RunMod
        super().__init__(maxlen=maxlen, sequence=sequence, mincache=mincache,
                         max_pulses=max_pulses)
        self.input = deque(maxlen=maxcache)
        # Create SEEDLink Client Object
        self.client = create_client(seedlink_url, on_data=self.get_input)
        # 
        for _n, _s, _c in streams:
            self.client.select_stream(_n, _s, _c)

    def get_input(self, trace):
        # Accumulate traces
        self.input.appendleft(trace)
        # If input length exceeds minimum cache size
        if self.measure_input(self.input) >= self.mincache:
            # Trigger pulse
            self.pulse(self.input)
            # Get new timestamp
            ti = UTCDateTime()
            if ti - self.t0 > self.report_interval:
                self.report()



    def run(self):
        self.t0 = UTCDateTime()
        self.client.run()


def RunWaveformClientMod(RunMod):

    def __init__(
            self,
            requests,
            base_url='IRIS',
            client_init_kwargs={},
            window_length=60.,
            sequence=[BaseMod(maxlen=1024)],
            mincache=1,
            maxcache=1024):
        
        super().__init__(
            mincache=mincache,
            maxcache=maxcache,
            sequence=sequence,
            max_pulses=None)

