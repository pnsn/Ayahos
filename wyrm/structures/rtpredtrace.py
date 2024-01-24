from obspy import Trace
from obspy.core.compatibility import round_away
import wyrm.util.input_compatability_checks as icc

class RtPredStack(Trace):

    def __init__(self, max_length=3e4, window=6000, blinding=(500,500), overlap=1800):
        """
        Initialize a Realtime Prediction Stack object

        :: INPUTS ::
        :param max_length: [int] maximum buffer length in samples
        :param blinding: [tuple of int] number of samples to blind
                        on either end of an input prediction
        :param overlap: [int] number of samples each successive 
                        prediction window overlaps by
        :param window: [int] number of samples in a prediction window
        """
        # max_length compatability checks
        self.max_length = icc.bounded_intlike(
            max_length,
            name='max_length',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # window compatability checks
        self.window = icc.bounded_intlike(
            window,
            name='window',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # blinding compatability checks
        if isinstance(blinding, tuple):
            blinding_l = icc.bounded_intlike(
                blinding[0],
                name='blinding[0]',
                minimum=0,
                maximum=self.window/2,
                inclusive=True
            )
            blinding_r = icc.bounded_intlike(
                blinding[-1],
                name='blinding[-1]',
                minimum=0,
                maximum=self.window/2,
                inclusive=True
            )
            self.blinding = (blinding_l, blinding_r)
        else:
            val = icc.bounded_intlike(
                blinding,
                name='blinding',
                minimum=0,
                maximum=self.window/2,
                inclusive=True
            )
            self.blinding = (val, val)
        # overlap compatability checks
        self.overlap = icc.bounded_intlike(
            overlap,
            name='overlap',
            minimum=0,
            maximum=self.window-1,
            inclusive=True
        )

        # initialize trace inheritance, starting with empty trace
        super().__init__(data=[], header={})
        # initialize completeness vector
        self.completeness_vector = np.ma.masked_array(
            data=np.zeros(self.max_length),
            mask=np.zeros(self.max_length))
        
        self._last_window_number = None

        self._calc_stacking_weights()

    def _calc_stacking_weights(self):
        raw_cts = np.zeros(self.window)
        blind_vect = np.ones(self.window)
        blind_vect[:self.blind[0]] = 0
        blind_vect[-self.blind[1]:] = 0

        i_ = 0
        while i_ < self.window:
            raw_cts[i_:] += blind_vect[:-i_]
            i_ += (self.window - self.overlap)
        
        norm_wgts = raw_cts**-1


    def __add__(self, trace, method='max')
        
        status = self._run_trace_crosschecks(trace)
        if isinstance(status, str):
            raise TypeError(status)
        
        if self.stats.starttime <= trace.stats.starttime:
            lt = self
            rt = trace
        else:
            lt = trace
            rt = self
        

        
        # Check for gap
        sr = self.stats.sampling_rate
        delta = (rt.stats.starttime - lt.stats.endtime)*sr
        delta = int(round_away(delta)) - 1
        delta_endtime= lt.stats.endtime - rt.stats.endtime
        out = self.__class__(header=deepcopy(lt.stats))

        if delta < 0 and delta_endtime < 0:
            # overlap
            delta = abs(delta)
            if np.all(np.equal(lt.data[-delta:], rt.data[:delta]))):
                

        if delta < 0:
            overlap = True
        else:
            overlap = False
        