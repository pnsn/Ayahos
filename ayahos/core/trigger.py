from seisbench.util.annotations import Pick, Detection
from ayahos.util.stats import est_curve_quantiles, fit_probability_peak
import numpy as np
from obspy import Trace

class GaussianModel(object):
    def __init__(self, mean=0, amplitude=1, stdev=1, skewness=0, kurtosis=3, kurt_type='Pearson'):
        if not all(isinstance(x, float) for x in [mean, amplitude, stdev, skewness, kurtosis]):
            raise TypeError
        if not isinstance(kurt_type, str):
            raise TypeError
        
        if np.isfinite(mean):
            self.mean = mean
        else:
            raise ValueError
        
        if np.isfinite(amplitude):
            self.amplitude = amplitude
        else:
            raise ValueError
        
        if np.isfinite(stdev):
            self.stdev = stdev
        else:
            raise ValueError
        
        if np.isfinite(skewness):
            self.skewness = skewness
        else:
            raise ValueError
        
        if kurt_type.lower() in ['fisher','pearson']:
            self.kurt_type = kurt_type.lower()
        else:
            raise ValueError

        if np.isfinite(kurtosis):
            self.kurtosis = kurtosis
        else:
            raise ValueError
        
    def

class Trigger(object):
    """
    This class serves as an extended container for storing statistical representations of individual modeled probabily peaks from
    probability time-series output by :class: `~seisbench.models.WaveformModel`-like objects. It is modeled after the
    :class: `~seisbench.util.annotations.Pick` class and the underlying structure of `obspy` trigger objects. It adds statistical measures under the ansatz that the probability
    curve is the form of a normal distribution probability density function. 

    :param trigger_trace: Trace-like object
    """    
    def __init__(self, trigger_trace, trigger_level, padding_samples=None, quantiles=[0.5]):
        if isinstance(trigger_trace, Trace):
            self.starttime = trigger_trace.stats.starttime
            self.sampling_rate = trigger_trace.stats.sampling_rate
            self.npts = trigger_trace.stats.npts
            self.id = trigger_trace.id
            self.site = trigger_trace.site
            if isinstance(trigger_trace.data, np.ma.MaskedArray):
                if not np.ma.is_masked(trigger_trace.data):
                    self.data = trigger_trace.data.filled()
            else:
                self.data = trigger_trace.data
        else:
            raise TypeError
        
        # Get maximum trigger level
        self.tmax = self.t0 + np.argmax(self.data)/self.sampling_rate
        self.pmax = np.max(self.data)

        if isinstance(trigger_level, float):
            if np.isfinite(trigger_level):
                self.trigger_level = trigger_level
            else:
                raise ValueError
        else:
            raise TypeError
        
        if isinstance(padding_samples, int):
            if padding_samples >= 0:
                self.padding_samples = padding_samples
            else:
                raise ValueError
        elif padding_samples is None:
            self.padding_samples = 0
        else:
            raise TypeError
        
        if isinstance(quantiles, float):
            if 0 < quantiles < 1:
                self.quantiles = [quantiles]
            else:
                raise ValueError
        elif isinstance(quantiles, list):
            if all(0 < q < 1 for q in quantiles):
                self.quantiles = quantiles
            else:
                raise ValueError
        else:
            raise TypeError
        # Run fitting / estimation processes
        self. = {'mean': None,
                        'amp': None,
                        'stdev': None,
                        'skew': None,
                        'kurt': None,
                        'kurt_type': None}
        self.nmod['mean']
        self.gau_amp = None
        self.gau_stdev = None
        self.gau_skew = None
        self.gau_kurt = None
        self.fit_gaussian()

        # Get residuals and misfits
        self.gau_res_l1 = None
        self.gau_res_l2 = None
        self.mean_peak_dt = None
        self.get_gaussian_residuals()
        self.quantile_values = [None]*len(self.quantiles)
        self.get_quantiles()
    
    def get_gaussian_representation(self, **options):
        """Estimate the mean, standard deviation, skewness, and kurtosis of this trigger
        treating the values in `self.data` as probability density values and their position as
        bin values. Wraps :meth:`~ayahos.util.features.est_curve_normal_stats`.

        :param **options: key word argument collector to pass to est_curve_normal_stats
        :type **options: kwargs
        """        
        x = np.arange(0, self.npts)
        amp, mu, sig, skew, kurt = fit_probability_peak(x, self.data, **options)
        self.gau_mean = mu/self.sampling_rate + self.starttime
        self.gau_amp = amp
        self.gau_stdev = sig/self.sampling_rate
        self.gau_skew = skew
        self.gau_kurt = kurt
    
    def get_gaussian_fit_residual(self, data):
        y = data
        x = np.arange(0, self.npts)
        Gm = (2.*np.pi*self.gau_std**2)**-0.5 * np.exp((-1.*(x - self.gau_mean)**2)/(2.*self.gau_std**2))
        rvect = Gm - y
        self.res_l1 = np.linalg.norm(rvect, ord=1)
        self.res_l2 = np.linalg.norm(rvect, ord=2)

    def get_quantiles(self, data):
        y = data
        x = np.arange(0, len(data))
