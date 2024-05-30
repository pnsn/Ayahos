# from seisbench.util.annotations import Pick, Detection
from ayahos.util.stats import estimate_moments, fit_normal_pdf_curve
import numpy as np
from ayahos import MLTrace
from scipy.cluster.vq import *
import logging
from collections import deque

Logger = logging.getLogger(__name__)

class GaussianModel(object):
    """An object hosting paramter fitting of a scaled gaussian model:

    .. math::
        y = p_0 e^{\\frac{(x - p_1)^2}{2p_2^2}}

    to input data {x, y}, with model parameters:
        - :math:`p_0 = A_{calc}`: prefactor, or amplitude
        - :math:`p_1 = \\mu_{calc}`: central value, or mean
        - :math:`p_2 = \\sigma_{calc}`: standard deviation

    The model also includes the paramter covariance matrix and model-data residuals.
     
    Also calculates and holds empirical estimates of the statistical moments of input data {x, y},
        - :math:`\\mu_{est}` -- central value or mean
        - :math:`\\sigma_{est}` -- standard deviation
        - :math:`\\mathcal{S}_{est}` -- skewness
        - :math:`\\mathcal{K}_{est}` -- kurtosis (Pearson or Fisher)

    :param kurt_type: type of kurtosis to use "Pearson" or "Fisher" (Pearson = Fisher + 3), defaults to Pearson
    :type kurt_type: str, optional
    :param dtype: specify alternative data type to use for all calculations, defaults to None
    :type dtype: None or numpy.float32-like, optional
    :assigns:
        - **self.kurt_type** (*str*) -- 'pearson' or 'fisher' assigned
        - **self.dtype** (*type* or *None*) -- type or none assigned, passed to :meth:`~ayahos.core.trigger.GaussianModel.estimate_moments`
    """    
    def __init__(self, kurt_type='Pearson', dtype=None):
        """Initialize a GaussianModel object

        :param kurt_type: type of kurtosis to use "Pearson" or "Fisher" (Pearson = Fisher + 3), defaults to Pearson
        :type kurt_type: str, optional
        :param dtype: specify alternative data type to use for all calculations, defaults to None
        :type dtype: None or numpy.float32-like, optional
        :assigns:
            - **self.kurt_type** (*str*) -- 'pearson' or 'fisher' assigned
            - **self.dtype** (*type* or *None*) -- type or none assigned, passed to :meth:`~ayahos.core.trigger.GaussianModel.estimate_moments`
    
        """        
        if kurt_type.lower() in ['fisher','pearson']:
            self.kurt_type = kurt_type.lower()
        else:
            raise ValueError
        if dtype is None:
            self.dtype = dtype
        else:
            try:
                np.ones(shape=(30,), dtype=dtype)
                self.dtype = dtype
            except TypeError:
                raise TypeError
        self.p = None
        self.cov = None
        self.res = None
        self.est_mean = None
        self.est_stdev = None
        self.est_skew = None
        self.est_kurt = None


    def fit_pdf_to_curve(self, x, y, threshold=0.1, mindata=30, p0=None):
        """Fit a scaled gaussian probability density function to the curve y=f(x) using
        the least squares fitting in :meth:`~ayahos.util.stats.fit_normal_pdf_curve`.

        :param x: independent variable values
        :type x: numpy.ndarray
        :param y: dependent variable values
        :type y: numpy.ndarray
        :param threshold: minimum y value to use for fitting, defaults to 0.1
        :type threshold: float, optional
        :param mindata: minimum number of datapoints required for a fitting, defaults to 30
        :type mindata: int, optional
        :param p0: initial parameter estimates [Amp, mean, stdev], defaults to None
        :type p0: 3-tuple-like or None, optional

        :updates:
            - **self.p** (*numpy.ndarray*) -- least squares best fit model parameters
            - **self.cov** (*numpy.ndarray*) -- model parameter covariance matrix
            - **self.res** (*numpy.ndarray*) -- model-data (:math:`y_{calc} - y`) residuals
        
        """        
        outs = fit_normal_pdf_curve(x,y,threshold=threshold, mindata=mindata, p0=p0)
        self.p = outs[0]
        self.cov = outs[1]
        self.res = outs[2]
        if self.dtype is not None:
            self.p = self.dtype(self.p)
            self.cov = self.dtype(self.cov)
            self.res = self.dtype(self.res)

    def estimate_moments(self, x, y):
        """Estimate first through fourth moments of the probability density function
        y = f(x)

        :param x: independent variable values
        :type x: numpy.ndarray
        :param y: dependent variable values
        :type y: numpy.ndarray
        :updates:
            - **self.est_mean** (*float*) -- central value / mean
            - **self.est_stdev** (*float*) -- standard deviation
            - **self.est_skew** (*float*) -- skewness
            - **self.est_kurt** (*float*) -- kurtosis (corresponding to **self.kurt_type**)
        """        
        if self.kurt_type == 'fisher':
            fisher = True
        else:
            fisher = False
        outs = estimate_moments(x, y, fisher=fisher, dtype=self.dtype)
        self.est_mean = outs[0]
        self.est_stdev = outs[1]
        self.est_skew = outs[2]
        self.est_kurt = outs[3]


class GaussTrigger(object):
    """
    This class serves as an extended container for storing statistical representations of individual modeled probabily peaks from
    probability time-series output by :class:`~seisbench.models.WaveformModel`-like objects. It is modeled after the
    :class:`~seisbench.util.annotations.Pick` class and the underlying structure of `obspy` trigger objects. It adds statistical measures under the ansatz that the probability
    curve is the form of a normal distribution probability density function. 

    :param source_trace: Trace-like object containing just data to be used for fitting to a :class:`~Ayahos.core.trigger.GaussianModel`
    :type source_trace: ayahos.core.mltrace.MLTrace
    :param trigger: trigger on and off indices (iON and iOFF), as generated by :meth:`~obspy.signal.trigger.trigger_onset` or wrapping methods
    :type trigger: tuple of int
    :param trigger_level: triggering level used to generate trigger
    :type trigger_level: float
    :param quantiles: quantiles to estimate with values expressed as  :math:`q\in(0,1)`, defaults to [0.159, 0.5, 0.841]
    :type quantiles: list-like of float
    :param **options: key word argument collector passed to :meth:`~ayahos.core.trigger.GaussianModel.estimate_moments`
    :type **options: kwargs
    
    :updated:
        - **self.starttime** (*obspy.core.utcdatetime.UTCDateTime) -- starting timestamp from **source_trace**
        - **self.sampling_rate** (*float*) -- sampling rate from **source_trace**
        - **self.network** (*str*) -- network attribute from **source_trace.stats**
        - **self.station** (*str*) -- station attribute from **source_trace.stats**
        - **self.location** (*str*) -- location attribute from **source_trace.stats**
        - **self.channel** (*str*) -- channel attribute from **source_trace.stats**
        - **self.model** (*str*) -- model attribute from **source_trace.stats**
        - **self.weight** (*str*) -- weight attribute from **source_trace.stats**
        - **self.iON** (*int*) -- trigger ON index value
        - **self.iOFF** (*int*) -- trigger OF index value
        - **self.trigger_level** (*float*) -- triggering threshold
        - **self.max** (*2-tuple*) -- index and value of maximum data value in source_trace.data[iON, iOFF]
        - **self.quantiles** (*dict of 2-tuple*) -- index and value of estimated quantiles, keyed by input quantile value
    
    :assigned:
        - **self.clustered** (*bool*) -- flag if this trigger has been incorporated into a cluster

    """    
    def __init__(self, source_trace, trigger, trigger_level, quantiles=[0.159, 0.5, 0.841], kurt_type='Pearson', **options):
        """Initialize a GaussTrigger object

        :param source_trace: _description_
        :type source_trace: _type_
        :param trigger: _description_
        :type trigger: _type_
        :param quantiles: _description_, defaults to [0.159, 0.5, 0.841]
        :type quantiles: list, optional
        :raises TypeError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises TypeError: _description_
        :raises ValueError: _description_
        :raises TypeError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises TypeError: _description_
        """        
        self.clustered = False
        if isinstance(source_trace, MLTrace):
            self.starttime = source_trace.stats.starttime
            self.sampling_rate = source_trace.stats.sampling_rate
            self.npts = source_trace.stats.npts
            self.network = source_trace.stats.network
            self.station = source_trace.stats.station
            self.location = source_trace.stats.location
            self.channel = source_trace.stats.channel
            self.model = source_trace.stats.model
            self.weight = source_trace.stats.weight
            if isinstance(source_trace.data, np.ma.MaskedArray):
                if not np.ma.is_masked(source_trace.data):
                    self.data = source_trace.data.filled()
            else:
                self.data = source_trace.data
        else:
            raise TypeError
        
        if isinstance(trigger, (list, tuple)):
            if len(trigger) == 2: 
                if trigger[0] < trigger[1]:
                    self.iON = trigger[0]
                    self.iOFF = trigger[1]
                else:
                    raise ValueError('trigger ON index is larger than OFF index')
            else:
                raise ValueError('trigger must be a 2-list or 2-tuple')
        else:
            raise TypeError('trigger must be type list or tuple')


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
        self.lsqmod = GaussianModel()
        x = np.arange(self.iON, self.iOFF)/self.sampling_rate + self.starttime.timestamp
        y = self.data
        self.lsqmod.fit_pdf_to_curve(x, y, **options)
        # Estimate the mean, stdev, skewness, and kurtosis 
        self.lsqmod.estimate_moments(x, y)

    def get_site(self):
        rstr = f'{self.network}.{self.station}'
        return rstr
    
    site = property(get_site)

    def get_id(self):
        rstr = f'{self.network}.{self.station}.{self.location}.{self.channel}.{self.model}.{self.weight}'
        return rstr
    
    id = property(get_id)

    def get_label(self):
        rstr = self.channel[-1]
        return rstr
    
    label = property(get_label)


class TriggerBuffer(deque):
    """A collections.deque-like object that buffers :class:`~ayahos.core.trigger.GaussTrigger` objects that
    fall within a specified time-range of the chronologically newest GaussTrigger object. This
    class provides methods built on :mod:`~scipy.cluster.vq` methods for K-means clustering and
    :mod:`~scipy.cluster.heirarchical` methods for agglomerative clustering. Both methods are shown
    to be performant at scale, as illustrated here:

    https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html

    :param max_length: maximum buffer length in seconds, defaults to 1
    :type max_length: float, optional
    :param keys: what MLTrace ID naming components to use for key matching, defaults to site
        Supported values: 'site' or 'id', which should match the hosting :class:`~ayahos.core.
    :param parameter_set: name of parameter set from each GaussTrigger object to use for features, defaults to 'lsq'
            Supported values:
                - 'lsq' -- GaussianModel parameters :math:`\\mathcal{A}, \\mu, \\sigma^2`, and L-2 norm of residuals
                - 'est' -- Estimated moments :math:`\\mu_{est}` and :math:`\\sigma^2`, maximum probability value :math:`\mathcal{P}_{max}` and location :math:`t_{max}`
                - 'all' -- GaussianModel parameters (as in lsq) and empirical parameters (as in est)
    :type parameter_set: str, optional
    
    """
    def __init__ (
            self,
            max_length=1,
            key='site',
            label='P',
            restrict_past_append=True,
            parameter_set='lsq',
            clustering_method='kmeans',
            distance_metric='ttest',

            ):
        if isinstance(max_length, (float, int)):
            if np.isfinite(max_length):
                if 0 < max_length <= 1e3:
                    self.max_length = max_length
                elif max_length > 1e3:
                    Logger.warn('max_length g.t. 1000 seconds, may be unnecessarily long')
                    
            
        # Inherit from collections.deque
        super().__init__()
        self.id_counts = {}
        self.max_timestamp = None
        self.min_timestamp = None
        self.cluster_centroids = {}
        self.cluster_membership = []

    def append(self, other, **options):
        if isinstance(other, GaussTrigger):
            pass
        else:
            raise TypeError('other must be type ayahos.core.trigger.GaussTrigger')
        self.__iadd__(other, **options)


    def _update_times(self):

    def _

    def _update_id_count(self):
        for trigger in self.triggers:
            _id = trigger.


    def cluster(self):
        if self.cluster_method = 'kmeans':
            self.prewhiten()
            centroids, distortion = kmeans(
                self.pwfeatures,
                )

#     self.nmod['mean']
#     self.gau_amp = None
#     self.gau_stdev = None
#     self.gau_skew = None
#     self.gau_kurt = None
#     self.fit_gaussian()

#     # Get residuals and misfits
#     self.gau_res_l1 = None
#     self.gau_res_l2 = None
#     self.mean_peak_dt = None
#     self.get_gaussian_residuals()
#     self.quantile_values = [None]*len(self.quantiles)
#     self.get_quantiles()

# def get_gaussian_representation(self, **options):
#     """Estimate the mean, standard deviation, skewness, and kurtosis of this trigger
#     treating the values in `self.data` as probability density values and their position as
#     bin values. Wraps :meth:`~ayahos.util.features.est_curve_normal_stats`.

#     :param **options: key word argument collector to pass to est_curve_normal_stats
#     :type **options: kwargs
#     """        
#     x = np.arange(0, self.npts)
#     amp, mu, sig, skew, kurt = fit_probability_peak(x, self.data, **options)
#     self.gau_mean = mu/self.sampling_rate + self.starttime
#     self.gau_amp = amp
#     self.gau_stdev = sig/self.sampling_rate
#     self.gau_skew = skew
#     self.gau_kurt = kurt

# def get_gaussian_fit_residual(self, data):
#     y = data
#     x = np.arange(0, self.npts)
#     Gm = (2.*np.pi*self.gau_std**2)**-0.5 * np.exp((-1.*(x - self.gau_mean)**2)/(2.*self.gau_std**2))
#     rvect = Gm - y
#     self.res_l1 = np.linalg.norm(rvect, ord=1)
#     self.res_l2 = np.linalg.norm(rvect, ord=2)

# def get_quantiles(self, data):
#     y = data
#     x = np.arange(0, len(data))
