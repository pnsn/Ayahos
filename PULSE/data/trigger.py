"""
:module: PULSE.data.pick
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose: This module contains three classes and supporting methods for progressive
    distillation of characteristic response function peaks into 

    :class:`~.Trigger` - contains time-series data/fold vectors inherited from
        a source charactristic response function trace and essential metadata
        for the trace ID, detection method, and triggering parameters. Provides
        methods for casting :class:`~.Trigger` object contents into :class:`~.TriggerModel`
        object(s) with :meth:`~.Trigger.fit_model`
    
    :class:`~.Model` - distills time-series data/fold vectors into compact
        numerical representations of triggers for a range of functional forms, 
        statistical measures, and the fit-quality thereof.

    :class:`~.Pick` - an :class:`~.AttribDict` that summarizes essential data,
        model parameters, and metadata for a labeled point-in-time feature. Provides
        methods for casting these values into a range of commonly-used pick formats
        
"""


import logging
from PULSE.util.stats import (estimate_moments,
                              estimate_quantiles,
                              rsquared,
                              lsq_model_fit)
import numpy as np
from scipy.optimize import leastsq

from obspy import Trace, UTCDateTime
from obspy.core.event import Pick, WaveformStreamID, ResourceIdentifier
from obspy.core.util.attribdict import AttribDict



from PULSE.data.foldtrace import FoldTrace


Logger = logging.getLogger(__name__)

class Trigger(FoldTrace):
    """A :class:`~.FoldTrace` child class intended to encapsulate 
    continuous time-series data and observation density (fold) of
    a single characteristic response function trigger

    :param FoldTrace: _description_
    :type FoldTrace: _type_
    """    

    def __init__(self, source_trace, samp_on, samp_off, thr_on, thr_off, samp_pad=None):
        """Initialize a :class:`~.Trigger` object

        :param source_trace: Trace-like object from which a trigger was derived
            E.g., the characteristic response function trace
        :type source_trace: PULSE.data.foldtrace.FoldTrace
        :param samp_on: sample index of the trigger onset in **source_trace**
        :type samp_on: int
        :param samp_off: sample index of the trigger offset in **source_trace**
        :type samp_off: int
        :param thr_on: trigger onset threshold value
        :type thr_on: float-like
        :param thr_off: trigger offset threshold value
        :type thr_off: float-like
        :param samp_pad: Additional padding samples to include before **samp_on** and
            after **samp_off** used to inform statistical measures of the trigger 
            morphology, defaults to None
            None - uses only samples from samp_on to samp_off
            int - additional samples on both sides
        :type samp_pad: NoneType or int, optional

        """        
        if not isinstance(source_trace, FoldTrace):
            raise TypeError

        if samp_pad is None:
            samp_pad = 0

        # Uses __setattr__ for compatability checks
        self.samp_pad = samp_pad
        self.samp_on = samp_on
        self.samp_off = samp_off
        self.thr_on = thr_on
        self.thr_off = thr_off
        # Generate view of input source_trace
        tts = source_trace.stats.starttime + \
            (self.samp_on - self.samp_pad)*source_trace.stats.delta
        tte = source_trace.stats.starttime + \
            (self.samp_off + self.samp_pad)*source_trace.stats.delta
        view = source_trace.view(starttime=tts, endtime=tte).copy()

        if not all(view.data >= 0):
            raise ValueError('View of source_trace contains negative values - are you sure this is a CRF?')

        super().__init__(data=view.data,
                         fold=view.fold,
                         header=view.stats)
        
        # Update samp_on and samp_off to new trimmed view of data
        self.samp_on = self.samp_pad
        self.samp_off = self.count() - self.samp_pad - 1
        
    def __setattr__(self, key, value):
        """Enforce Type and Value rules for setting attribute values new
        to this :class:`~.Trigger` object.

        :param key: name of the attribute being set
        :type key: str
        :param value: new value to set to the attribute **key**
        :type value: varies
            - samp_X attributes: int-like
            - thr_X attributes: float-like
        """        
        if key in ['samp_pad','samp_on','samp_off']:
            try:
                value = int(value)
            except:
                TypeError(f'could not convert {key} {value} into int')
            # if isinstance(value, (int, float)):
            #     if value == int(value):
            #         value = int(value)
            #     else:
            #         raise TypeError(f'{key} must be int-like')
            # else:
            #     breakpoint()
            #     raise TypeError(f'{key} must be int-like')
            if value < 0:
                raise ValueError(f'{key} must be non-negative')
        if key in ['thr_on','thr_off']:
            if isinstance(value, (int, float)):
                value = float(value)
            else:
                raise TypeError(f'{key} must be float-like')
            if value <= 0:
                raise ValueError(f'{key} must be positive')
            
        super().__setattr__(key, value)
    
    def get_max_index(self):
        """Return the index of the maximum
        value

        :return: _description_
        :rtype: _type_
        """        
        return np.argmax(np.abs(self.data))

    def get_max_time(self):
        """Get the timestamp of the absolute maximum value in the
        **data** attribute of this :class:`~.Trigger` object

        :return: _description_
        :rtype: _type_
        """
        imax = self.get_max_index()
        return self.stats.starttime + imax*self.stats.sampling_rate
    
    tmax = property(get_max_time)
    samp_max = property(get_max_index)
    pmax = property(max)

    def to_model(self, model='gaussian', fitter='lsq', **options):
        """Cast (meta)data from this :class:`~PULSE.data.pick.Trigger` object into
        a :class:`~PULSE.data.pick.Model` object for the specified
        model type and fitting approach

        :param model: model name, defaults to 'gaussian'
        :type model: str, optional
        :param fitter: fitter name, defaults to 'lsq'
        :type fitter: str, optional
        :returns: **model** (*PULSE.data.pick.Model*) -- trigger model object
        """        
        model = Model(self, model=model, fitter=fitter, **options)
        return model
            

class Model(object):
    meta_fields = {'network','station','location','channel','starttime','sampling_rate','npts'}

    def __init__(self, trigger, model='gaussian', fitter='lsq', **options):
        if not isinstance(trigger, Trigger):
            raise TypeError
        if model not in ['trigger','quantile','gaussian','triangle','uniform']:
            raise NotImplementedError
        else:
            self.model = model
        if fitter not in ['lsq','est']:
            raise NotImplementedError
        else:
            self.fitter = fitter

        # Populate Stats
        stats = trigger.stats.copy()
        self.stats = AttribDict({_k: stats[_k] for _k in self.meta_fields})
        # Update with mean fold, npts, and padding samples
        self.stats.update({'fold_mean': np.mean(trigger.fold),
                           'padding': trigger.samp_pad,
                           'thr_on': trigger.thr_on,
                           'thr_off': trigger.thr_off})

        # Create sample index
        xv = np.arange(trigger.count())

        # Least Squares Fittings
        if fitter == 'lsq':

            if model not in ['gaussian','triangle','uniform']:
                raise NotImplementedError
   
            # Run Fitting & directly capture params and pcov
            self.m, self.Cm, res = lsq_model_fit(xv, trigger.data,
                                                        model=model, **options)
            # Use residuals to get fit-quality-metrics
            self.mfq = self.fit_quality_metrics(res)
    
            # Label model / covariance matrix entries
            if model == 'gaussian':
                self.names = ['amp','mean','var']
            elif model == 'triangle':
                self.names = ['amp','onset','peak','offset']
            elif model == 'uniform':
                self.names = ['amp','onset','offset']
            else:
                breakpoint()
                raise NotImplementedError(f'somehow got into lsq for {model}...debugging required')
            
        # Estimated Statistical Measures
        elif fitter == 'est':
                        
            # All estimated parameter sets are assumed to have IID parameters
            self.Cm = np.eye(len(self.m))

            # Trigger parameters
            if model == 'trigger':
                self.m = np.array([trigger.pmax, trigger.samp_on, trigger.samp_max, trigger.samp_off])    
                self.Cm = np.eye(len(self.m))
                self.names = ['amp','onset','peak','offset']
            # Quantile estimates from CRF data
            elif model == 'quantile':
                loc, prob = estimate_quantiles(xv, trigger.data, **options)
                self.m = loc
                self.Cm = np.eye(len(self.m))
                self.names = [f'q{_y:.3f}' for _y in prob]
            # Gaussian moment estimates from CRF data
            elif model == 'gaussian':
                moments = estimate_moments(xv, trigger.data, **options)
                pmean = trigger.data[np.round(moments[0])]
                self.m = np.array([pmean] + [_m for _m in moments])
                self.names = ['amp','mean','std','skewness','kurtosis']
            else:
                breakpoint()
        else:
            breakpoint()

    def model_fit_quality(self, res, obs):
        mfq = {'l2': np.linalg.norm(res, ord=2),
              'l1': np.linalg.norm(res, ord=1),
              'r2':rsquared(res, obs)}
        return mfq  
    
    def __repr__(self):
        rstr = ''
        for _k, _v in self.stats.items():
            rstr += f'{_k:>16}:{_v}\n'
        rstr += f'-- {self.model} --\n'
        rstr += f'm:  {self.m}\n'
        if self.fitter == 'lsq':
            rstr += f'Cm: {self.Cm}\n'
        return rstr
    

##### SUPPORTING METHODS #####

## LSQ FITTING FUNCTIONS ###
# TODO: eventually re-cast these as scipy.stats.rv_discrete objects

def scaled_gaussian(p, x):
    """Discrete representation of a scaled Gaussian distribution (bellcurve)
    
    :math:`p_0 e^{\\frac{-(x - p_1)^2}{2 p_2}}`,
    with 
     - :math:`p_0` = amplitude
     - :math:`p_1` = mean / central value
     - :math:`p_2` = variance (standard deviation squared)

    Suggest Use Case
    ----------------
    Outputs form ML model predictions for phase arrivals
    and other similar point-in-time features that produce
    prediction time-series with bell-curve like shapes arising
    from the underlying series of convolutions in these models.

    :param p: model parameter vector
    :type p: array-like
    :param x: sample domain vector
    :type x: array-like
    :returns: **y** (*numpy.ndarray*) -- model value vector
    """    
    amp = p[0]
    mean = p[1]
    var = p[2]
    y = amp*np.exp((-1.*(x - mean)**2)/(2.*var))
    return y

def scaled_uniform(p, x):
    """Discrete representation of a scaled uniform distribution

                   / U = 1, x in [p1, p2)
    y = p0*U[x] = |
                   \ U = 0, else

    p = [amplitude, onset, offset]

    Suggested Use Case
    ------------------
    Outputs from ML model predictions for detection
    (EQTransformer) and noise (PhaseNet) labels can
    be roughly approximated with scaled uniform distributions

    For training EQTransformer, uniform distributions were used
    to train the detection label (Mousavi et al., 2022)

    :param p: parameter vector
    :type p: array-like
    :param x: sample domain vector
    :type x: array-like
    :returns: **y** (*numpy.ndarray*) - modeled amplitude vector
    """

    amp = p[0]
    onset = p[1]
    offset = p[2]
    y = np.zeros(x.shape)
    idx = (x <= onset) & (x + 1 < offset)
    y[idx] += amp
    return y


def scaled_triangle(p, x):
    """Discrete represenation of a scaled triangular distribution
                  / T = 0, x <= p1
                  | T = 2(x - p1)/(p2 - p1)(p3 - p1), x in (p1, p2)
    y = p0*T[x] = | T = 1, x = p2
                  | T = 2(p2 - x)/(p2 - p1)(p2 - p3), x in (p2, p3]
                  \ T = 0, x > p3

    p = [amplitude, onset, peak, offset]
    
    Suggested Use Cases
    -------------------
     - Symmetric triangular models are used for pick time training data
       (e.g., Mousavi et al., 2022; Ni et al., 2024) 
     
     - Characteristic response functions (e.g., classic STA/LTA) can
       be roughly approximated as asymmetric triangular models.

    :param p: parameter vector
    :type p: array-like
    :param x: sample domain vector
    :type x: array-like
    :returns: **y** (*numpy.ndarray*) - modeled amplitude vector 
    """


    amp = p[0]
    start = p[1]
    peak = p[2]
    end = p[3]

    y1 = (x - start)/(peak - start)
    y1 *= (y1 >= 0) & (y1 <= 1)
    y2 = (x - end)/(peak - end)
    y2 *= (y2 >= 0) & (y2 < 1)
    y = amp*(y1 + y2)

    return y

def data_model_errors(p, x, y, model):
    """Calculate the data-model residual vector

    :param p: model parameter vector
    :type p: numpy.ndarray
    :param x: sample position vector
    :type x: numpy.ndarray
    :param y: observed sample value vector
    :type y: numpy.ndarray
    :param model: model name
        Supported: 'gaussian','triangle','uniform'
    :type model: _type_
    :returns: **res** (*numpy.ndarray*) -- data-model residual vector
    """    
    if model == 'gaussian':
        y_calc = scaled_gaussian(p, x)
    elif model == 'triangle':
        y_calc = scaled_triangle(p, x)
    elif model == 'uniform':
        y_calc = scaled_uniform(p, x)
    else:
        raise NotImplementedError
    
    res = y - y_calc
    return res

def lsq_model_fit(x_obs, y_obs, model='gaussian', p0=None):
    """Fit a specified model type to observed data vectors
    using :meth:`~scipy.optimize.leastsq`

    :param x_obs: observed sample positions
    :type x_obs: numpy.ndarray
    :param y_obs: observed sample values
    :type y_obs: numpy.ndarray
    :param model: name of the model to fit, defaults to 'gaussian'
        Supported: 'gaussian','triangle','uniform'
    :type model: str, optional
    :param p0: initial parameters, defaults to None
        Passed to :meth:`~scipy.optimize.lsq
        If "None" the 
    :type p0: NoneType or array-like, optional

    :returns:
     - **popt** (*numpy.ndarray*) -- LSQ best-fit model parameter vector
     - **pcov** (*numpy.ndarray*) -- LSQ model covariance matrix
     - **res** (*numpy.ndarray*) -- data-model residual vector
    """    
    if np.ndim(x_obs) != 1:
        raise ValueError
    if np.ndim(y_obs) != 1:
        raise ValueError
    if x_obs.shape != y_obs.shape:
        raise ValueError
    if model not in ['gaussian','triangle','uniform']:
        raise ValueError(f'"model" {model} not supported')
    
    # If initial parameters is None, do a reasonable first-pass estimate from the data
    if p0 is None:
        # For all, max value
        p0 = [np.nanmax(y_obs)]
        if model == 'gaussian': 
            # mean position, half-width squarred
            p0 += [np.nanmean(x_obs), (0.5*(np.nanmax(x_obs) - np.nanmin(x_obs)))**2]
        elif model == 'triangle':
            # max position, left edge, right edge
            p0 += [np.argmax(y_obs), np.nanmin(x_obs), np.nanmax(x_obs)]
        elif model == 'uniform':
            p0 += [np.nanmin(x_obs), np.nanmax(x_obs)]

    popt, pcov = leastsq(data_model_errors, p0, args=(x_obs, y_obs, model), full_output=True)
    res = data_model_errors(popt, x_obs, y_obs, model=model)
    return popt, pcov, res


## STATISTICAL MEASURE ESTIMATION METHODS ###

def estimate_quantiles(x, y, q=[0.16, 0.5, 0.84]):
    """Approximate the quantiles of a evenly binned population represented as a discrete y = f(x) using the following formulation:

    .. math::
        i = argmin\\left(\\left|\\frac{q - cumsum(y)}{\Sigma_i y_i}\\right|\\right)
    .. math::
        q_x = x_i, q_y = y_i

    :param x: independent parameter values
    :type x: numpy.ndarray
    :param y: dependent parameter, values y[i] stand as frequency proxy for values x[i]
    :type y: numpy.ndarray
    :param q: quantiles to calculate with :math: `q \\in [0, 1]`, defaults to [0.159, 0.5, 0.841]
        NOTE: 
    :return: 
        - **qx** (*numpy.ndarray*) -- Approximated quantile positions (x-values)
        - **qy** (*numpy.ndarray*) -- Approximated quantile probability values (y-values)

    Notes:
        - cumsum and sum are called as :meth:`~numpy.nancumsum` and  :meth:`~numpy.nansum` to strengthen method against NaN values.
        - Default `q` values approximate the mean (:math:`\\mu`) and :math:`\\mu\\pm 1\\sigma` values of a normal distribution.
    """
    if 0.5 not in q:
        q.append(0.5)
    q.sort
    csy = np.nancumsum(y)
    sy = np.nansum(y)
    qx = np.array([x[np.argmin(np.abs(_q - csy / sy))] for _q in q])
    qy = np.array([y[np.argmin(np.abs(_q - csy / sy))] for _q in q])
    return qx, qy


def estimate_moments(x, y, fisher=False, dtype=None):
    """
    Estimate the mean and standard deviation of a population represented
    by a discrete, evenly sampled function y = f(x), using y as weights
    and x as population bins.

    Estimates are made as the weighted mean and the weighted standard deviation:
    https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf

    Estimates of skewness and kurtosis are made using the weighted formulation for the 3rd and 4th moments described here:
    https://www.mathworks.com/matlabcentral/answers/10058-skewness-and-kurtosis-of-a-weighted-distribution

    :param x: independent variable values (population bins)
    :type x: numpy.ndarray
    :param y: dependent variable values (weights)
    :type y: numpy.ndarray
    :param fisher: should the kurtosis be calculated as the Fisher formulation? I.e., normal distribution has ikurt = 0. Defaults to False
    :type fisher: bool, optional
    :param dtype: optional data type (re)formatting, defaults to None
    :type dtype: type, optional
    :returns:
        - **est_mean** (*float*) -- estimated central value (mean)
        - **est_stdev** (*float*) -- estimated standard deviation
        - **est_skew** (*float*) -- estimate skewness
        - **est_kurt** (*float*) -- estimated kurtosis (kurt = kurt - 3 for `fisher = True`)
    """
    if not isinstance(x, np.ndarray):
        try:
            x = np.array(x)
        except:
            raise TypeError
    if dtype is None:
        pass
    elif x.dtype != dtype:
        x = x.astype(dtype)

    if not isinstance(y, np.ndarray):
        try:
            y = np.array(y)
        except:
            raise TypeError
    if dtype is None:
        pass
    elif y.dtype != dtype:
        y = y.astype(dtype)

    # Remove the unweighted mean (perhaps redundant)
    dx = x - np.nanmean(x)
    # Calculate y-weigted mean of delta-X values
    dmean = np.nansum(dx * y) / np.nansum(y)
    # Then add the unweighted mean back in
    est_mean = dmean + np.nanmean(x)

    # Calculate the y-weighted standard deviation of delta-X values
    # Compose elements
    # Numerator
    std_num = np.nansum(y * (dx - dmean) ** 2)
    # N-prime: number of non-zero (finite) weights
    Np = len(y[(y > 0) & (np.isfinite(y))])
    # Denominator
    std_den = (Np - 1.0) * np.nansum(y) / Np
    # Compose full expression for y-weighted std
    est_std = np.sqrt(std_num / std_den)

    # Calculate weighted 3rd moment
    wm3 = np.nansum(y * (dx - dmean) ** 3.0) / np.nansum(y)
    # And weighted skewness
    est_skew = wm3 / est_std**3.0

    # Calculate weighted 4th moment
    wm4 = np.nansum(y * (dx - dmean) ** 4.0) / np.nansum(y)
    # And weighted kurtosis
    est_kurt = wm4 / est_std**4.0
    if fisher:
        est_kurt -= 3.0

    # Calculate weighted 4th moment (kurtosis)
    return est_mean, est_std, est_skew, est_kurt


### DISTRIBUTION COMPARISON METHODS ###

def ssmd(mean_ref, mean_test, var_ref, var_test):
    """Calculate the strictly standardized mean difference
    of two gaussian distributions

    :param mean_ref: reference distribution mean
    :type mean_ref: float-like
    :param mean_test: test distribution mean
    :type mean_test: float-like
    :param var_ref: reference distribution variance
    :type var_ref: float-like
    :param var_test: test distribution variance
    :type var_test: float-like
    :returns: **beta** (*float-like*) -- strictly standardized mean difference
    """    
    num = np.abs(mean_ref - mean_test)
    den = np.sqrt(var_ref + var_test)
    beta = num/den
    return beta


def kld_gaussian(mean_ref, mean_test, var_ref, var_test):
    """
    Calculate the Kullbeck-Leibler divergence score of
    a reference Gaussian distribution and an other
    Gaussian distribution 

    :math:`\\frac{1}{2} \\left( \\frac{\\sigma_0^2}{\\sigma_1^2} + \\frac{(\mu_1 - \mu_0)^2}{\\sigma_1^2} - 1 ln \\left(\\frac{\\sigma_1^2}{\\sigma_0^2}\\right) \\right)`

    with 
     - :math:`\\mu_0` the true mean
     - :math:`\\mu_1` the estimated mean
     - :math:`\\sigma_0^2` the true distribution variance
     - :math:`\\sigma_1^2` the estimated distribution variance

    Thus the KL divergence approaches 0 as the distributions become
    more similar and is unbounded as they become more distinct.

    also see: https://en.wikipedia.org/wiki/Kullback–Leibler_divergence

    :param mean_ref: reference distribution mean
    :type mean_ref: float-like
    :param mean_test: test distribution mean
    :type mean_test: float-like
    :param var_ref: reference distribution variance
    :type var_ref: float-like
    :param var_test: test distribution variance
    :type var_test: float-like
    :returns: **kdl** (*float-like*) -- Kullbeck-Leibler divergence score
    """
    p0 = var_ref/var_test
    p1 = ((mean_test - mean_ref)**2)/var_test
    p2 = var_test/var_ref
    kdl = 0.5*(p0 + p1 - 1. + np.log(p2))
    return kdl















# class Model(AttribDict):
#     """An :class:`~.AttribDict` object designed to hold a parametric
#     representation of a characteristic response function trigger curve,
#     essential metadata, and parameteric representation misfit quantities

#     Metadata
#     --------
#      - network -- source trace network code
#      - station -- source trace station code
#      - location -- source trace location code
#      - channel -- source trace channel code
#      - detector -- detection method name (e.g., 'classicstalta', 'EQTransformer.pnw')
#      - label -- feature label (e.g., 'detection', 'P', 'S', 'N')

#     Parametric Information
#     ----------------------
#      - starttime -- reference start time for converting elements of **m** from index values to time 
#      - sampling_rate -- sampling rate for converting elements of **m** from index values to time 
#      - m -- model/estimated value vector
#      - Cm -- model/estimated value covariance matrix (for fitter='obs', Cm = Identity Matrix)
#      - model_labels -- names of entries/rows/columns in **m** and **Cm**
    
#     Model Information
#     ----------------- 
#      - model_type -- parametric representation name
#         - 'trigger' -- **m** contains trigger onset, offset, and peak values
#         - 'gaussian' -- **m** contains parameters for a scaled gaussian/normal distribution
#         - 'quantile' -- **m** contains quantiles estimated from trigger samples
#         - 'triangle' -- **m** contains parameters estimated for a scaled triangular distribution
#         - 'uniform' -- **m** contains parameters estimate for a scaled uniform distribution
#      - fitter -- parametric representation fitting method name
#         - 'obs' -- **m** entries estimated from observations
#         - 'lsq' -- **m** and **Cm** estimated from least squares fitting of **model_type**

#     Fit Quality Metrics
#     -------------------
#     Note: These are only populated if using fitter='lsq'
#      - rl2 -- L2 norm of data-model residuals
#      - lr1 -- L1 norm of data-model residuals
#      - R2 -- R squared metric of data-model residuals



#     :param AttribDict: _description_
#     :type AttribDict: _type_
#     :raises TypeError: _description_
#     :raises TypeError: _description_
#     :raises KeyError: _description_
#     :raises KeyError: _description_
#     :raises KeyError: _description_
#     :raises ValueError: _description_
#     :return: _description_
#     :rtype: _type_
#     """

#     readonly = ['model_labels']

#     defaults = {'network': '',
#                 'station': '',
#                 'location': '',
#                 'channel': '',
#                 'detector': '',
#                 'label': '',
#                 'model_type': '',
#                 'fitter': '',
#                 'sampling_rate': 1.,
#                 'starttime': UTCDateTime(0),
#                 'mean_fold': np.nan,
#                 'm': np.array([]),
#                 'Cm': np.array([]),
#                 'model_labels': [],
#                 'rl2': np.nan,
#                 'rl1': np.nan,
#                 'R2': np.nan}
    
#     _types = {'network': str,
#               'station': str,
#               'location': str,
#               'channel': str,
#               'detector': str,
#               'label': str,
#               'model_type': str,
#               'fitter': str,
#               'sampling_rate': float,
#               'starttime': UTCDateTime,
#               'mean_fold': float,
#               'm': np.ndarray,
#               'Cm': np.ndarray,
#               'model_labels': list,
#               'rl2': float,
#               'rl1': float,
#               'R2': float}
    

#     def __init__(self, **kwargs):
#         self.update(kwargs)
    
#     def __setattr__()

#     _readonly = ['pick_time']
#     _refresh_keys = ['pick_param','params','ref_time','ref_sampling_rate']

#     defaults = {'network': '',
#                 'station': '',
#                 'location': '',
#                 'channel': '',
#                 'detection_method': '',
#                 'trigger_model': '',
#                 'pick_time': None,
#                 'label': '',
#                 'ref_time': UTCDateTime(0),
#                 'ref_sampling_rate': 1.,
#                 'pick_param':None,
#                 'params': {},
#                 'pcov': None,
#                 'fit_quality': np.nan,
#                 'fit_metric': None}
    
#     _types = {'network': str,
#               'station': str,
#               'location': str,
#               'channel': str,
#               'detection_method': str,
#               'trigger_model': str,
#               'pick_time': (type(None), UTCDateTime),
#               'label': str,
#               'ref_time': UTCDateTime,
#               'ref_sampling_rate': (int, float),
#               'pick_param': (type(None), str),
#               'params': dict,
#               'pcov': (type(None), np.ndarray),
#               'fit_quality': float,
#               'fit_metric': (type(None), str)
#               }
    
#     ## DUNDER METHODS ##
#     def __init__(self, header={}):
#         if not isinstance(header, dict):
#             raise TypeError('header must be type dict')
#         # If params is specified, set this first
#         if 'params' in header.keys():
#             self.__setattr__('params', header.pop('params'))
#         # If pcov is specified set this second
#         if 'pcov' in header.keys():
#             self.__setattr__('pcov', header.pop('pcov'))
        
#         # Finally, set the rest of the parameters
#         for _k, _v in header.items():
#             self.__setattr__(_k, _v)
    
#     def __setattr__(self, key, value):
#         # Cross-check with defaults and _types
#         if key in self.defaults.keys():
#             if isinstance(value, self._types[key]):
#                 pass
#             else:
#                 raise TypeError(f'value of type {type(value)} not supported for "{key}"')
#         else:
#             raise KeyError(f'key "{key}" not in defaults')
        
#         if key in self._readonly:
#             raise KeyError(f'{key} is readonly')

#         # Make sure pick_param is specified in params
#         if key == 'pick_param':
#             if value is None:
#                 return
#             elif value not in self.params.keys():
#                 raise KeyError('cannot assign "pick_param" as "{value}" - not mapped in "params" keys')

#         # Make sure pcov matches scale of params, if not None
#         if key == 'pcov':
#             if value is None:
#                 return
#             elif value.shape != (len(self.params), len(self.params)):
#                 raise ValueError('scale of parameter covariance matrix (pcov) does not match scale of "params"')

#         if key in self._refresh_keys:
#             super(Pick, self).__setitem__(key, value)
#             self.__dict__['pick_time'] =  self._get_pick_time()
#             return
#         else:
#             super(Pick, self).__setattr__(key, value)

#     def _get_pick_time(self):
#         """Supporting private method to calculate the pick_time from
#         other writeable entries
#         """        
#         t0 = self.ref_time
#         if len(self.params) > 0 and self.pick_param != self.defaults['pick_param']:
#             dt = self.params[self.pick_param]/self.sampling_rate
#             return t0 + dt
#         else:
#             return None
        
#     def __str__(self):
#         prioritized_keys = ['network','station','location','channel',
#                             'pick_time','label','detection_method',
#                             'trigger_model','pick_param']
#         return self._pretty_str(prioritized_keys=prioritized_keys)
    
#     def _repr_pretty(self, p, cycle):
#         p.text(str(self))


#     def get_seed_string(self):
#         return '.'.join([self[_k] for _k in ['network','station','location','channel']]) 

#     id = property(get_seed_string)   


# # class GaussianModel(AttribDict)




    # ### STATISTICAL MEASURE METHODS ###
        
    # def estimate_gaussian_moments(self, fisher=False):
    #     """Under the assumption that this :class:`~.Trigger`'s **data** 
    #     has a similar shape to a normal/Gaussian probability density function,
    #     estimate the mean, standard deviation, skewness, and kurtosis of the
    #     roughly Gaussian PDF. This method provides options to use the 
    #     Pearson or Fisher definitions of Kurtosis

    #     :param fisher: Use the Fisher definition of kurtosis?
    #         I.e., a true normal distribution has a kurtosis of 3.
    #         Defaults to False
    #     :type fisher: bool, optional
    #     :returns: **result** (*dict*) - dictionary with keyed values for
    #         the mean, std, skew, and kurt_X. The "_X" corresponds
    #         to the kurtosis definition used.
    #     """        
    #     xv = np.arange(0, self.count())
    #     mean, std, skew, kurt = estimate_moments(xv, self.data, fisher=fisher) 

    #     out = [mean*self.stats.delta, std*self.stats.delta, skew, kurt]
    #     names = ['mean','std','skew']
    #     if fisher:
    #         names.append('kurt_f')
    #     else:
    #         names.append('kurt_p')
        
    #     result = dict(zip(names, out))
    #     if not hasattr(self.measures, 'moments'):
    #         self.measures.moments = result
    #     else:
    #         self.measures.moments.update(result)
    #     return result
    
    # def estimate_quantiles(self, quantiles=[0.16, 0.50, 0.84], decimals=4):
    #     """Under the assumption that this :class:`~.Trigger`'s **data**
    #     represent a probability density function, estimate the specified
    #     quantiles of that PDF.

    #     :param quantiles: Quantiles to estimate, defaults to [0.16, 0.50, 0.84]
    #     :type quantiles: list, optional
    #     :param decimals: quantile precision for labeling quantile estimates
    #         Not used in calculating the actual quantiles. Defaults to 4
    #     :type decimals: int, optional
    #     :returns: **result** (*dict*) -- dictionary with keys of **quantiles** 
    #         and the delta seconds values relative to this :class:`~.Trigger`'s
    #         **starttime** for each specified quantile

    #     """        
    #     xv = np.arange(0, self.count())*self.stats.delta
    #     out = estimate_quantiles(xv, self.data, q=quantiles)
    #     # Convert into delta seconds
    #     out = [_o * self.stats.delta for _o in out]
    #     names = [np.round(_q, decimals=decimals) for _q in quantiles]
    #     result = dict(zip(names, out))

    #     if not hasattr(self.measures,'quantiles'):
    #         self.measures.quantiles = {}
    #     self.measures.quantiles.update(result)
    #     return result


    # ### PROPERTY METHODS ###




    # def to_obspy_pick(self, uncertainty=None):
    #     method_prefix = f'smi:local/{self.stats.model}/{self.stats.weight}'
    #     if uncertainty is None:

    #         pick = Pick(resource_id = ResourceIdentifier(prefix='smi:local/pulse/data/pick/Trigger/to_pick_simple'),
    #                     time=self.tmax,
    #                     method_id=ResourceIdentifier(id=method_prefix),
    #                     waveform_id=WaveformStreamID(
    #                         network_code=self.stats.network,
    #                         station_code=self.stats.station,
    #                         location_code=self.stats.location,
    #                         channel_code=self.stats.channel),
    #                     phase_hint=self.stats.component,
    #                     evaluation_status='automatic')
    #     else:
    #         raise NotImplementedError('Uncertainty passing methods not yet developed')
    #     return pick


    # def to_pulse_pick(self, pick_param='mean', model='gaussian', fit_quality_metric='Rsquared'):
    #     if model not in ['simple','gaussian','triangle','boxcar']:
    #         raise NotImplementedError(f'model "{model}" not supported')
    #     if fit_quality_metric not in ['Rsquared']:
    #         raise NotImplementedError(f'quality_metric "{quality_metric}" not supported')

    #     header = {_k:self.stats[_k] for _k in ['network','station','location','channel']}
    #     header.update({'detection_method': self.id_keys['mod'],
    #                    'trigger_model': model,
    #                    'ref_time': self.stats.starttime,
    #                    'ref_sampling_rate': self.stats.sampling_rate,
    #                    'pick_param': pick_param,
    #                    ''})


    #     kwargs = {'id_string': self.id,
    #               'fit_model': model}
        
    #     if model == 'simple':
    #         if param == 'max':
    #             kwargs.update({'ref_time': self.tmax,
    #                            'ref_level': self.pmax,
    #                            'ref_param': 'obs_trigger_max'})
    #         elif param == 'thr_on':
    #             kwargs.update({'ref_time': self.stats.starttime + self.samp_on*self.stats.delta,
    #                            'ref_level': self.thr_on,
    #                            'ref_param': 'obs_trigger_on'})
    #         elif param == 'thr_off':
    #             kwargs.update({'ref_time': self.stats.starttime +\
    #                              self.samp_off*self.stats.delta,
    #                            'ref_level': self.thr_off,
    #                            'ref_param': 'obs_trigger_off'})
    #     # Model fits     
    #     else:
    #         # Fit model with least-squares using seconds-united
    #         popt, pcov, res = lsq_fit_model(
    #             self.times(), self.data, model=model, p0=None
    #         )
    #         if param == 'max':
    #             kwargs.update({
    #                 'ref_time': self.stats.starttime + popt[1],
    #                 'ref_level': popt[0],
    #                 'ref_param': 'model_mean'})
    #         # Handle special case for boxcar thr_on/thr_off
    #         elif model == 'boxcar':
    #             if param == 'thr_on':
    #                 kwargs.update({'ref_time': self.stats.starttime + popt[1] - popt[0],
    #                                'ref_level': self.thr_on,
    #                                'ref_param': 'model_trigger_on'})
    #             elif param == 'thr_off':



    #         if quality_metric == 'Rsquared':
    #             fit_quality = rsquared(res, self.data)
        
    #     pick = PulsePick(**kwargs)

    #         ppick = PulsePick(
    #             id_string = self.id,
    #             label = self.id_keys['component'],
    #             ref_time = self.tmax,
    #             ref_level = self.pmax,
    #             ref_param = model,
    #             ref
    #         )
