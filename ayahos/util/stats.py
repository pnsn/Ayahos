"""
:module: wyrm.util.stats
:auth: Nathan T Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose: Provides supporting methods for fitting probability density functions to data curves
    {x, y}, measuring statistical moments, calculating residuals, and estimating quantiles to 
    provide compressed representations 
"""
import numpy as np
from obspy import UTCDateTime
from obspy.signal.trigger import trigger_onset
from pandas import DataFrame
from scipy.optimize import leastsq

##########################################
# Methods for estimating statistics from #
# y = f(x) representations of histograms #
##########################################


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
    # if 0.5 not in q:
    #     q.append(0.5)
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






#########################################################
# Methods for fitting normal pdfs to probability curves #
#########################################################

def scaled_normal_pdf(p, x):
    """
    Model a scaled normal distribution (Gaussian) with parameters:  
    p[0] = :math:`A`       - Amplitude of the distribution  
    p[1] = :math:`\\mu`    - Mean of the distribution  
    p[2] = :math:`\\sigma^`- Variance of the distribution  

    .. math::
        y = A e^{-\\frac{(x - \\mu)^2}{2\sigma^2}}

    for sample locations x

    :param p: model parameter vector
    :type p: array-like
    :param x: sample points
    :type x: numpy.ndarray
    :return y: modeled probability values
    :rtype y: numpy.ndarray
    """
    y = p[0] * np.exp((-1.*(x - p[1])**2)/(2*p[2]))
    return y


def normal_pdf_error(p, x, y_obs):
    """
    Calculate the misfit between an offset normal distribution (Gaussian)
    with parameters:
    :math:`p_0 = A`         - Amplitude of the distribution
    :math:`p_1 = \\mu`      - Mean of the distribution
    :math:`p_2 = \\sigma^2` - Variance of the distribution

    and X, y_obs data that may

    :: INPUTS ::
    :param p: Model parameters
    :type p: numpy.ndarray
    :param x: [array-like] independent variable, sample locations
    :param y_obs: [array-like] dependent variable at sample locations

    :: OUTPUT ::
    :return y_err: [array-like] misfit calculated as y_obs - y_cal
    """
    # Calculate the modeled y-values given positions x and parameters p[:]
    y_cal = scaled_normal_pdf(p, x)
    y_err = y_obs - y_cal
    return y_err


def fit_normal_pdf_curve(x, y, threshold=0.1, mindata=30, p0=None):
    """Fit a Gaussian (Normal) distribution to the PDF approximated as
    y = PDF(x) using data with values :math:`y>=threshold` and 
    :meth:`~scipy.optimize.leastsq` with parameter values :math:`p = {A, \\mu, \\sigma}`
    such that

    .. math::
        y = A e^{\\frac{-(x - \\mu)^2}{2\\sigma^2}}

    :param x: sample locations
    :type x: numpy.ndarray
    :param y: probability density values
    :type y: np.ndarray
    :param threshold: minimum value of y to use for curve fitting, defaults to 0.1
    :type threshold: float, optional
    :param mindata: minimum number of datapoints required for fitting, defaults to 30
    :type mindata: int, optional
    :param p0: initial parameter value guesses, defaults to None
    :type p0: array-like, optional
    :return: 
        - **pout** (*numpy.ndarray*) -- best-fit parameter values :math:`A, \\mu, \\sigma^2`  
        - **pcov** (*numpy.ndarray*) -- best-fit parameter covariance matrix  
        - **res** (*numpy.ndarray*) -- model-data residual vector  
    """    
    if np.ndim(x) == 1 and np.ndim(y) == 1 and x.shape == y.shape:
        pass
    else:
        raise ValueError('dimensions of x and y are non-1-dimensional and/or unequal')
    
    if any(y_ >= threshold for y_ in y):
        pass
    else:
        raise ValueError('thereshold is too high for y values provided')

    ind = y >= threshold
    if sum(ind) >= mindata:
        pass
    else:
        raise ValueError('thresholded y results in too few datapoints')
    
    xv = x[ind]
    yv = y[ind]
    if p0 is None:
        p0 = [np.nanmax(yv), np.nanmean(xv), 0.34*(np.nanmax(xv) - np.nanmin(xv))]
    pout, pcov = leastsq(normal_pdf_error, p0, args=(xv, yv), full_output=True)
    res = normal_pdf_error(pout, xv, yv)
    return pout, pcov, res

# # SIMPLE GAUSSIAN MIXTURE MODEL - work in progress #

# def simple_gmm(xy_pairs, threshold, p0_sets=None, **options):
#     """Using a set of {x,y} pairs, create a 1-D Gaussian Mixture Model
    

#     :param xy_pairs: _description_
#     :type xy_pairs: _type_
#     :param threshold: _description_
#     :type threshold: _type_
#     :param p0_sets: _description_, defaults to None
#     :type p0_sets: _type_, optional
#     :return: _description_
#     :rtype: _type_
#     """
#     output = []
#     for i_, (x_, y_) in enumerate(xy_pairs):
#         if p0_sets is not None:
#             p0 = p0_sets[i_]
#         else:
#             p0 = None
#         output.append(fit_normal_pdf_curve(x_, y_, threshold=threshold, p0=p0, **options))
#     return output



# TODO: METHODS FOR FITTING LÉVY STABLE DISTRIBUTIONS







# GRAVEYARD #

# def calc_overlap_coefficient(mean1, mean2, stdev1, stdev2):
#     """Calculate the overlap coefficient (:math:`\mathcal{OC}`) of two normal distributions (:math:`N(\mu_i, \sigma_i)`)

#     .. math::
#         \mathcal{OC} = \\frac{|\mu_1 - \mu_2|}{\sqrt{\sigma^2_1 + \sigma^2_2}}
    
#     :param mean1: mean (center) of the first distribution
#     :type mean1: float or numpy.ndarray
#     :param mean2: mean (center) of the second distribution
#     :type mean2: float or numpy.ndarray
#     :param stdev1: standard deviation (scale) of the first distribution
#     :type stdev1: float or numpy.ndarray
#     :param stdev2: standard deviation (scale) of the first distribution
#     :type stdev2: float or numpy.ndarray
#     """
#     num = mean1 - mean2
#     den = np.sqrt(stdev1**2 + stdev2**2)

# def fit_probability_peak(x, y, fit_thr_coef=0.1, mindata=30, p0=None):
#     """
#     Fit a normal distribution to a discretized probability density function represented by :math:`y = pdf(x)` using
#     a least-squares fitting with :meth:`~scipy.optimize.leastsq`
        
#     :param prediction_trace: [obspy.core.trace.Trace]
#                                 Trace object windowed to contain a single prediction peak
#                                 with relevant metadata
#     :param obs_utcdatetime:  [datetime.datetime] or [None]
#                                 Optional reference datetime to compare maximum probability
#                                 timing for calculating delta_t. This generally is used
#                                 for an analyst pick time.
#     :param treshold_coef:    [float]
#                                 Threshold scaling value for the maximum data value used to
#                                 isolating samples for fitting the normal distribution
#     :param mindata:           [int]
#                                 Minimum number of data requred for extracting features
#     :param p0:                [array-like]
#                                 Initial normal distribution fit values
#                                 Default is None, which assumes
#                                 - amplitude = nanmax(data),
#                                 - mean = mean(epoch_times where data >= threshold)
#                                 - std = 0.25*domain(epoch_times where data >= threshold)

#     :: OUTPUTS ::
#     :return amp:            [float] amplitude of the model distribution
#                                 IF ndata < mindata --> this is the maximum value observed
#     :return mean:           [float] mean of the model distribution in epoch time
#                                 IF ndata < mindata --> this is the timestamp of the maximum observed value
#     :return std:            [float] standard deviation of the model distribution in seconds
#                                 IF ndata < mindata --> np.nan
#     :return cov:            [numpy.ndarray] 3,3 covariance matrix for <amp, mean, std>
#                                 IF ndata < mindata --> np.ones(3,3)*np.nan
#     :return err:            [float] L-2 norm of data-model residuals
#                                 IF ndata < mindata --> np.nan
#     :return ndata:          [int] number of data used for model fitting
#                                 IF ndata < mindata --> ndata
#     """
    
#     # Get data
#     data = prediction_trace.data
#     # Get thresholded index
#     ind = data >= fit_thr_coef * np.nanmax(data)
#     # Get epoch times of data
#     d_epoch = prediction_trace.times(type="timestamp")
#     # Ensure there are enough data for normal distribution fitting
#     if sum(ind) >= mindata:
#         x_vals = d_epoch[ind]
#         y_vals = data[ind]
#         # If no initial parameter values are provided by user, use default formula
#         if p0 is None:
#             p0 = [
#                 np.nanmax(y_vals),
#                 np.nanmean(x_vals),
#                 0.25 * (np.nanmax(x_vals) - np.nanmin(x_vals)),
#             ]
#         outs = leastsq(normal_pdf_error, p0, args=(x_vals, y_vals), full_output=True)
#         amp, mean, std = outs[0]
#         cov = outs[1]
#         err = np.linalg.norm(normal_pdf_error(outs[0], x_vals, y_vals))

#         return amp, mean, std, cov, err, sum(ind)

#     else:
#         return (
#             np.nanmax(data),
#             float(d_epoch[np.argwhere(data == np.nanmax(data))]),
#             np.nan,
#             np.ones((3, 3)) * np.nan,
#             np.nan,
#             sum(ind),
#         )


# def process_predictions(
#     prediction_trace,
#     et_obs=None,
#     thr_on=0.1,
#     thr_off=0.1,
#     fit_pad_sec=0.1,
#     fit_thr_coef=0.1,
#     ndata_bounds=[30, 9e99],
#     quantiles=[0.25, 0.5, 0.75],
# ):
#     """Extract statistical fits of normal distributions to prediction peaks from
#     ML prediction traces that trigger above a specified threshold.

#     :: INPUTS ::
#     :param prediction_trace:    [obspy.core.trace.Trace]
#         Trace containing phase onset prediction probability timeseries data
#     :param et_obs:              [None or list of epoch times]
#         Observed pick times in epoch time (timestamps) associated with the
#         station/phase-type for `prediction_trace`
#     :param thr_on:              [float] trigger-ON threshold value
#     :param thr_off:             [float] trigger-OFF threshold value
#     :param fit_pad_sec:         [float]
#         amount of padding on either side of data bounded by trigger ON/OFF
#         times for calculating Gaussian fits to the probability peak(s)
#     :param fit_thr_coef:    [float] Gaussian fit data
#     :param ndata_bounds:    [2-tuple of int]
#         minimum & maximum count of data for each trigger window
#     :param quantiles:       [list of float]
#         quantile values to assess within a trigger window under assumptions
#         stated in documentation of est_curve_quantiles()
#     :: OUTPUT ::
#     :return df_out:     [pandas.dataframe.DataFrame]
#         DataFrame containing the following metrics for each trigger
#         and observed pick:
#         'et_on'     - Trigger onset time [epoch]
#         'et_off'    - Trigger termination time [epoch]
#         'p_scale'   - Probability scale from Gaussian fit model \in [0,1]
#         'q_scale'   - Probability value at the estimated median (q = 0.5)
#         'm_scale'   - Maximum estimated probability value
#         'et_mean'   - Expectation peak time from Gaussian fit model [epoch]
#         'et_max'    - timestamp of the maximum probability [epoch]
#         'det_obs_prob' - delta time [seconds] of observed et_obs[i] - et_max
#                             Note: this will be np.nan if there are no picks in
#                                   the trigger window
#         'et_std'    - Standard deviation of Gaussian fit model [seconds]
#         'L2 res'    - L2 norm of data - model residuals for Gaussian fit
#         'ndata'     - number of data considered in the Gaussian model fit
#         'C_pp'      - variance of model fit for p_scale
#         'C_uu'      - variance of model fit for expectation peak time
#         'C_oo'      - variance of model fit for standard deviation
#         'C_pu'      - covariance of model fit for p & u
#         'C_po'      - covariance of model fit for p & o
#         'C_uo'      - covariance of model fit for u & o
#     """
#     # Define output column names
#     cols = [
#         "et_on",
#         "et_off",
#         "p_scale",
#         "q_scale",
#         "m_scale",
#         "et_mean",
#         "et_med",
#         "et_max",
#         "det_obs_prob",
#         "et_std",
#         "L2 res",
#         "ndata",
#         "C_pp",
#         "C_uu",
#         "C_oo",
#         "C_pu",
#         "C_po",
#         "C_uo",
#     ]
#     # Ensure median is included in quantiles
#     quantiles = list(quantiles)
#     med_ind = None
#     for _i, _q in enumerate(quantiles):
#         if _q == 0.5:
#             med_ind = _i
#     if med_ind is None:
#         quantiles.append(0.5)
#         med_ind = -1

#     cols += [f"q{_q:.2f}" for _q in quantiles]
#     # Get pick indices with Obspy builtin method
#     triggers = trigger_onset(
#         prediction_trace.data,
#         thr_on,
#         thr_off,
#         max_len=ndata_bounds[1],
#         max_len_delete=True,
#     )
#     times = prediction_trace.times(type="timestamp")
#     # Iterate across triggers:
#     feature_holder = []
#     for _trigger in triggers:
#         _t0 = times[_trigger[0]]
#         _t1 = times[_trigger[1]]
#         # If there are observed time picks provided, search for picks
#         wind_obs = []
#         if isinstance(et_obs, list):
#             for _obs in et_obs:
#                 if _t0 <= _obs <= _t1:
#                     wind_obs.append(_obs)
#         _tr = prediction_trace.copy().trim(
#             starttime=UTCDateTime(_t0) - fit_pad_sec,
#             endtime=UTCDateTime(_t1) + fit_pad_sec,
#         )
#         # Conduct gaussian fit
#         outs = fit_probability_peak(
#             _tr, fit_thr_coef=fit_thr_coef, mindata=ndata_bounds[0]
#         )
#         # Get timestamp of maximum observed data
#         et_max = _tr.times(type="timestamp")[np.argmax(_tr.data)]

#         # Get times of quantiles:
#         qet, qmed, q = est_curve_quantiles(
#             _tr.times(type="timestamp"), _tr.data, q=quantiles
#         )

#         # Iterate across observed times, if provided
#         # First handle the null
#         if len(wind_obs) == 0:
#             _det_obs_prob = np.nan
#             feature_line = [
#                 _t0,
#                 _t1,
#                 outs[0],
#                 outs[1],
#                 et_max,
#                 _det_obs_prob,
#                 outs[2],
#                 outs[4],
#                 outs[5],
#                 outs[3][0, 0],
#                 outs[3][1, 1],
#                 outs[3][2, 2],
#                 outs[3][0, 1],
#                 outs[3][0, 2],
#                 outs[3][1, 2],
#             ]
#             if quantiles:
#                 feature_line += list(qet)
#             feature_holder.append(feature_line)
#         # Otherwise produce one line with each delta time calculation
#         elif len(wind_obs) > 0:
#             for _wo in wind_obs:
#                 _det_obs_prob = _wo - et_max
#                 feature_line = [
#                     _t0,
#                     _t1,
#                     outs[0],
#                     outs[1],
#                     et_max,
#                     _det_obs_prob,
#                     outs[2],
#                     outs[4],
#                     outs[5],
#                     outs[3][0, 0],
#                     outs[3][1, 1],
#                     outs[3][2, 2],
#                     outs[3][0, 1],
#                     outs[3][0, 2],
#                     outs[3][1, 2],
#                 ]
#                 if quantiles:
#                     feature_line += list(qouts)

#                 feature_holder.append(feature_line)

#     df_out = DataFrame(feature_holder, columns=cols)
#     return df_out
