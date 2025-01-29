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
from scipy.integrate import quad
from scipy.optimize import leastsq
from scipy.sparse import coo_array

# FUNCTIONAL MODELS OF TRIGGER GEOMETRIES #


def scaled_gaussian(p, x):
    """Functional form of a scaled Gaussian function (bellcurve)
    
    :math:`p_0 e^{\\frac{-(x - p_1)^2}{2 p_2}}`,
    with 
     - :math:`p_0` = amplitude
     - :math:`p_1` = mean / central value
     - :math:`p_2` = variance (standard deviation squared)

    This sees typical use fitting curves to ML model predictions
    of arrival time labels in PULSE workflows.

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
    """Functional form of a scaled uniform distribution

    This sees typical use fitting cureves to ML model
    predictions for noise and/or detection labels in 
    PULSE workflows

    :param p: _description_
    :type p: _type_
    :param x: _description_
    :type x: _type_
    :return: _description_
    :rtype: _type_
    """

    amp = p[0]
    onset = p[1]
    offset = p[2]
    y = np.zeros(x.shape)
    idx = (x <= onset) & (x + 1 < offset)
    y[idx] += amp
    return y


def scaled_triangle(p, x):
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



def scaled_iso_triangle(p, x):
    amp = p[0]
    center = p[1]
    halfwidth = p[2]
    # Upslope
    y1 = (x - center + halfwidth)/halfwidth
    y1 *= (y1 >= 0) & (y1 <= 1)
    # Downslope
    y2 = -1.*(x - center - halfwidth)/halfwidth
    y2 *= (y2 >= 0) & (y2 < 1)
    # Sum and scale
    y = amp*(y1 + y2)
    return y


def model_errors(p, x, y, model):
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

def rsquared(res, obs):
    ss_res = np.sum(res**2)
    ss_tot = np.sum((obs - np.mean(obs))**2)
    r2 = 1. - (ss_res / ss_tot)
    return r2

def lsq_model_fit(x_obs, y_obs, model='gaussian', p0=None):
    if np.ndim(x_obs) != 1:
        raise ValueError
    if np.ndim(y_obs) != 1:
        raise ValueError
    if x_obs.shape != y_obs.shape:
        raise ValueError
    if model not in ['gaussian','triangle','boxcar']:
        raise ValueError(f'"model" {model} not supported')
    
    # If initial parameters is None, do a reasonable first-pass estimate from the data
    if p0 is None:
        p0 = [np.nanmax(y_obs), np.nanmean(x_obs), (np.nanmax(x_obs) - np.nanmean(x_obs))/2]
    popt, pcov = leastsq(model_errors, p0, args=(x_obs, y_obs, model), full_output=True)
    res = model_errors(popt, x_obs, y_obs, model)
    return popt, pcov, res

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


#########################################################
# Methods for fitting normal pdfs to probability curves #
#########################################################

def scaled_triangle(p, x):
    """Scaled isosceles triangle function
    with parameters:
    
    p[0] = Amplitdue
    p[1] = Central coordinate ("mean")
    p[2] = Half-width

    :param p: paramter values as described above
    :type p: array-like
    :param x: sample coordinate vector
    :type x: numpy.ndarray
    :returns: **y** (*numpy.ndarray*) -- 
    """
    # Upslope
    y1 = (x - p[1] + p[2])/p[2]
    y1 *= (y1 >= 0) & (y1 <= 1)
    # Downslope
    y2 = -1.*(x - p[1] - p[2])/p[2]
    y2 *= (y2 >= 0) & (y2 < 1)
    # Sum and scale
    y = p[0]*(y1 + y2)
    return y

def scaled_box(p, x):
    """Scaled, shifted boxcar function with parameters
    p[0] = Amplitude
    p[1] = central coordinate
    p[2] = half-width

    :param p: _description_
    :type p: _type_
    :param x: _description_
    :type x: _type_
    :return: _description_
    :rtype: _type_
    """    
    y = np.zeros(x.shape)
    idx = ((x - p[1]) <= p[2]) & (-(x - p[1] + 1) < p[2])
    y[idx] += p[0]
    return y


def scaled_gaussian(p, x):
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


def gaussian_misfit(p, x, y_obs):
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
    y_cal = scaled_gaussian(p, x)
    y_err = y_obs - y_cal
    return y_err


def scaled_gaussian_misfit(p, x, y_obs):
    """Calculate the misfit of a shifted, scaled
    Gaussian PDF to observed values

    p[0] = model amplitude
    p[1] = model central location coordinate (mean)
    p[2] = model scale (variance)

    :param p: model parameter vector
    :type p: numpy.ndarray
    :param x: sample location vector
    :type x: numpy.ndarray
    :param y_obs: observed sample value vector
    :type y_obs: numpy.ndarray
    :returns: **res** (*numpy.ndarray*) -- model - data residual vector 
    """    
    y_cal = scaled_gaussian(p, x)
    res = y_cal - y_obs
    return res

def scaled_triangle_misfit(p, x, y_obs):
    """Calculate the misfit of a shifted, scaled
    isosceles triangle function to observed values

    p[0] = model amplitude
    p[1] = model central location coordinate (mean)
    p[2] = model scale (half-width)

    :param p: model parameter vector
    :type p: numpy.ndarray
    :param x: sample location vector
    :type x: numpy.ndarray
    :param y_obs: observed sample value vector
    :type y_obs: numpy.ndarray
    :returns: **res** (*numpy.ndarray*) -- model - data residual vector 
    """    
    y_cal = scaled_triangle(p, x)
    res = y_cal - y_obs
    return res

def scaled_box_misfit(p, x, y_obs):
    """Calculate the misfit of a shifted, scaled boxcar
    function to observed values

    p[0] = model amplitude
    p[1] = model central location coordinate (mean)
    p[2] = model scale (half-width)

    :param p: model parameter vector
    :type p: numpy.ndarray
    :param x: sample location vector
    :type x: numpy.ndarray
    :param y_obs: observed sample value vector
    :type y_obs: numpy.ndarray
    :returns: **res** (*numpy.ndarray*) -- model - data residual vector 
    """    
    y_cal = scaled_box(p, x)
    res = y_cal - y_obs
    return res

def fit_geometric_model(x, y, function='gaussian', min_y=0.1, mindata=30, p0=None):
    """_summary_

    :param x: _description_
    :type x: _type_
    :param y: _description_
    :type y: _type_
    :param function: _description_, defaults to 'gaussian'
    :type function: str, optional
    :param min_y: _description_, defaults to 0.1
    :type min_y: float, optional
    :param mindata: _description_, defaults to 30
    :type mindata: int, optional
    :param p0: _description_, defaults to None
    :type p0: _type_, optional
    :raises AttributeError: _description_
    :raises AttributeError: _description_
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """    
    if np.ndim(x) == 1 and np.ndim(y) == 1 and x.shape == y.shape:
        pass
    else:
        raise AttributeError('x and y must be equal length vectors')
    ind = y >= min_y
    _x = x[ind]
    _y = y[ind]
    if len(_y) < mindata:
        raise AttributeError(f'Insufficient data ({sum(ind)} < {mindata}) that meet min_y <= {min_y}')
    
    funcs = {'gaussian': scaled_gaussian_misfit,
             'normal': scaled_gaussian_misfit,
             'boxcar': scaled_box_misfit,
             'box': scaled_box_misfit,
             'triangle': scaled_triangle_misfit}

    if function in funcs.keys():
        func = funcs[function]
    else:
        raise ValueError(f'function "{function}" not supported')

    # if any(_x[1:] - _x[:-1] != 1):
    #     raise AttributeError('Non-consecutive elements present')
    if p0 is None:
        # If using gaussian
        if function in ['gaussian','normal']:
            # Estimate mean and standard deviation
            p0 = [1, np.mean(_x), 0.34*(_x[-1] - _x[0])]
        # If using triangle or box
        else:
            # Estimate mean and half-width
            p0 = [1, np.mean(_x), 0.5*(_x[-1] - _x[0])]
    # Run Least Squares inversion for model parameters and model covariance matrix
    popt, pcov = leastsq(func, p0, args=(_x, _y), full_output=True)
    # Calculate residuals for popt
    res = func(popt, _x, _y)
    return popt, pcov, res
    

def fit_gaussian(x, y, threshold=0.1, mindata=30, p0=None):
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
    pout, pcov = leastsq(gaussian_misfit, p0, args=(xv, yv), full_output=True)
    res = gaussian_misfit(pout, xv, yv)
    return pout, pcov, res





def kld_uniform(center_true, halfwidth_true, center_est, halfwidth_est):
    a = center_true - halfwidth_true
    b = center_true + halfwidth_true
    c = center_est - halfwidth_est
    d = center_est + halfwidth_est
    nd = (d - c)/(b - a)
    return np.log(nd)


def kld_triangle_integrand(x, p_true, p_est):
    p = scaled_triangle(p_true, x)
    q = scaled_triangle(p_est, x)
    if p == 0 or q == 0:
        return 0
    return p * np.log(p / q)

def kld_triangle(p_true, p_est):
    # Get integration bounds (overlap)
    lb = max(p_true[1] - p_true[2], p_est[1] - p_est[2])
    ub = min(p_true[1] + p_true[2], p_est[1] + p_est[2])
    # If no overlap, return infinity
    if lb > ub:
        return np.inf
    kld, _ = quad(kld_triangle_integrand, lb, ub, args=tuple(list(p_true) + list(p_est)))




def kld_gaussian(mean_true, var_true, mean_est, var_est):
    """
    Calculate the Kullbeck-Leibler divergence score of
    an estimated (modeled) Gaussian distribution relative
    to the true (reference) Gaussian distribution 

    :math:`\\frac{1}{2} \\left( \\frac{\\sigma_0^2}{\\sigma_1^2} + \\frac{(\mu_1 - \mu_0)^2}{\\sigma_1^2} - 1 ln \\left(\\frac{\\sigma_1^2}{\\sigma_0^2}\\right) \\right)`

    with 
     - :math:`\\mu_0` the true mean
     - :math:`\\mu_1` the estimated mean
     - :math:`\\sigma_0^2` the true distribution variance
     - :math:`\\sigma_1^2` the estimated distribution variance

    Thus the KL divergence approaches 0 as the distributions become
    more similar and is unbounded as they become more distinct.

    also see: https://en.wikipedia.org/wiki/Kullback–Leibler_divergence

    :param mean_true: reference distribution mean
    :type mean_true: float-like
    :param var_true: reference distribution variance
    :type var_true: float-like
    :param mean_est: estimated distribution mean
    :type mean_est: float-like
    :param var_est: estimated distribution variance
    :type 

    """
    p0 = var_true/var_est
    p1 = ((mean_est - mean_true)**2)/var_est
    p2 = var_est/var_true
    kld = 0.5*(p0 + p1 - 1 + np.log(p2))
    return kld

def total_variation_distance(kld_score):
    """
    Calculate a variation distance using the formulation from Bretagnolle and Huber (1978)
    for a Kullbak-Leiber divergence score (:math:`D_{KL}`)

    :math:`\\delta = \\sqrt{1 - e^{-D_{KL}}}`

    Provides a representation of divergence that has :math:`\\delta \\in (0, 1]`
    with increasing divergence approaching 0 and exact similarity returning 1.

    Reference: https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Interpretations
    """
    kld = kld_score
    d = np.sqrt(1. - np.exp(-kld))
    return d


def kld_score_reciprocal(kld_score):
    """
    Reciprocal representation of the Kullbak-Leiber divergence score (:math:`D_{KL}`)
    such that scores for similar distributions approach 1 and scores for dissimilar
    distributions approach 0, given as:

    :math:`R_{KL} = \\frac{1}{D_{KL} + 1}

    Note: This is an ad-hoc reformulation of the Kullbak-Leiber divergence score
    such that :math:`R_{KL} \in (0, 1]`. A more appropriate metric might be the
    total-variation distance provided with :meth:`~.total_variation_distance`

    """

    kld = kld_score
    kldr = 1. / (kld + 1.)
    return kldr


def kl_div_pairwise(means, vars):
    """Calculate pairwise Kullbak-Leiber divergence scores for 
    a set of normal distributions defined by vectorts of mean
    and variance pairs

    :param means: vector of distribution means
    :type means: numpy.ndarray
    :param vars: vector of distribution variances
    :type vars: numpy.ndarray
    :returns: **output** (*numpy.ndarray*) - symmetric matrix
        of Kullbak-Leiber divergence scores
    """    
    if len(means.shape) != 1:
        raise AttributeError
    if len(vars.shape) != 1:
        raise AttributeError
    if len(means) != len(vars):
        raise ValueError

    output = np.full(shape=(len(means), len(means)), fill_value=np.nan)
    for ii, imu in enumerate(means):
        for jj, jmu in enumerate(means):
            if ii < jj:
                _kld = kld_score(imu, vars[ii], jmu, vars[jj])
                output[ii, jj] = _kld
                output[jj, ii] = _kld
            elif ii == jj:
                output[ii, jj] = 0
    return output

def two_sample_z_test(mean0, var0, mean1, var1):
    num = np.abs(mean0 - mean1)
    den = np.sqrt(var0 + var1)
    return num/den

def kl_div_sparse(means, vars, max_mean_difference=3.):
    """Calculate pairwise Kullbak-Leiber divergence scores for 
    a set of normal distributions defined by vectorts of mean
    and variance pairs where the maximum difference between
    mean pairs does not exceed an arbitrary threshold value.

    :param means: vector of distribution means
    :type means: numpy.ndarray
    :param vars: vector of distribution variances
    :type vars: numpy.ndarray
    :returns: **output** (*scipy.sparse.coo_array*) -- sparse array
        of Kullbak-Leiber divergence scores for distribution pairs
        that meet the threshold criteria
    """    
    if len(means) != len(vars):
        raise ValueError
    data = []
    row = []
    col = []
    ij_max = 0
    for ii, imu in enumerate(means):
        for jj, jmu in enumerate(means):
            if ii < jj:
                if np.abs(imu - jmu) <= max_mean_difference:
                    # Do upper triangle
                    _dat = kld_score(imu, vars[ii], jmu, vars[jj])
                    data.append(_dat)
                    row.append(ii)
                    col.append(jj)
                    # And lower triangle
                    data.append(_dat)
                    row.append(jj)
                    col.append(ii)
                    if jj > ij_max:
                        ij_max = jj
                    
    # Convert into a COOrdinate Sparse Array
    output = coo_array(data, (row, col), shape=(ij_max, ij_max))
    return output





# class GaussianModel(object):
#     """An object hosting paramter fitting of a scaled gaussian model:

#     .. math::
#         y = p_0 e^{\\frac{(x - p_1)^2}{2p_2^2}}

#     to input data {x, y}, with model parameters:
#         - :math:`p_0 = A_{calc}`: prefactor, or amplitude
#         - :math:`p_1 = \\mu_{calc}`: central value, or mean
#         - :math:`p_2 = \\sigma_{calc}`: standard deviation

#     The model also includes the paramter covariance matrix and model-data residuals.
     
#     Also calculates and holds empirical estimates of the statistical moments of input data {x, y},
#         - :math:`\\mu_{est}` -- central value or mean
#         - :math:`\\sigma_{est}` -- standard deviation
#         - :math:`\\mathcal{S}_{est}` -- skewness
#         - :math:`\\mathcal{K}_{est}` -- kurtosis (Pearson or Fisher)

#     :param kurt_type: type of kurtosis to use "Pearson" or "Fisher" (Pearson = Fisher + 3), defaults to Pearson
#     :type kurt_type: str, optional
#     :param dtype: specify alternative data type to use for all calculations, defaults to None
#     :type dtype: None or numpy.float32-like, optional
#     :assigns:
#         - **self.kurt_type** (*str*) -- 'pearson' or 'fisher' assigned
#         - **self.dtype** (*type* or *None*) -- type or none assigned, passed to :meth:`~ayahos.core.trigger.GaussianModel.estimate_moments`
#     """    
#     def __init__(self, kurt_type='Pearson', dtype=None):
#         """Initialize a GaussianModel object

#         :param kurt_type: type of kurtosis to use "Pearson" or "Fisher" (Pearson = Fisher + 3), defaults to Pearson
#         :type kurt_type: str, optional
#         :param dtype: specify alternative data type to use for all calculations, defaults to None
#         :type dtype: None or numpy.float32-like, optional
#         :assigns:
#             - **self.kurt_type** (*str*) -- 'pearson' or 'fisher' assigned
#             - **self.dtype** (*type* or *None*) -- type or none assigned, passed to :meth:`~ayahos.core.trigger.GaussianModel.estimate_moments`
    
#         """        
#         if kurt_type.lower() in ['fisher','pearson']:
#             self.kurt_type = kurt_type.lower()
#         else:
#             raise ValueError
#         if dtype is None:
#             self.dtype = dtype
#         else:
#             try:
#                 np.ones(shape=(30,), dtype=dtype)
#                 self.dtype = dtype
#             except TypeError:
#                 raise TypeError
#         self.p = None
#         self.cov = None
#         self.res = None
#         self.est_mean = None
#         self.est_stdev = None
#         self.est_skew = None
#         self.est_kurt = None


#     def fit_pdf_to_curve(self, x, y, threshold=0.1, mindata=30, p0=None):
#         """Fit a scaled gaussian probability density function to the curve y=f(x) using
#         the least squares fitting in :meth:`~ayahos.util.stats.fit_normal_pdf_curve`.

#         :param x: independent variable values
#         :type x: numpy.ndarray
#         :param y: dependent variable values
#         :type y: numpy.ndarray
#         :param threshold: minimum y value to use for fitting, defaults to 0.1
#         :type threshold: float, optional
#         :param mindata: minimum number of datapoints required for a fitting, defaults to 30
#         :type mindata: int, optional
#         :param p0: initial parameter estimates [Amp, mean, stdev], defaults to None
#         :type p0: 3-tuple-like or None, optional

#         :updates:
#             - **self.p** (*numpy.ndarray*) -- least squares best fit model parameters
#             - **self.cov** (*numpy.ndarray*) -- model parameter covariance matrix
#             - **self.res** (*numpy.ndarray*) -- model-data (:math:`y_{calc} - y`) residuals
        
#         """        
#         outs = fit_normal_pdf_curve(x,y,threshold=threshold, mindata=mindata, p0=p0)
#         self.p = outs[0]
#         self.cov = outs[1]
#         self.res = outs[2]
#         if self.dtype is not None:
#             self.p = self.dtype(self.p)
#             self.cov = self.dtype(self.cov)
#             self.res = self.dtype(self.res)

#     def estimate_moments(self, x, y):
#         """Estimate first through fourth moments of the probability density function
#         y = f(x)

#         :param x: independent variable values
#         :type x: numpy.ndarray
#         :param y: dependent variable values
#         :type y: numpy.ndarray
#         :updates:
#             - **self.est_mean** (*float*) -- central value / mean
#             - **self.est_stdev** (*float*) -- standard deviation
#             - **self.est_skew** (*float*) -- skewness
#             - **self.est_kurt** (*float*) -- kurtosis (corresponding to **self.kurt_type**)
#         """        
#         if self.kurt_type == 'fisher':
#             fisher = True
#         else:
#             fisher = False
#         outs = estimate_moments(x, y, fisher=fisher, dtype=self.dtype)
#         self.est_mean = outs[0]
#         self.est_stdev = outs[1]
#         self.est_skew = outs[2]
#         self.est_kurt = outs[3]

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
