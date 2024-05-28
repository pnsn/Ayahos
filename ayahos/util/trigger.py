import numpy as np
from obspy.signal.trigger import trigger_onset
from pandas import DataFrame

def expandable_trigger(pred_trace, pthr=0.2, ethr=0.01, ndata_bounds=[15, 9e99], oob_delete=True):
    charfct = pred_trace.data
    # Based on obspy.signal.trigger.trigger_onset
    t_main = trigger_onset(
        charfct,
        pthr,
        pthr,
        max_length=max(ndata_bounds),
        max_len_delete=oob_delete)
    t_exp = trigger_onset(
        charfct,
        ethr,
        ethr,
        max_length=max(ndata_bounds),
        max_len_delete=oob_delete)
    
    passing_triggers = []
    # Iterate across triggers
    for mtrig in t_main:
        mi0 = mtrig[0]
        mi1 = mtrig[1]
        for etrig in t_exp:
            ei0 = etrig[0]
            ei1 = etrig[1]
            # If expanded trigger is, or contains main trigger
            if ei0 <= mi0 < mi1 <= ei1:
                # If delete bool is True and expanded trigger is too small, pass
                if oob_delete and ei1 - ei0 < min(ndata_bounds):
                    pass
                # in all other cases, append
                else:
                    passing_triggers.append(etrig)
    passing_triggers = np.array(passing_triggers, dtype=np.int64)
    return passing_triggers
    
def triggers_to_time(triggers, t0, dt):
    times = np.full(shape=triggers.shape, fill_value=t0)
    times += triggers*dt
    return times



def process_est_prediction_stats(
    prediction_trace,
    thr=0.1,
    extra_quantiles=[0.05, 0.2, 0.3, 0.7, 0.8, 0.95],
    pad_sec=0.05,
    ndata_bounds=[15, 9e99],
):
    """
    Run triggering with a uniform threshold on prediction traces and extract a set of gaussian and quantile
    statistical representations of prediction probability peaks that exceed the trigger threshold

    :: INPUTS ::
    :param prediction_trace:    [obspy.core.trace.Trace]
        Trace containing phase onset prediction probability timeseries data
    :param thr:              [float] trigger-ON/-OFF threshold value
    :param pad_sec:             [float]
        amount of padding on either side of data bounded by trigger ON/OFF
        times for for including additional, marginal population samples
        for estimating gaussian and quantile statistics
    :param extra_quantiles: [list of float]
        Additional quantiles to assess beyond Q1 (q = .25), Q3 (q = .75), and median (q = .5)
    :param ndata_bounds:    [2-tuple of int]
        minimum & maximum count of data for each trigger window
    :param quantiles:       [list of float]
        quantile values to assess within a trigger window under assumptions
        stated in documentation of est_curve_quantiles()
    :: OUTPUT ::
    :return df_out:     [pandas.dataframe.DataFrame]
        DataFrame containing the following metrics for each trigger:
            'et_on'     Epoch ON trigger time
            'et_off'    Epoch OFF trigger time
            'et_max'    Epoch max probability time
            'p_max'     Max probability value
            'et_mean'   Epoch mean probability time
            'p_mean'    Mean probability value
            'dt_std'    Delta time standard deviation [seconds]
            'skew'      Estimated skewness of probability distribution
            'kurt'      Estimated kurtosis of probability distribution
            'pdata'     Number of data used for statistical measures
            'et_med'    Epoch median probability time
            'p_med'     Median probability value
            'dt_q1'     Delta time for 1st Quartile (0.25 quantile)
                            relative to et_med: et_q1 - et_med
            'dt_q3'     Delta time for 3rd Quartile (0.75 quantile)
                            relative to et_med: et_q3 - et_med
            f'dt_q{extra_quantiles:.2f}'
                        Delta time(s) for extra_quantiles
                            relative to et_med: et_q{} - et_med
    """
    # create dictionary holder for triggers


    # Define default statistics for each trigger
    cols = [
        "et_on",
        "et_off",
        "et_max",
        "p_max",
        "et_mean",
        "p_mean",
        "dt_std",
        "skew",
        "kurt",
        "pdata",
        "et_med",
        "p_med",
        "dt_q1",
        "dt_q3",
    ]
    # Define default quantiles
    quants = [0.025, 0.159, 0.5, 0.84, 0.975]
    # Get epoch time vector from trace
    times = prediction_trace.times(type="timestamp")
    preds = prediction_trace.data
    # Get pick indices with Obspy builtin method
    triggers = trigger_onset(
        preds,
        thr,
        thr,
        max_len=ndata_bounds[1],
        max_len_delete=True,
    )
    # Append extra_quantiles
    if isinstance(extra_quantiles, float):
        quants += [extra_quantiles]
        cols += [f"dt_q{extra_quantiles:.2f}"]
    elif isinstance(extra_quantiles, list):
        quants += extra_quantiles
        cols += [f"dt_q{_eq:.2f}" for _eq in extra_quantiles]

    # Iterate across triggers to extract statistics
    holder = []
    for _trigger in triggers:
        # Get trigger ON and OFF timestamps
        _t0 = times[_trigger[0]]
        _t1 = times[_trigger[1]]
        # Get windowed data for statistical estimation
        ind = (times >= _t0 - pad_sec) & (times <= _t1 + pad_sec)
        _times = times[ind]
        _preds = preds[ind]
        # Run gaussian statistics
        et_mean, dt_std, skew, kurt = est_curve_normal_stats(_times, _preds)
        p_mean = _preds[np.argmin(np.abs(et_mean - _times))]
        # Run quantiles
        et_q, p_q = est_curve_quantiles(_times, _preds, q=quants)
        line = [
            _t0,
            _t1,
            _times[np.argmax(_preds)],
            np.max(_preds),
            et_mean,
            p_mean,
            dt_std,
            skew,
            kurt,
            len(_times),
            et_q[0],
            p_q[0],
            et_q[1] - et_q[0],
            et_q[2] - et_q[0],
        ]
        # Add extra quantiles if provided
        if len(et_q) > 3:
            line += [_et - et_q[0] for _et in et_q[3:]]
        # Append trigger statistics to holder
        holder.append(line)
    try:
        df_out = DataFrame(holder, columns=cols)
    except:
        breakpoint()

    return df_out


def process_predictions(
    prediction_trace,
    et_obs=None,
    thr_on=0.1,
    thr_off=0.1,
    fit_pad_sec=0.1,
    fit_thr_coef=0.1,
    ndata_bounds=[30, 9e99],
    quantiles=[0.25, 0.5, 0.75],
):
    """Extract statistical fits of normal distributions to prediction peaks from
    ML prediction traces that trigger above a specified threshold.

    :: INPUTS ::
    :param prediction_trace:    [obspy.core.trace.Trace]
        Trace containing phase onset prediction probability timeseries data
    :param et_obs:              [None or list of epoch times]
        Observed pick times in epoch time (timestamps) associated with the
        station/phase-type for `prediction_trace`
    :param thr_on:              [float] trigger-ON threshold value
    :param thr_off:             [float] trigger-OFF threshold value
    :param fit_pad_sec:         [float]
        amount of padding on either side of data bounded by trigger ON/OFF
        times for calculating Gaussian fits to the probability peak(s)
    :param fit_thr_coef:    [float] Gaussian fit data
    :param ndata_bounds:    [2-tuple of int]
        minimum & maximum count of data for each trigger window
    :param quantiles:       [list of float]
        quantile values to assess within a trigger window under assumptions
        stated in documentation of est_curve_quantiles()
    :: OUTPUT ::
    :return df_out:     [pandas.dataframe.DataFrame]
        DataFrame containing the following metrics for each trigger
        and observed pick:
        'et_on'     - Trigger onset time [epoch]
        'et_off'    - Trigger termination time [epoch]
        'p_scale'   - Probability scale from Gaussian fit model \in [0,1]
        'q_scale'   - Probability value at the estimated median (q = 0.5)
        'm_scale'   - Maximum estimated probability value
        'et_mean'   - Expectation peak time from Gaussian fit model [epoch]
        'et_max'    - timestamp of the maximum probability [epoch]
        'det_obs_prob' - delta time [seconds] of observed et_obs[i] - et_max
                            Note: this will be np.nan if there are no picks in
                                  the trigger window
        'et_std'    - Standard deviation of Gaussian fit model [seconds]
        'L2 res'    - L2 norm of data - model residuals for Gaussian fit
        'ndata'     - number of data considered in the Gaussian model fit
        'C_pp'      - variance of model fit for p_scale
        'C_uu'      - variance of model fit for expectation peak time
        'C_oo'      - variance of model fit for standard deviation
        'C_pu'      - covariance of model fit for p & u
        'C_po'      - covariance of model fit for p & o
        'C_uo'      - covariance of model fit for u & o
    """
    # Define output column names
    cols = [
        "et_on",
        "et_off",
        "p_scale",
        "q_scale",
        "m_scale",
        "et_mean",
        "et_med",
        "et_max",
        "det_obs_prob",
        "et_std",
        "L2 res",
        "ndata",
        "C_pp",
        "C_uu",
        "C_oo",
        "C_pu",
        "C_po",
        "C_uo",
    ]
    # Ensure median is included in quantiles
    quantiles = list(quantiles)
    med_ind = None
    for _i, _q in enumerate(quantiles):
        if _q == 0.5:
            med_ind = _i
    if med_ind is None:
        quantiles.append(0.5)
        med_ind = -1

    cols += [f"q{_q:.2f}" for _q in quantiles]
    # Get pick indices with Obspy builtin method
    triggers = trigger_onset(
        prediction_trace.data,
        thr_on,
        thr_off,
        max_len=ndata_bounds[1],
        max_len_delete=True,
    )
    times = prediction_trace.times(type="timestamp")
    # Iterate across triggers:
    feature_holder = []
    for _trigger in triggers:
        _t0 = times[_trigger[0]]
        _t1 = times[_trigger[1]]
        # If there are observed time picks provided, search for picks
        wind_obs = []
        if isinstance(et_obs, list):
            for _obs in et_obs:
                if _t0 <= _obs <= _t1:
                    wind_obs.append(_obs)
        _tr = prediction_trace.copy().trim(
            starttime=UTCDateTime(_t0) - fit_pad_sec,
            endtime=UTCDateTime(_t1) + fit_pad_sec,
        )
        # Conduct gaussian fit
        outs = fit_probability_peak(
            _tr, fit_thr_coef=fit_thr_coef, mindata=ndata_bounds[0]
        )
        # Get timestamp of maximum observed data
        et_max = _tr.times(type="timestamp")[np.argmax(_tr.data)]

        # Get times of quantiles:
        qet, qmed, q = est_curve_quantiles(
            _tr.times(type="timestamp"), _tr.data, q=quantiles
        )

        # Iterate across observed times, if provided
        # First handle the null
        if len(wind_obs) == 0:
            _det_obs_prob = np.nan
            feature_line = [
                _t0,
                _t1,
                outs[0],
                outs[1],
                et_max,
                _det_obs_prob,
                outs[2],
                outs[4],
                outs[5],
                outs[3][0, 0],
                outs[3][1, 1],
                outs[3][2, 2],
                outs[3][0, 1],
                outs[3][0, 2],
                outs[3][1, 2],
            ]
            if quantiles:
                feature_line += list(qet)
            feature_holder.append(feature_line)
        # Otherwise produce one line with each delta time calculation
        elif len(wind_obs) > 0:
            for _wo in wind_obs:
                _det_obs_prob = _wo - et_max
                feature_line = [
                    _t0,
                    _t1,
                    outs[0],
                    outs[1],
                    et_max,
                    _det_obs_prob,
                    outs[2],
                    outs[4],
                    outs[5],
                    outs[3][0, 0],
                    outs[3][1, 1],
                    outs[3][2, 2],
                    outs[3][0, 1],
                    outs[3][0, 2],
                    outs[3][1, 2],
                ]
                if quantiles:
                    feature_line += list(qouts)

                feature_holder.append(feature_line)

    df_out = DataFrame(feature_holder, columns=cols)
    return df_out