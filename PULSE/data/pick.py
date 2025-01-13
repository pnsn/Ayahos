"""
:module: PULSE.data.pick
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This contains class definitions for Trigger data objects used
    to compactly describe characteristic response function triggers

"""
import logging
from PULSE.util.stats import estimate_moments, fit_normal_pdf_curve, estimate_quantiles
import numpy as np
import pandas as pd

from obspy import Trace, UTCDateTime
from obspy.core.util.attribdict import AttribDict
from PULSE.data.foldtrace import FoldTrace
from scipy.cluster.vq import *

Logger = logging.getLogger(__name__)

from obspy.core.util.attribdict import AttribDict

class Trigger(FoldTrace):

    def __init__(self, source_trace, pt_on, pt_off, thr_on, thr_off, pt_pad=None):
        if not isinstance(source_trace, FoldTrace):
            raise TypeError
        
        if pt_pad is None:
            pt_pad = 0
        # Use __setattr__ safety checks
        self.pt_pad = pt_pad
        self.pt_on = pt_on
        self.pt_off = pt_off
        # Generate view of input source_trace
        tts = source_trace.stats.starttime + \
            (self.pt_on - self.pad_samples)*source_trace.stats.delta
        tte = source_trace.stats.starttime + \
            (self.pt_off + self.pad_samples)*source_trace.stats.delta
        view = source_trace.view(starttime=tts, endtime=tte).copy()

        if not all(view.data >= 0):
            raise ValueError('View of source_trace contains negative values - are you sure this is a CRF?')

        super().__init__(data=view.data,
                         fold=view.fold,
                         header=view.stats)
        
        self.pt_on = self.pt_pad
        self.pt_off = self.count() - self.pt_pad
        self.thr_on = thr_on
        self.thr_off = thr_off
    
    def __setattr__(self, key, value):
        """Enforce Type and Value rules for setting attribute values new
        to this :class:`~.Trigger` object.

        :param key: name of the attribute being set
        :type key: str
        :param value: new value to set to the attribute **key**
        :type value: varies
            - pt_X attributes: int-like
            - thr_X attributes: float-like
        """        
        if key in ['pt_pad','pt_on','pt_off']:
            if isinstance(value, (int, float)):
                if value == int(value):
                    value = int(value)
                else:
                    raise TypeError(f'{key} must be int-like')
            else:
                raise TypeError(f'{key} must be int-like')
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
        return np.argmax(np.abs(self.data))

    def get_max_time(self):
        """Get the timestamp of the absolute maximum value in the
        **data** attribute of this :class:`~.Trigger` object

        :return: _description_
        :rtype: _type_
        """
        xmax = self.get_max_index()
        return self.stats.starttime + xmax*self.stats.sampling_rate
    
    tmax = property(get_max_time)
    xmax = property(get_max_index)
    pmax = property(max)

    def basic_summary(self) -> dict:
        """Render a dictionary that contains a basic summary of the features
        of this :class:`~.Trigger` object

        Fields
        ------
        network - network code
        station - station code
        location - location code
        channel - channel code
        component - component character
        model - model code
        weight - weight code
        starttime - trigger starttime
        sampling_rate - trigger sampling rate
        xmax - index of the maximum CRF value
        pmax - maximum CRF value
        xon - index of the trigger onset
        pon - trigger onset value
        xoff - index of the trigger offset
        poff - trigger offset value
        fold_mean - mean value of the fold or this trigger

        :return: **summary** (*dict*)
        :rtype: dict
        """        
        summary = {}
        for _k in ['network','station','location','channel','component','model','weight','starttime','sampling_rate']:
            summary.update({_k: self.stats[_k]})
        summary.update({'xmax':self.xmax,
                        'pmax':self.pmax,
                        'xon': self.pt_on,
                        'pon': self.thr_on,
                        'xoff': self.pt_off,
                        'poff': self.thr_off,
                        'fold_mean': np.mean(self.fold)})
        return summary

    def estimate_gaussian_moments(self, fisher=False):
        """Under the assumption that this :class:`~.Trigger`'s **data** 
        has a similar shape to a normal/Gaussian probability density function,
        estimate the mean, standard deviation, skewness, and kurtosis of the
        roughly Gaussian PDF. This method provides options to use the 
        Pearson or Fisher definitions of Kurtosis

        :param fisher: Use the Fisher definition of kurtosis?
            I.e., a true normal distribution has a kurtosis of 3.
            Defaults to False
        :type fisher: bool, optional
        :returns: **result** (*dict*) - dictionary with keyed values for
            the mean, std, skew, and kurt_X. The "_X" corresponds
            to the kurtosis definition used.
        """        
        xv = np.arange(0, self.count())
        mean, std, skew, kurt = estimate_moments(xv, self.data, fisher=fisher) 

        out = [mean*self.stats.delta, std*self.stats.delta, skew, kurt]
        names = ['mean','std','skew']
        if fisher:
            names.append('kurt_f')
        else:
            names.append('kurt_p')
        
        result = dict(zip(names, out))
        return result
    
    def estimate_quantiles(self, quantiles=[0.16, 0.50, 0.84], decimals=4):
        """Under the assumption that this :class:`~.Trigger`'s **data**
        represent a probability density function, estimate the specified
        quantiles of that PDF.

        :param quantiles: Quantiles to estimate, defaults to [0.16, 0.50, 0.84]
        :type quantiles: list, optional
        :param decimals: quantile precision for labeling quantile estimates
            Not used in calculating the actual quantiles. Defaults to 4
        :type decimals: int, optional
        :returns: **result** (*dict*) -- dictionary with keys of **quantiles** 
            and the delta seconds values relative to this :class:`~.Trigger`'s
            **starttime** for each specified quantile

        """        
        xv = np.arange(0, self.count())*self.stats.delta
        out = estimate_quantiles(xv, self.data, q=quantiles)
        # Convert into delta seconds
        out = [_o * self.stats.delta for _o in out]
        names = [np.round(_q, decimals=decimals) for _q in quantiles]
        result = dict(zip(names, out))
        return result
    
    def fit_scaled_normal_pdf(self, mindata=30.):
        if self.count() >= mindata:
            yv = self.data
        else:
            # Symmetrically Zero-pad
            padsamples = (mindata - self.count())//2 + 1
            yv = np.r_[np.zeros(padsamples), self.data, np.zeros(padsamples)]

        xv = np.arange(len(yv))
        # Compose initial guess parameters: amplitude, mean, variance
        p0 = [self.pmax, self.xmax, (0.5*(self.pt_off - self.pt_on))**2]
        # Fit normal distribution
        pout, pcov, res = fit_normal_pdf_curve(xv, yv, threshold=0., mindata=mindata, p0=p0)
        return pout, pcov, res
    
    def scaled_gaussian_fit(self, mindata=30.) -> dict:

        pout, pcov, res = self.fit_scaled_normal_pdf(mindata=mindata)
        index = ['network','station','location','channel','model','weight',\
                 't0','sr','l2','p0_amp','p1_mean','p2_var']
        values = [self.stats[_k] for _k in index[:6]]
        values += list(pout)
        values.append(np.linalg.norm(res, order=2))
        for ii in range(3):
            for jj in range(ii+1):
                values.append(pcov[ii,jj])
                index.append(f'v_{ii}{jj}')
        
        series = pd.Series(data=values, index=index)
        return series
    
    def gq_fit_summary(self, mindata=30., fisher=False, quantiles=[0.16, 0.25, 0.5, 0.75, 0.84]):
        series = self.scaled_gaussian_fit(mindata=30.)
        g_results = self.estimate_gaussian_moments(fisher=fisher)



    def to_pick(self, pick='max', uncertainty='trigger', confidence='pmax_scaled', **kwargs):
        """
        Convert the metadata in this trigger into an ObsPy :class:`~.Pick` object
        with a specified formatting to convey trigger size and peak value




        sigma options
        ---------------------
        None -- No uncertainties reported
        'trigger' -- uses the time off
        'pmax_weighted' -- confidence interval for the pick uncertainty is calculated as:
          :math:`100*(1 - \\frac{0.5*(thr_ON + thr_OFF)}{pmax})`
        
        """
        if pick not in ['max','mean','median']:
            raise ValueError(f'pick type "{pick}" not supported.')
        if uncertainty not in ['sigma','std','iqr']:
            raise ValueError(f'uncertainty type "{uncertainty}" not supported.')
        
        pkwargs = {}

        # Get pick time
        if pick == 'max':
            tpick = self.tmax()
        elif pick == 'mean':
            if 'fisher' in kwargs.keys():
                fisher = kwargs['fisher']
            result = self.estimate_gaussian_moments(**pkwargs)
            tpick = self.stats.starttime + result['mean']
        elif pick == 'median':
            if 'decimals' in kwargs.keys():
                pkwargs.update({'decimals': kwargs['decimals']})
            result = self.estimate_quantiles(quantiles=[0.5], **pkwargs)
            tpick = self.stats.starttime + result[0.5]
        else:
            raise ValueError
        
        # Get uncertainties
        if sigma 


    def to_gaussian(self, fisher=False):
        moments = self.estimate_gaussian_moments(fisher=fisher)






class Trigger(object):
    """A class providing a more-detailed representation of a characteristic
    response function (CRF) trigger.

    See :meth:`~obspy.signal.trigger.trigger_onset`

    :param source_trace: Trace-like object used to generate
    :type source_trace: PULSE.data.foldtrace.FoldTrace
    :param trigger: trigger on and off indices (iON and iOFF)
    :type trigger: 2-tuple of int
    :param trigger_level: triggering level used to generate trigger
    :type trigger_level: float
    :param padding_samples: extra samples to include outboard of the trigger, defaults to 0
    :type padding_samples: non-negative int, optional

    :attributes:
        - **self.trace.data** (*numpy.ndarray*) -- data samples from **source_trace** for this trigger
        - **self.trace.fold** (*numpy.ndarray*) -- fold values from **source_trace** for this trigger
        - **self.trace.starttime** (*obspy.core.utcdatetime.UTCDateTime) -- starting timestamp from **source_trace**
        - **self.trace.sampling_rate** (*float*) -- sampling rate from **source_trace**
        - **self.network** (*str*) -- network attribute from **source_trace.stats**
        - **self.station** (*str*) -- station attribute from **source_trace.stats**
        - **self.location** (*str*) -- location attribute from **source_trace.stats**
        - **self.channel** (*str*) -- channel attribute from **source_trace.stats**
        - **self.model** (*str*) -- model attribute from **source_trace.stats**
        - **self.weight** (*str*) -- weight attribute from **source_trace.stats**
        - **self.iON** (*int*) -- trigger ON index value
        - **self.tON** (*obspy.core.utcdatetime.UTCDateTime*) -- time of trigger ON
        - **self.iOFF** (*int*) -- trigger OF index value
        - **self.tOFF** (*obspy.core.utcdatetime.UTCDateTime*) -- time of trigger OFF
        - **self.trigger_level** (*float*) -- triggering threshold
        - **self.tmax** (*obspy.core.utcdatetime.UTCDateTime*) -- time maximum data value in source_trace.data[iON, iOFF]
        - **self.pmax** (*float*) -- maximum data value in source_trace.data[iON:iOFF]
        - **self.pick2k** (*ayahos.core.pick.Pick2KMsg*) -- TYPE_PICK2K data class object
    """
    def __init__(self, source_trace, trigger, trigger_level, padding_samples=0):
        if isinstance(source_trace, Trace):
            pass
        else:
            raise TypeError

        # Compat check for trigger
        if isinstance(trigger, (list, tuple, np.ndarray)):
            if len(trigger) == 2: 
                if trigger[0] < trigger[1]:
                    self.iON = trigger[0]
                    self.tON = source_trace.stats.starttime + self.iON/source_trace.stats.sampling_rate
                    self.iOFF = trigger[1]
                    self.tOFF = source_trace.stats.starttime + self.iOFF/source_trace.stats.sampling_rate
                else:
                    raise ValueError('trigger ON index is larger than OFF index')
            else:
                raise ValueError('trigger must be a 2-element array-like')
        else:
            raise TypeError('trigger must be type numpy.ndarray, list, or tuple')

        # Compat. Check for padding_samples
        if isinstance(padding_samples, int):
            self.padding_samples = padding_samples
        else:
            raise TypeError

        # Get data snippet
        self.trace = source_trace.view_copy(
            starttime=self.tON - self.padding_samples/source_trace.stats.sampling_rate,
            endtime=self.tOFF + self.padding_samples/source_trace.stats.sampling_rate)

        # Get maximum trigger level
        self.tmax = self.tON + np.argmax(self.trace.data)/self.trace.stats.sampling_rate
        self.pmax = np.max(self.trace.data)

        # Compatability check on trigger_level
        if isinstance(trigger_level, float):
            if np.isfinite(trigger_level):
                self.trigger_level = trigger_level
            else:
                raise ValueError
        else:
            raise TypeError
        self.pref_pick_pos = self.iON + (self.tmax - self.tON)*self.trace.stats.sampling_rate
        self.pref_pick_time = self.tmax
        self.pref_pick_prob = self.pmax
        self.pick_type='max'
        # Placeholder for pick obj
        self.pick2k = None

    def get_site(self):
        rstr = f'{self.trace.stats.network}.{self.trace.stats.station}'
        return rstr
    
    site = property(get_site)

    def get_id(self):
        rstr = self.trace.id
        return rstr
    
    id = property(get_id)

    def get_label(self):
        rstr = self.trace.stats.channel[-1]
        return rstr
    
    label = property(get_label)  

    def set_pick2k(self, pick2k):
        if isinstance(pick2k, Pick2KMsg):
            self.pick2k = pick2k

    def get_pick2k_msg(self):
        if self.pick2k is not None:
            msg = self.pick2k.generate_msg()
        else:
            msg = ''
        return msg

    def __repr__(self):
        rstr = f'{self.__class__.__name__}\n'
        rstr += f'Pick Time: {self.pref_pick_time} | '
        rstr += f'Pick Type: {self.pick_type} | '
        rstr += f'Pick Value: {self.pref_pick_prob}\n'
        rstr += f'Window Position: {self.iON} / {self.pref_pick_pos} \ {self.iOFF}\n'
        rstr += f'Padded Trigger Trace:\n{self.trace.__repr__()}\n'
        
        return rstr




class GaussTrigger(Trigger):
    """
    This :class:`~ayahos.core.trigger.Trigger` child class adds a scaled Gaussian model fitting
    to trigger data with the model

    .. math:
        CRF(t) = p_0*e^{\\frac{-(t - p_1)^2}{2 p_2}}
    with
        - :math:`p_0` = Amplitude/scale of the Gaussian
        - :math:`p_1` = Mean/central value of the Gaussian (:math:`\\mu`)
        - :math:`p_2` = Variance of the Gaussian (:math:`\\sigma^2`)

    fit using :meth:`~ayahos.util.stats.fit_normal_pdf_curve` with threshold=trigger_level
    unless otherwise specified in **options. The model parameters are tied to individual
    attributes "scale", "mean", and "var". The model covariance matrix and model-data
    residuals are assigned to the "cov" and "res" attributes, respectively. This class
    provides convenience methods for calculating L1 and L2 norms of model-data residuals

    :added attributes:
        - **

    """    
    def __init__(self, source_trace, trigger, trigger_level, padding_samples=0, **options):
        """Initialize a GaussTrigger object

        """        
        super().__init__(source_trace, trigger, trigger_level, padding_samples=padding_samples)
        
        x = np.arange(self.iON, self.iOFF)/self.trace.sampling_rate + self.trace.starttime.timestamp
        y = self.trace.data
        # If user does not define the threshold for fitting, use the trigger_level
        if 'threshold' not in options.keys():
            options.update({'threshold': self.trigger_level})
        p, cov, res = fit_normal_pdf_curve(x, y, **options)
        self.scale = p[0]
        self.mean = p[1]
        self.var = p[2]
        self.cov = cov
        self.res = res
        # Overwrite reference inherited from Trigger
        self.pref_pick_time = self.mean
        self.pref_pick_prob = self.scale
        self.pick_type='Gaussian mean'

    def get_l1(self):   
        norm = np.linalg.norm(self.res, ord=1)
        return norm
    
    L1 = property(get_l1)

    def get_l2(self):
        norm = np.linalg.norm(self.res, order=2)
        return norm
    
    L2 = property(get_l2)

    def __repr__(self):
        rstr = super().__repr__()
        rstr += f'\nResidual | L1: {self.L1} | L2: {self.L2}'


class QuantTrigger(Trigger):

    def __init__(self, source_trace, trigger, trigger_level, padding_samples=10, quantiles=[0.159, 0.5, 0.841]):

        super().__init__(source_trace, trigger, trigger_level, padding_samples=padding_samples)

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
        
        if 0.5 not in self.quantiles:
            self.quantiles.append(0.5)
        
        self.quantiles.sort

        y = self.trace.data
        x = np.arange(self.iON, self.iOFF)
        samples, probabilities = estimate_quantiles(x, y)
        self.probabilities = probabilities
        self.times = [self.tON + smp/self.trace.sampling_rate for smp in samples]
        self.tmed = self.times[self.quantiles==0.5]
        self.pmed = self.probabilities[self.quantiles==0.5]

        self.pref_pick_time = self.tmed
        self.pref_pick_prob = self.pmed
        self.pick_type='Estimated median'

    # def __repr__(self):
    #     rstr = super().__repr__():
    #     rstr += 