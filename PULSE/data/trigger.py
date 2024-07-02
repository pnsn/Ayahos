"""
:module: camper.data.pick
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This contains class definitions for Pick and Trigger data objects used
    to compactly describe characteristic response function triggers and arrival
    time picks for seismic body waves.
"""
import logging
from PULSE.util.stats import estimate_moments, fit_normal_pdf_curve, estimate_quantiles
import numpy as np
from obspy import Trace, UTCDateTime
from obspy.core.util.attribdict AttribDict
from PULSE.data.mltrace import MLTrace
from scipy.cluster.vq import *

Logger = logging.getLogger(__name__)

from obspy.core.util.attribdict import AttribDict

class Logo(AttribDict):
    """
    A class for Earthworm LOGO information
    """
    defaults = {
        'MOD_ID': None,
        'INST_ID': None,
        'TYPE': None,
        'TYPE_NAME': None,
    }
    
    _types = {'MOD_ID': (type(None), int),
              'INST_ID': (type(None), int),
              'TYPE': (type(None), int),
              'TYPE_NAME': (type(None), str)}

    def __init__(self, **kwargs):
        super().__init__()
        for _k, _v in kwargs.items():
            if _k in self.defaults:
                if _k in ['MOD_ID','INST_ID','TYPE']:
                    if 0 <= _v <= 255:
                        if int(_v) == _v:
                            self.update({_k: int(_v)})
                        else:
                            raise TypeError
                    else:
                        raise ValueError
                else:
                    if isinstance(_v, str):
                        if 'TYPE' in _v:
                            self.update({_k: _v})
                        else:
                            raise SyntaxError
                    else:
                        raise TypeError
            else:
                raise KeyError

class Trigger(object):
    """A class providing a more detailed representation of a characteristic
    response function (CRF) trigger.

    See :meth:`~obspy.signal.trigger.trigger_onset`

    :param source_trace: Trace-like object used to generate
    :type source_trace: ayahos.core.mltrace.MLTrace
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
    def __init__(self, source_trace, trigger, trigger_level, padding_samples=0, logo=None):
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