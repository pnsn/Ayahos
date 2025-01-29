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
from PULSE.util.stats import estimate_moments, estimate_quantiles
import numpy as np
from pandas import Series

from obspy import Trace, UTCDateTime
from obspy.core.event import Pick, QuantityError, WaveformStreamID, ResourceIdentifier
from obspy.core.util.attribdict import AttribDict
from PULSE.data.foldtrace import FoldTrace
from scipy.cluster.vq import *

Logger = logging.getLogger(__name__)

from obspy.core.util.attribdict import AttribDict

class Pick(AttribDict):

    _readonly = ['pick_time']
    _refresh_keys = ['pick_param','params','ref_time','ref_sampling_rate']

    defaults = {'network': '',
                'station': '',
                'location': '',
                'channel': '',
                'detection_method': '',
                'trigger_model': '',
                'pick_time': None,
                'label': '',
                'ref_time': UTCDateTime(0),
                'ref_sampling_rate': 1.,
                'pick_param':None,
                'params': {},
                'pcov': None,
                'fit_quality': {}}
    
    _types = {'network': str,
              'station': str,
              'location': str,
              'channel': str,
              'detection_method': str,
              'trigger_model': str,
              'pick_time': (type(None), UTCDateTime),
              'label': str,
              'ref_time': UTCDateTime,
              'ref_sampling_rate': (int, float),
              'pick_param': (type(None), str),
              'params': dict,
              'pcov': (type(None), np.ndarray),
              'fit_quality': dict
              }
    
    def __init__(self, header={}):
        if not isinstance(header, dict):
            raise TypeError('header must be type dict')
        # If params is specified, set this first
        if 'params' in header.keys():
            self.__setattr__('params', header.pop('params'))
        # If pcov is specified set this second
        if 'pcov' in header.keys():
            self.__setattr__('pcov', header.pop('pcov'))
        
        # Finally, set the rest of the parameters
        for _k, _v in header.items():
            self.__setattr__(_k, _v)
        
    def __setattr__(self, key, value):
        # Cross-check with defaults and _types
        if key in self.defaults.keys():
            if isinstance(value, self._types[key]):
                pass
            else:
                raise TypeError(f'value of type {type(value)} not supported for "{key}"')
        else:
            raise KeyError(f'key "{key}" not in defaults')
        
        if key in self._readonly:
            raise KeyError(f'{key} is readonly')

        # Make sure pick_param is specified in params
        if key == 'pick_param':
            if value is None:
                return
            elif value not in self.params.keys():
                raise KeyError('cannot assign "pick_param" as "{value}" - not mapped in "params" keys')

        # Make sure pcov matches scale of params, if not None
        if key == 'pcov':
            if value is None:
                return
            elif value.shape != (len(self.params), len(self.params)):
                raise ValueError('scale of parameter covariance matrix (pcov) does not match scale of "params"')

        if key in self._refresh_keys:
            super(Pick, self).__setitem__(key, value)
            self.__dict__['pick_time'] =  self._get_pick_time()
            return
        else:
            super(Pick, self).__setattr__(key, value)

    def _get_pick_time(self):
        """Supporting private method to calculate the pick_time from
        other writeable entries
        """        
        t0 = self.ref_time
        if len(self.params) > 0 and self.pick_param != self.defaults['pick_param']:
            dt = self.params[self.pick_param]/self.sampling_rate
            return t0 + dt
        else:
            return None
        
    def __str__(self):
        prioritized_keys = ['network','station','location','channel',
                            'pick_time','label','detection_method',
                            'trigger_model','pick_param']
        return self._pretty_str(prioritized_keys=prioritized_keys)
    
    def _repr_pretty(self, p, cycle):
        p.text(str(self))


    def get_seed_string(self):
        return '.'.join([self[_k] for _k in ['network','station','location','channel']]) 

    id = property(get_seed_string)   


    def get_similarity(self, other, metric='kld', soft_max=5, null_threshold=2):
        # Make sure other is a Pick object
        if not isinstance(other, Pick):
            raise TypeError(f'other must be type PULSE.data.pick.Pick')
        # Make sure this pick and other are compatable - same seismic site, trigger model and pick parameter
        for _k in ['network','station','trigger_model','pick_param']:
            if self[_k] != other[_k]:
                msg = f'{_k} mismatch: self ({self[_k]}) != ({other[_k]}) other'
                raise AttributeError(msg)
            
        # No comparison for strictly observation    
        if self.trigger_model == 'obs':
            raise NotImplementedError('Similarity testing not supported for "obs"erved trigger models')
        
        # Apply softmax for QnD test of separation
        du = self.params[self.pick_param] - other.params[other.pick_param]
        
        

        # If using Kullbeck-Leibler divergence
        if metric == 'kld':
            if self.trigger_model == 'gaussian':
                kld = kld_gaussian(self.params['mean'], self.params['var'],
                                   other.params['mean'], other.params['var'])
            elif self.trigger_model == 'triangle':
                kld = kld_triangle_est(list(self.params.values()),
                                       list(other.params.values()))
            elif self.trigger_model == 'boxcar':
                kld = kld_uniform(self.params['onset'],self.params['offset'],
                                  other.params['onset'], self.params['offset'])











    # def __init__(
    #         self,
    #         id_string: str,
    #         label: str,
    #         ref_time: UTCDateTime,
    #         ref_level: float,
    #         ref_param: str,
    #         fit_model: str,
    #         fit_quality: float):
        
        
    #     self.id = id_string
    #     self.label = label
    #     self.ref_time = ref_time
    #     self.ref_level = ref_level
    #     self.ref_param = ref_param
    #     self.fit_model = fit_model
    #     self.fit_quality = fit_quality


    # def __setattr__(self, key, value):
    #     if key in ['id_string', 'ref_param','fit_model', 'label']:
    #         if not isinstance(value, str):
    #             raise TypeError
    #     if key == 'ref_time':
    #         if not isinstance(value, UTCDateTime):
    #             raise TypeError
    #     if key in ['ref_level','fit_quality']:
    #         if not isinstance(value, float):
    #             raise TypeError
    #         elif not 0 <= value:
    #             raise ValueError(f'{key} value must be non-negative')
    #         elif key == 'fit_quality' and value > 1:
    #             raise SyntaxError(f'{key} value must be in [0, 1]')
            

    # def to_dict(self):
    #     fields = ['id','label','ref_time','ref_level','ref_param','fit_model','fit_quality']
    #     out = {fld: getattr(self, fld) for fld in fields}
    #     return out
            
                


    # def basic_summary(self) -> dict:
    #     """Render a dictionary that contains a basic summary of the features
    #     of this :class:`~.Trigger` object

    #     Fields
    #     ------
    #     network - network code
    #     station - station code
    #     location - location code
    #     channel - channel code
    #     component - component character
    #     model - model code
    #     weight - weight code
    #     starttime - trigger starttime
    #     sampling_rate - trigger sampling rate
    #     imax - index of the maximum CRF value
    #     pmax - maximum CRF value
    #     ion - index of the trigger onset
    #     pon - trigger onset value
    #     ioff - index of the trigger offset
    #     poff - trigger offset value
    #     fold_mean - mean value of the fold or this trigger

    #     :return: **summary** (*dict*)
    #     :rtype: dict
    #     """        
    #     summary = {}
    #     for _k in ['network','station','location','channel','component','model','weight','starttime','sampling_rate']:
    #         summary.update({_k: self.stats[_k]})
    #     summary.update({'imax':self.xmax,
    #                     'pmax':self.pmax,
    #                     'ion': self.samp_on,
    #                     'pon': self.thr_on,
    #                     'ioff': self.samp_off,
    #                     'poff': self.thr_off,
    #                     'fold_mean': np.mean(self.fold)})
    #     return summary

    
    # def _to_gaussian_pick(self):

    #     if not hasattr(self.measures,'gau_model'):
    #         raise AttributeError('Gaussian model fit not yet run on this Trigger')
        
    #     # Get mean pick time
    #     tp = self.stats.starttime + self.measures.gau_model['m'][1]*self.stats.delta

    
    
    

    
#     def scaled_gaussian_fit(self, mindata=30.) -> dict:

#         pout, pcov, res = self.fit_scaled_normal_pdf(mindata=mindata)
#         index = ['network','station','location','channel','model','weight',\
#                  't0','sr','l2','p0_amp','p1_mean','p2_var']
#         values = [self.stats[_k] for _k in index[:6]]
#         values += list(pout)
#         values.append(np.linalg.norm(res, order=2))
#         for ii in range(3):
#             for jj in range(ii+1):
#                 values.append(pcov[ii,jj])
#                 index.append(f'v_{ii}{jj}')
        
#         series = pd.Series(data=values, index=index)
#         return series
    
#     def gq_fit_summary(self, mindata=30., fisher=False, quantiles=[0.16, 0.25, 0.5, 0.75, 0.84]):
#         series = self.scaled_gaussian_fit(mindata=30.)
#         g_results = self.estimate_gaussian_moments(fisher=fisher)



#     def to_pick(self, pick='max', uncertainty='trigger', confidence='pmax_scaled', **kwargs):
#         """
#         Convert the metadata in this trigger into an ObsPy :class:`~.Pick` object
#         with a specified formatting to convey trigger size and peak value

#         sigma options
#         ---------------------
#         None -- No uncertainties reported
#         'trigger' -- uses the time off
#         'pmax_weighted' -- confidence interval for the pick uncertainty is calculated as:
#           :math:`100*(1 - \\frac{0.5*(thr_ON + thr_OFF)}{pmax})`
        
#         """
#         if pick not in ['max','mean','median']:
#             raise ValueError(f'pick type "{pick}" not supported.')
#         if uncertainty not in [None, 'sigma','std','iqr']:
#             raise ValueError(f'uncertainty type "{uncertainty}" not supported.')
        
#         pkwargs = {}

#         # Get pick time
#         if pick == 'max':
#             tpick = self.tmax()
#         elif pick == 'mean':
#             if 'fisher' in kwargs.keys():
#                 fisher = kwargs['fisher']
#             result = self.estimate_gaussian_moments(**pkwargs)
#             tpick = self.stats.starttime + result['mean']
#         elif pick == 'median':
#             if 'decimals' in kwargs.keys():
#                 pkwargs.update({'decimals': kwargs['decimals']})
#             result = self.estimate_quantiles(quantiles=[0.5], **pkwargs)
#             tpick = self.stats.starttime + result[0.5]
#         else:
#             raise ValueError
        
#         # Get uncertainties
#         if sigma is None:
#             tsigma = None
#         elif sigma == 'pmax_scaled':
#             tsigma = QuantityError(
#                 lower_uncertainty=None,
#                 upper_uncertainty=None,
#                 confidence_level=None
#             )


#         pick = Pick(time=tpick,
#                     resource_id=ResourceIdentifier(prefix='smi:local/pulse/data/pick'),
#                     time_errors=tsigma,
#                     waveform_id=WaveformStreamID(
#                         network_code=self.stats.network,
#                         station_code=self.stats.station,
#                         location_code=self.stats.location,
#                         channel_code=self.stats.channel),
#                     method_id=ResourceIdentifier(id=f'smi:local/pulse/data/pick/{self.stats.model}/{self.stats.weight}'),
#                     phase_hint=self.id_keys['component'])
        
#         return pick



#     def to_gaussian(self, fisher=False):
#         moments = self.estimate_gaussian_moments(fisher=fisher)






# class Trigger(object):
#     """A class providing a more-detailed representation of a characteristic
#     response function (CRF) trigger.

#     See :meth:`~obspy.signal.trigger.trigger_onset`

#     :param source_trace: Trace-like object used to generate
#     :type source_trace: PULSE.data.foldtrace.FoldTrace
#     :param trigger: trigger on and off indices (iON and iOFF)
#     :type trigger: 2-tuple of int
#     :param trigger_level: triggering level used to generate trigger
#     :type trigger_level: float
#     :param padding_samples: extra samples to include outboard of the trigger, defaults to 0
#     :type padding_samples: non-negative int, optional

#     :attributes:
#         - **self.trace.data** (*numpy.ndarray*) -- data samples from **source_trace** for this trigger
#         - **self.trace.fold** (*numpy.ndarray*) -- fold values from **source_trace** for this trigger
#         - **self.trace.starttime** (*obspy.core.utcdatetime.UTCDateTime) -- starting timestamp from **source_trace**
#         - **self.trace.sampling_rate** (*float*) -- sampling rate from **source_trace**
#         - **self.network** (*str*) -- network attribute from **source_trace.stats**
#         - **self.station** (*str*) -- station attribute from **source_trace.stats**
#         - **self.location** (*str*) -- location attribute from **source_trace.stats**
#         - **self.channel** (*str*) -- channel attribute from **source_trace.stats**
#         - **self.model** (*str*) -- model attribute from **source_trace.stats**
#         - **self.weight** (*str*) -- weight attribute from **source_trace.stats**
#         - **self.iON** (*int*) -- trigger ON index value
#         - **self.tON** (*obspy.core.utcdatetime.UTCDateTime*) -- time of trigger ON
#         - **self.iOFF** (*int*) -- trigger OF index value
#         - **self.tOFF** (*obspy.core.utcdatetime.UTCDateTime*) -- time of trigger OFF
#         - **self.trigger_level** (*float*) -- triggering threshold
#         - **self.tmax** (*obspy.core.utcdatetime.UTCDateTime*) -- time maximum data value in source_trace.data[iON, iOFF]
#         - **self.pmax** (*float*) -- maximum data value in source_trace.data[iON:iOFF]
#         - **self.pick2k** (*ayahos.core.pick.Pick2KMsg*) -- TYPE_PICK2K data class object
#     """
#     def __init__(self, source_trace, trigger, trigger_level, padding_samples=0):
#         if isinstance(source_trace, Trace):
#             pass
#         else:
#             raise TypeError

#         # Compat check for trigger
#         if isinstance(trigger, (list, tuple, np.ndarray)):
#             if len(trigger) == 2: 
#                 if trigger[0] < trigger[1]:
#                     self.iON = trigger[0]
#                     self.tON = source_trace.stats.starttime + self.iON/source_trace.stats.sampling_rate
#                     self.iOFF = trigger[1]
#                     self.tOFF = source_trace.stats.starttime + self.iOFF/source_trace.stats.sampling_rate
#                 else:
#                     raise ValueError('trigger ON index is larger than OFF index')
#             else:
#                 raise ValueError('trigger must be a 2-element array-like')
#         else:
#             raise TypeError('trigger must be type numpy.ndarray, list, or tuple')

#         # Compat. Check for padding_samples
#         if isinstance(padding_samples, int):
#             self.padding_samples = padding_samples
#         else:
#             raise TypeError

#         # Get data snippet
#         self.trace = source_trace.view_copy(
#             starttime=self.tON - self.padding_samples/source_trace.stats.sampling_rate,
#             endtime=self.tOFF + self.padding_samples/source_trace.stats.sampling_rate)

#         # Get maximum trigger level
#         self.tmax = self.tON + np.argmax(self.trace.data)/self.trace.stats.sampling_rate
#         self.pmax = np.max(self.trace.data)

#         # Compatability check on trigger_level
#         if isinstance(trigger_level, float):
#             if np.isfinite(trigger_level):
#                 self.trigger_level = trigger_level
#             else:
#                 raise ValueError
#         else:
#             raise TypeError
#         self.pref_pick_pos = self.iON + (self.tmax - self.tON)*self.trace.stats.sampling_rate
#         self.pref_pick_time = self.tmax
#         self.pref_pick_prob = self.pmax
#         self.pick_type='max'
#         # Placeholder for pick obj
#         self.pick2k = None

#     def get_site(self):
#         rstr = f'{self.trace.stats.network}.{self.trace.stats.station}'
#         return rstr
    
#     site = property(get_site)

#     def get_id(self):
#         rstr = self.trace.id
#         return rstr
    
#     id = property(get_id)

#     def get_label(self):
#         rstr = self.trace.stats.channel[-1]
#         return rstr
    
#     label = property(get_label)  

#     def set_pick2k(self, pick2k):
#         if isinstance(pick2k, Pick2KMsg):
#             self.pick2k = pick2k

#     def get_pick2k_msg(self):
#         if self.pick2k is not None:
#             msg = self.pick2k.generate_msg()
#         else:
#             msg = ''
#         return msg

#     def __repr__(self):
#         rstr = f'{self.__class__.__name__}\n'
#         rstr += f'Pick Time: {self.pref_pick_time} | '
#         rstr += f'Pick Type: {self.pick_type} | '
#         rstr += f'Pick Value: {self.pref_pick_prob}\n'
#         rstr += f'Window Position: {self.iON} / {self.pref_pick_pos} \ {self.iOFF}\n'
#         rstr += f'Padded Trigger Trace:\n{self.trace.__repr__()}\n'
        
#         return rstr




# class GaussTrigger(Trigger):
#     """
#     This :class:`~ayahos.core.trigger.Trigger` child class adds a scaled Gaussian model fitting
#     to trigger data with the model

#     .. math:
#         CRF(t) = p_0*e^{\\frac{-(t - p_1)^2}{2 p_2}}
#     with
#         - :math:`p_0` = Amplitude/scale of the Gaussian
#         - :math:`p_1` = Mean/central value of the Gaussian (:math:`\\mu`)
#         - :math:`p_2` = Variance of the Gaussian (:math:`\\sigma^2`)

#     fit using :meth:`~ayahos.util.stats.fit_normal_pdf_curve` with threshold=trigger_level
#     unless otherwise specified in **options. The model parameters are tied to individual
#     attributes "scale", "mean", and "var". The model covariance matrix and model-data
#     residuals are assigned to the "cov" and "res" attributes, respectively. This class
#     provides convenience methods for calculating L1 and L2 norms of model-data residuals

#     :added attributes:
#         - **

#     """    
#     def __init__(self, source_trace, trigger, trigger_level, padding_samples=0, **options):
#         """Initialize a GaussTrigger object

#         """        
#         super().__init__(source_trace, trigger, trigger_level, padding_samples=padding_samples)
        
#         x = np.arange(self.iON, self.iOFF)/self.trace.sampling_rate + self.trace.starttime.timestamp
#         y = self.trace.data
#         # If user does not define the threshold for fitting, use the trigger_level
#         if 'threshold' not in options.keys():
#             options.update({'threshold': self.trigger_level})
#         p, cov, res = fit_normal_pdf_curve(x, y, **options)
#         self.scale = p[0]
#         self.mean = p[1]
#         self.var = p[2]
#         self.cov = cov
#         self.res = res
#         # Overwrite reference inherited from Trigger
#         self.pref_pick_time = self.mean
#         self.pref_pick_prob = self.scale
#         self.pick_type='Gaussian mean'

#     def get_l1(self):   
#         norm = np.linalg.norm(self.res, ord=1)
#         return norm
    
#     L1 = property(get_l1)

#     def get_l2(self):
#         norm = np.linalg.norm(self.res, order=2)
#         return norm
    
#     L2 = property(get_l2)

#     def __repr__(self):
#         rstr = super().__repr__()
#         rstr += f'\nResidual | L1: {self.L1} | L2: {self.L2}'


# class QuantTrigger(Trigger):

#     def __init__(self, source_trace, trigger, trigger_level, padding_samples=10, quantiles=[0.159, 0.5, 0.841]):

#         super().__init__(source_trace, trigger, trigger_level, padding_samples=padding_samples)

#         if isinstance(quantiles, float):
#             if 0 < quantiles < 1:
#                 self.quantiles = [quantiles]
#             else:
#                 raise ValueError
#         elif isinstance(quantiles, list):
#             if all(0 < q < 1 for q in quantiles):
#                 self.quantiles = quantiles
#             else:
#                 raise ValueError
#         else:
#             raise TypeError
        
#         if 0.5 not in self.quantiles:
#             self.quantiles.append(0.5)
        
#         self.quantiles.sort

#         y = self.trace.data
#         x = np.arange(self.iON, self.iOFF)
#         samples, probabilities = estimate_quantiles(x, y)
#         self.probabilities = probabilities
#         self.times = [self.tON + smp/self.trace.sampling_rate for smp in samples]
#         self.tmed = self.times[self.quantiles==0.5]
#         self.pmed = self.probabilities[self.quantiles==0.5]

#         self.pref_pick_time = self.tmed
#         self.pref_pick_prob = self.pmed
#         self.pick_type='Estimated median'

#     # def __repr__(self):
#     #     rstr = super().__repr__():
#     #     rstr += 