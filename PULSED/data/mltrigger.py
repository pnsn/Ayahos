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
from PULSED.util.stats import estimate_moments, fit_normal_pdf_curve, estimate_quantiles
import numpy as np
from obspy import Trace, UTCDateTime
from PULSED.data.mltrace import MLTrace
from scipy.cluster.vq import *

Logger = logging.getLogger(__name__)

class Pick2KMsg(object):
    """Class for holding data and metadata necessary for an EARTHWORM TYPE_PICK2K message

    http://www.earthwormcentral.org/documentation4/PROGRAMMER/y2k-formats.html

    :param mod_id: module ID number
        must match the MOD_ID attribute of the :class:`~ayahos.core.ayahos.Ayahos` object orchestrating this module
        mod_id:math:`\in[1,255]
    :type mod_id: int
    :param inst_id: installation ID number, code 255 for internal use only
        inst_id:math:`\in[1,255]
    :type inst_id: int
    :param seq_no: sequence number, code for associating this pick with coda data
        seq_no:math:`\in[0,9999]`
    :type seq_no: int
    :param net: network code (e.g., 'UW'), truncates at 2 characters
    :type net: str
    :param sta: station / site code (e.g., 'GNW'), truncates at 5 characters
    :type sta: str
    :param comp: component code (e.g., 'HHZ'), truncates at 3 characters
    :type comp: str
    :param phz: phase hint (e.g., 'P','S')
    :type phz: str
    :param qual: pick quality ranging from 0 (best) to 4 (worst)
    :type qual: int
    :param time: pick time
    :type time: obspy.core.utcdatetime.UTCDateTime

    TODO: Work out how to propagate this data around prediction
    :param pol: polarity of first break, defaults to '.' (undetermined), truncates at 1 character
        TODO: QC this 'undetermined' nomenclature
    :type pol: str, optional
    :param amp1: amplitude of first peak after arrival time, defaults to None
    :type amp1: int or None, optional
    :param amp2: amplitude of second peak after arrival time, defaults to None
    :type amp2: int or None, optional
    :param amp3: amplitdue of third peak after arrival time, defaults to None
    :type amp3: int or None, optional
    """    
    def __init__(
            self,
            mod_id,
            inst_id,
            seq_no,
            net,
            sta,
            comp,
            phz,
            qual,
            time,
            pol='.',
            amp1=None,
            amp2=None,
            amp3=None
            ):
        """_summary_

        
        :raises ValueError: _description_
        :raises TypeError: _description_
        :raises ValueError: _description_
        :raises TypeError: _description_
        :raises TypeError: _description_
        :raises TypeError: _description_
        :raises TypeError: _description_
        :raises TypeError: _description_
        :raises ValueError: _description_
        :raises TypeError: _description_
        :raises TypeError: _description_
        :raises TypeError: _description_
        :raises TypeError: _description_
        """        
        self.type = 10
        if isinstance(mod_id, int):
            if 0 < mod_id < 256:
                self.mod_id = mod_id
            else:
                raise ValueError
        else:
            raise TypeError
        if isinstance(inst_id, int):
            if 0 < inst_id < 256:
                self.inst_id = inst_id
            else:
                raise ValueError
        elif inst_id is None:
            self.inst_id = 255
        else:
            raise TypeError
        
        if isinstance(seq_no, int):
            if 0 <= seq_no <= 9999:
                self.seq_no = seq_no
            else:
                raise ValueError
        else:
            raise TypeError

        if isinstance(net, str):
            if len(net) <= 2:
                self.net = net
            else:
                self.net = net[:2]
        else:
            raise TypeError
        
        if isinstance(sta, str):
            if len(sta) <= 2:
                self.sta = sta
            else:
                self.sta = sta[:5]
        else:
            raise TypeError
        if isinstance(comp, str):
            if len(comp) <= 2:
                self.comp = comp
            else:
                self.comp = comp[:3]
        else:
            raise TypeError
        
        if isinstance(phz, str):
            if len(phz) <= 2:
                self.phz = phz
            else:
                self.phz = phz[:2]
        else:
            raise TypeError
        
        if qual in [0, 1, 2, 3, 4]:
            self.qual = qual
        else:
            raise ValueError


        if isinstance(time, UTCDateTime):
            self.time = time
        else:
            raise TypeError('must be type obspy.core.utcdatetime.UTCDateTime')
        
        if pol in ['U','D','.']:
            self.pol = pol
        elif pol is None:
            self.pol = '.'

        if isinstance(amp1, int):
            if len(str(amp1)) <= 8:
                self.amp1=amp1
            else:
                Logger.warning('amp1 exceeds TYPE_PICK2K scale - assigning 99999999')
                self.amp1 = 99999999
        elif amp1 is None:
            self.amp1 = ''
        else:
            raise TypeError
        
        if isinstance(amp2, int):
            if len(str(amp2)) <= 8:
                self.amp2=amp2
            else:
                Logger.warning('amp2 exceeds TYPE_PICK2K scale - assigning 99999999')
                self.amp2 = 99999999
        elif amp2 is None:
            self.amp2 = ''
        else:
            raise TypeError
        
        if isinstance(amp3, int):
            if len(str(amp3)) <= 8:
                self.amp3=amp3
            else:
                Logger.warning('amp3 exceeds TYPE_PICK2K scale - assigning 99999999')
                self.amp3 = 99999999
        elif amp3 is None:
            self.amp3 = ''
        else:
            raise TypeError
        
        self.msg = self.generate_msg()

    def generate_msg(self):
        """Generate an EARTHWORM TYPE_PICK2K formatted message string
        from the attribute data in this Pick2KMsg object

        :return: TYPE_PICK2K formatted message string
        :rtype: str
        """        
        ids = f'{self.type:>3}{self.module_id:>3}{self.inst_id:>3} {self.seq_no:>4}'
        codes = f'{self.sta:<5}{self.net:<2}{self.comp:<3}'
        attr = f'{self.pol:1}{self.qual:1}{self.phz:2}'
        idate = f'{self.time.year:4}{self.time.month:02}{self.time.day:02}'
        itime = f'{self.time.hour:02}{self.time.minute:02}{self.time.second}'
        csec = str(self.time.microsecond)[:2]
        amps = f'{self.amp1:8}{self.amp2:8}{self.amp3:8}'
        msg = f'{ids} {codes} {attr}{idate}{itime}{csec}{amps}\n'
        return msg


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
        - **self.data** (*numpy.ndarray*) -- data samples from **source_trace** for this trigger
        - **self.fold** (*numpy.ndarray*) -- fold values from **source_trace** for this trigger
        - **self.starttime** (*obspy.core.utcdatetime.UTCDateTime) -- starting timestamp from **source_trace**
        - **self.sampling_rate** (*float*) -- sampling rate from **source_trace**
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
        # Compat check for source_trace
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
        else:
            raise TypeError

        # Compat check for trigger
        if isinstance(trigger, (list, tuple)):
            if len(trigger) == 2: 
                if trigger[0] < trigger[1]:
                    self.iON = trigger[0]
                    self.tON = self.starttime + self.iON/self.sampling_rate
                    self.iOFF = trigger[1]
                    self.tOFF = self.starttime + self.iOFF/self.sampling_rate
                else:
                    raise ValueError('trigger ON index is larger than OFF index')
            else:
                raise ValueError('trigger must be a 2-list or 2-tuple')
        else:
            raise TypeError('trigger must be type list or tuple')

        # Compat. Check for padding_samples
        if isinstance(padding_samples, int):
            self.padding_samples = padding_samples
        else:
            raise TypeError

        # Get data snippet
        trig_trace = source_trace.view_copy(
            starttime=self.tON - self.padding_samples/self.sampling_rate,
            endtime=self.tOFF + self.padding_samples/self.sampling_rate)
        self.data = trig_trace.data
        self.fold = trig_trace.fold

        # Get maximum trigger level
        self.tmax = self.tON + np.argmax(self.data)/self.sampling_rate
        self.pmax = np.max(self.data)

        # Compatability check on trigger_level
        if isinstance(trigger_level, float):
            if np.isfinite(trigger_level):
                self.trigger_level = trigger_level
            else:
                raise ValueError
        else:
            raise TypeError
        
        self.pref_pick_time = self.tmax
        self.pref_pick_prob = self.pmax

        # Placeholder for pick obj
        self.pick2k = None

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

    def set_pick2k(self, pick2k):
        if isinstance(pick2k, Pick2KMsg):
            self.pick2k = pick2k

    def get_pick2k_msg(self):
        if self.pick2k is not None:
            msg = self.pick2k.generate_msg()
        else:
            msg = ''
        return msg


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
        
        x = np.arange(self.iON, self.iOFF)/self.sampling_rate + self.starttime.timestamp
        y = self.data
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
    
    def get_l1(self):
        norm = np.linalg.norm(self.res, ord=1)
        return norm
    
    L1 = property(get_l1)

    def get_l2(self):
        norm = np.linalg.norm(self.res, order=2)
        return norm
    
    L2 = property(get_l2)


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

        y = self.data
        x = np.arange(self.iON, self.iOFF)
        samples, probabilities = estimate_quantiles(x, y)
        self.probabilities = probabilities
        self.times = [self.tON + smp/self.sampling_rate for smp in samples]
        self.tmed = self.times[self.quantiles==0.5]
        self.pmed = self.probabilities[self.quantiles==0.5]

        self.pref_pick_time = self.tmed
        self.pref_pick_prob = self.pmed