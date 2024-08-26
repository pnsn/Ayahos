"""
:module: PULSE.data.pick
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:
"""


import logging
import numpy as np
from obspy import UTCDateTime
from PULSE.data.mltrace import MLTrace
from PULSE.data.message import Logo
import seisbench.util.annotations as sua

Logger = logging.getLogger(__name__)

def index_to_datetime(trace,index):
    ts = trace.stats.starttime
    sr = trace.stats.sampling_rate
    ti = ts + index/sr
    return ti
    

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
        - **self.iON** (*int*) -- trigger ON index value
        - **self.tON** (*obspy.core.utcdatetime.UTCDateTime*) -- time of trigger ON
        - **self.iOFF** (*int*) -- trigger OF index value
        - **self.tOFF** (*obspy.core.utcdatetime.UTCDateTime*) -- time of trigger OFF
        - **self.trigger_level** (*float*) -- triggering threshold
        - **self.tmax** (*obspy.core.utcdatetime.UTCDateTime*) -- time maximum data value in source_trace.data[iON, iOFF]
        - **self.pmax** (*float*) -- maximum data value in source_trace.data[iON:iOFF]
    """
    def __init__(self, logo, pick_id, source_trace, trigger, trigger_level, padding_samples=0, chan_map={'P':'Z','S':'E'}):
        if isinstance(source_trace, MLTrace):
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



        # Map ID-related parameters
        self.phase = self.trace.comp
        self.trace.set_comp(self.chan_map[self.phase])
        self.id = self.trace.id


    def to_seisbench(self):
        """Provide a :class:`~seisbench.util.annotations.Pick` object representation of this trigger

        :return: **out** (*seisbench.util.annotations.Pick*) - seisbench pick object
        :rtype: _type_
        """        
        out = sua.Pick(
            self.trace.nslc,
            start_time=self.tON,
            end_time=self.tOFF,
            peak_time=self.pref_pick_time,
            peak_value=self.pref_pick_prob,
            phase=self.phase)
        return out

    def TYPE_PICK_SCNL(self, quality=4, fm='?', amp1=0, amp2=0, amp3=0):
        """
        Create a TYPE_PICK_SCNL message from this trigger

        http://www.earthwormcentral.org/documentation3/PROGRAMMER/location_codes/msgs_addloc.txt
        """
        if fm not in ['U','D','?']:
            raise ValueError('fm must be "U", "D", or "?"')
        if quality not in [0,1,2,3,4]:
            raise ValueError('quality must be an int-like in range [0,4], with 0 = best')
        for amp in [amp1, amp2, amp3]:
            # Check int
            if int(amp) != amp:
                raise ValueError
            # Check positive
            elif abs(amp) != amp:
                raise ValueError
            
        logo = f'8 {self.logo.MOD_ID} {self.logo.INST_ID} {self.pick_id}'
        scnl = self.trace.scnl
        fmq = f'{fm}{quality}'
        time = self.pref_pick_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        phz = self.phase
        amps = f'{amp1} {amp2} {amp3}'
        msg = f'{logo} {scnl} {fmq} {time} {phz} {amps}\n'
        return msg


class Pick(sua.Pick):
    """Generalized class for storing a pick extending the :class:`~seisbench.util.annotations.Pick` class
    and using elements of the :class:`~phaseworm.Pick` class from the SeisBench and PhaseWorm projects, respectively.

    """    
    def __init__(self, trace_id, start_time, end_time=None, peak_time=None, peak_value=None, phase=None, module_logo=None):
        super().__init__(
            trace_id = trace_id,
            start_time = start_time,
            end_time = end_time,
            peak_time = peak_time,
            peak_value = peak_value,
            phase = phase
        )
        if module_logo is None:
            self.logo = None
        elif isinstance(module_logo, Logo):
            self.logo = module_logo

    def TYPE_PICK2K(self):
        ilogo = self.logo.copy()['TYPE'] = 10


    
    def TYPE_PICK_SCNL(self):
        self.logo['TYPE'] = 8
        







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
        ids = f'{self.type:>3}{self.mod_id:>3}{self.inst_id:>3} {self.seq_no:>4}'
        codes = f'{self.sta:<5}{self.net:<2}{self.comp:<3}'
        attr = f'{self.pol:1}{self.qual:1}{self.phz:2}'
        idate = f'{self.time.year:4}{self.time.month:02}{self.time.day:02}'
        itime = f'{self.time.hour:02}{self.time.minute:02}{self.time.second}'
        csec = str(self.time.microsecond)[:2]
        amps = f'{self.amp1:8}{self.amp2:8}{self.amp3:8}'
        msg = f'{ids} {codes} {attr}{idate}{itime}{csec}{amps}\n'
        return msg
    

# class Pick_SCNL(object):
