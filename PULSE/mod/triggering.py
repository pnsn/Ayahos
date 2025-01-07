import logging
from collections import deque

from obspy.signal.trigger import trigger_onset

from PULSE.mod.base import BaseMod
from PULSE.data.dictstream import DictStream
from PULSE.data.pick import Trigger

Logger = logging.getLogger(__name__)

class CRFTriggerMod(BaseMod):

    def __init__(
        self,
        thr_on=0.1,
        thr_off=None,
        pt_pad=None,
        fold_thr=1.,
        max_trig_len=1000,
        max_trig_len_delete=True,
        max_pulse_size=1000,
        maxlen=None,
        name=None
    ):
        """A PULSE module for conducting triggering on input collections
        of :class:`~.DictStream`-like objects to generate :class:`~.Trigger`
        objects. This class assumes that :class:`~.FoldTrace` objects contained
        in the input elements have already been converted into Characteristic
        Response Functions (i.e., strictly non-negative-valued **data** vectors).

        Modules to create these CRFs include:
         - :class:`~PULSE.mod.detecting.SBMMod` - SeisBench ML Detection Models
         - :class:`~PULSE.mod.detecting.ObspyCRFMod` - Obspy CRF Detection Methods

        :param thr_on: trigger onset threshold value, defaults to 0.1
        :type thr_on: float, optional
        :param thr_off: trigger offset threshold value, defaults to None
            Choice of "None" uses **thr_on** as the offset value
        :type thr_off: None or float, optional
        :param pt_pad: number of extra samples to include when generating, 
            :class:`~.Trigger` objects, defaults to None
        :type pt_pad: int or None, optional
        :param max_trig_len: Maximum trigger length in samples,
            defaults to 1000
        :type max_trig_len: int, optional
        :param max_trig_len_delete: Should overly-long triggers be deleted,
            defaults to True
        :type max_trig_len_delete: bool, optional
        :param max_pulse_size: number of iterations to conduct per call of
            :meth:`~.pulse`, defaults to 1000
        :type max_pulse_size: int, optional
        :param maxlen: maximum size of the **output** :class:`~.deque` object,
            defaults to None
        :type maxlen: int or None, optional
        :param name: Name suffix to apply to this TriggerMod's **name** attribute, defaults to None
        :type name: str or None, optional
        """        
        # Inherit from BaseMod
        super().__init__(max_pulse_size=max_pulse_size, maxlen=maxlen, name=name)
        # Compatability Check with PULSE.data.trigger.Trigger.__init__
        self.tikwargs = {}
        self.tokwargs = {}
        if isinstance(thr_on, (int, float)):
            thr_on = float(thr_on)
        else:
            raise TypeError('thr_on must be float-like')
        if thr_on <= 0:
            raise ValueError('thr_on must be positive-valued')
        if thr_off is None:
            thr_off = thr_on
        elif isinstance(thr_off, (int, float)):
            thr_off = float(thr_off)
        else:
            raise TypeError('thr_off must be float-like or NoneType')
        if thr_off <= 0:
            raise ValueError(f'thr_off must be positive-valued')
        
        if pt_pad is None:
            pt_pad = 0
        elif isinstance(pt_pad, (int, float)):
            pt_pad = int(pt_pad)
        else:
            raise TypeError('pt_pad must be int-like or NoneType')
        if pt_pad < 0:
            raise ValueError('pt_pad must be non-negative')
        # Assemble common Trigger.__init__ kwargs
        self.init_kwargs.update({'thr_on':thr_on,
                              'thr_off':thr_off,
                              'pt_pad': pt_pad})
        
        # Compatability checks for trigger_onset
        if not isinstance(max_trig_len_delete, bool):
            raise TypeError('max_trig_len_delete must be type bool')
        if isinstance(max_trig_len, (int, float)):
            max_trig_len = int(max_trig_len)
        else:
            raise TypeError('max_trig_len must be int-like')
        if max_trig_len <= 0:
            raise ValueError('max_trig_len must be positive')
        # Assemble common trigger_onset kwargs
        self.onset_kwargs.update({
            'thres1': thr_on,
            'thres2': thr_off,
            'max_len': max_trig_len,
            'max_len_delete': max_trig_len_delete})
        
        # Compatability check for fold_thr
        if isinstance(fold_thr, (int, float)):
            fold_thr = float(fold_thr)
        else:
            raise TypeError('fold_thr must be float-like')
        if fold_thr <= 0:
            raise ValueError('fold_thr must be positive')
        else:
            self.fold_thr = fold_thr

    def run_unit_process(self, unit_input: DictStream) -> deque:
        if not isinstance(unit_input, DictStream):
            raise TypeError('unit_input must be type DictStream')
        unit_output = deque()
        for ft in unit_input:
            triggers_raw = trigger_onset(ft.data, **self.onset_kwargs)
            for _t in triggers_raw:
                # Skip triggers with any under-informed samples
                # This catches any blinding as well as data gaps
                if any(ft.fold[_t[0]:_t[1]] < self.fold_thr):
                    continue
                self.init_kwargs.update(dict(zip(['pt_on','pt_off'], _t)))
                trigger = Trigger(source_trace=ft,
                                  **self.init_kwargs)
                unit_output.appendleft(trigger)
        return unit_output

    def put_unit_output(self, unit_output: deque) -> None:
        for _ in range(len(unit_output)):
            _t = unit_output.pop()
            self.output.appendleft(_t)
