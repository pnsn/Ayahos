"""
:module: PULSE.mod.picker
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose: This module provides a class definition for a picking unit-task module "PickMod"
    that conducts single-threshold triggering and maximum-value picking on :class:`~PULSE.data.foldtrace.FoldTrace`
    objects. Picks are documented using :class:`~obspy.core.event.Pick` objects with additional information
    on the trigger amplitude and duration saved in their **time_error** attributes.
"""

from collections import deque

import numpy as np
from obspy.core.event import Pick, WaveformStreamID, ResourceIdentifier, QuantityError
from obspy.signal.trigger import trigger_onset

from PULSE.data.dictstream import DictStream
from PULSE.data.foldtrace import FoldTrace
from PULSE.mod.base import BaseMod


class PickerMod(BaseMod):
    """
    A PULSE class for triggering and picking arrivals on :class:`~.FoldTrace`
    objects that have already been converted from waveforms to a characteristic
    response function (CRF). Triggering is currently conducted with a single
    threshold value and pick times are taken as the time of the maximum CRF
    value within each trigger.

    triggering uses :meth:`~obspy.signal.trigger.trigger_onset`

    Such CRFs include classic approaches like STA/LTA or predictions from
    SeisBench :class:`~seisbench.models.WaveformModel` prediction outputs.

    Output picks are formatted as :class:`~obspy.core.event.Pick` objects
    and use the component code of their originating :class:`~.FoldTrace`
    as their **phase_hint** attribute.

    Pick Object Definitions
    -----------------------
    **NOTE: A key re-definition here is the time_errors attribute**

    time - maximum CRF value timing
    time_errors - lower_uncertainty = delta seconds from `time` to trigger onset
                - upper_uncertainty = delta seconds from `time` to trigger offset
                - confidence_level = (1 - threshold/max_CRF_value)*100
    resource_id - model and weight names are saved in the path, if present
    method_id - model and weight names are saved in the path, if present
    waveform_id - NSLC code (drops Model and Weight identifiers), with component codes replaced with `?`
    phase_hint - component code of the source :class:`~.FoldTrace` object

    Parameters
    ----------
    :param min_fold: minimum fold value required for all
        samples in a trigger, defaults to 1.
    :type min_fold: float, optional
    :param threshold: triggering threshold, defaults to 0.3
    :type threshold: float, optional
    :param max_trig_len: maximum trigger length in samples,
        defaults to 1000
    :type max_trig_len: int, optional
    :param max_trig_len_delete: should triggers exceeding
        max_trig_len be deleted? Defaults to True
    :type max_trig_len_delete: bool, optional
    :param maxlen: maximum length of the **output** deque,
        defaults to None
        also see :class:`~collections.deque`
    :type maxlen: None or int, optional
    :param max_pulse_size: maximum number of iterations per
        call of :meth:`~.PickMod.pulse`, defaults to 1000
    :type max_pulse_size: int, optional
    :param name: optional suffix to add to the **name**
        attribute of this PickMod, defaults to None
    :type name: None or str, optional

    """

    def __init__(self,
                 min_fold=1.,
                 thr_on=0.3,
                 thr_off=None,
                 max_trig_len=1000,
                 max_trig_len_delete=True,
                 maxlen=None,
                 max_pulse_size=1000,
                 name=None):
        """Initialize a :class:`~.PickMod` object

        Parameters
        ----------
        :param min_fold: minimum fold value required for all
            samples in a trigger, defaults to 1.
        :type min_fold: float, optional
        :param threshold: triggering threshold, defaults to 0.3
        :type threshold: float, optional
        :param max_trig_len: maximum trigger length in samples,
            defaults to 1000
        :type max_trig_len: int, optional
        :param max_trig_len_delete: should triggers exceeding
            max_trig_len be deleted? Defaults to True
        :type max_trig_len_delete: bool, optional
        :param maxlen: maximum length of the **output** deque,
            defaults to None
            also see :class:`~collections.deque`
        :type maxlen: None or int, optional
        :param max_pulse_size: maximum number of iterations per
            call of :meth:`~.PickMod.pulse`, defaults to 1000
        :type max_pulse_size: int, optional
        :param name: optional suffix to add to the **name**
            attribute of this PickMod, defaults to None
        :type name: None or str, optional
        """        
        # Inherit from BaseMod
        super().__init__(max_pulse_size=max_pulse_size, maxlen=maxlen, name=name)
        # Compatability check for min_fold
        if not isinstance(min_fold, (float, int)):
            raise TypeError('min_fold must be float-like')
        elif min_fold < 0:
            raise ValueError('min_fold must be non-negative')
        else:
            self.min_fold = min_fold
        # Compatability check for threshold
        if not isinstance(threshold, (float, int)):
            raise TypeError('threshold must be float-like')
        elif threshold <= 0:
            raise ValueError('threshold must be a positive value')
        else:
            self.threshold = float(threshold)
        # Compatability check for max_trig_len
        if not isinstance(max_trig_len, int):
            raise TypeError('max_trig_len must be type int')
        elif max_trig_len < 1:
            raise ValueError('max_trig_len must be non-zero')
        else:
            self.max_trig_len = max_trig_len
        # Compatability check for max_trig_len_delete
        if not isinstance(max_trig_len_delete, bool):
            raise TypeError('max_trig_len_delete must be type bool')
        else:
            self.mtl_delete = max_trig_len_delete
        
    def run_unit_process(self, unit_input: DictStream) -> deque:
        """Take conduct triggering and picking on :class:`~.FoldTrace` objects
        in an input :class:`~.DictStream` object and convert triggers into
        :class:`~.Pick` objects with formatting as described in the :class:`~.PickMod`
        header

        POLYMORPHIC METHOD: last udpated with :class:`~.PickMod`

        :param unit_input: dictstream object containing FoldTraces that have already
            been converted into non-negative valued, characteristic response functions
        :type unit_input: DictStream
        :returns: **unit_output** (*collections.deque*) - collection of :class:`~.Pick` objects
        """      
        if not isinstance(unit_input, DictStream):
            raise TypeError('unit_input must be type DictStream')  
        # Create unit_output deque
        unit_output = deque()
        # Iterate across foldtraces in dictstream
        for ft in unit_input:
            # Generate prefix for pick resource_id
            pidr_prefix = f'smi:local/PULSE/{self.name}'
            # Generate id for method_id
            method_id = f'smi:local/PULSE/{self.name}'
            # Append model and stats metadata to resourceidentifier prefix/id
            if ft.stats.model != '':
                method_id += f'/{ft.stats.model}'
                pidr_prefix += f'/{ft.stats.model}'
            if ft.stats.weight != '':
                pidr_prefix += f'/{ft.stats.weight}'
                method_id += f'/{ft.stats.weight}'

            # Compse channel seed ID
            seedid = WaveformStreamID(seed_string=ft.id_keys['inst'] + '?')

            # Trigger on all data
            triggers = trigger_onset(ft.data,
                                     thres1=self.threshold,
                                     thres2=self.threshold,
                                     max_len=self.max_trig_len,
                                     max_len_delete=self.mtl_delete)
            # Iterate across triggers
            for _t in triggers:
                # Get subset data and fold
                _data = ft.data[_t[0]:_t[1]]
                _fold = ft.fold[_t[0]:_t[1]]
                # v0: Reject triggers with any blinded values
                if any(_fold < self.min_fold):
                    breakpoint()
                    continue
                # TODO: in future version, allow for partially blinded triggers?
                # Get relative index of the CRF max
                _dnM = np.argmax(_data)
                # Get the maximum value of the CRF
                _maxval = np.max(_data)
                # Get absolute index of the max in the CRF vector
                _nM = _t[0] + _dnM
                # Get the timestamp of the max value (pick), trigger onset(i), and trigger offset(f)
                tp = ft.stats.starttime + _nM*ft.stats.delta
                ti = ft.stats.starttime + _t[0]*ft.stats.delta
                tf = ft.stats.starttime + _t[1]*ft.stats.delta
                clevel = (1. - self.threshold/_maxval)*100
                # Get component code
                phz_hint = ft.stats.component
                # Compose Pick Object
                pick = Pick(resource_id=ResourceIdentifier(prefix=pidr_prefix),
                            waveform_id=seedid,
                            time=tp,
                            time_errors=QuantityError(lower_uncertainty=tp - ti,
                                                      upper_uncertainty=tf - tp,
                                                      confidence_level=clevel),
                            method_id=ResourceIdentifier(id=method_id),
                            evaluation_mode='automatic',
                            evaluation_status='preliminary',
                            phase_hint=phz_hint
                            )
                # Appendleft pick object to unit_output
                unit_output.appendleft(pick)
        # Return unit_output
        return unit_output
    
    def put_unit_output(self, unit_output: deque) -> None:
        """Extend this :class:`~.PickMod` object's **output** deque
        with contents of **unit_output**, ensuring that the left
        to right increase in object (processing) age is preserved.

        POLYMORPHIC METHOD: last updated with :class:`~.PickMod`

        :param unit_output: collection of :class:`~.Pick` objects
        :type unit_output: deque
        """
        for _e in range(len(unit_output)):
            _p = unit_output.pop()
            self.output.appendleft(_p)
