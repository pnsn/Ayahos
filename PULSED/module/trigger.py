"""
:module: PULSED.module.trigger
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This conatins the class definition for a module that facilitates triggering on :class:`~PULSED.data.mltrace.MLTrace`
    objects containing characteristic response function (ML phase pick probability) time series and conversion of triggers
    into :class:`~PULSED.data.trigger.Trigger` objects.

Classes
-------
:class:`~PULSED.module.trigger.TriggerMod`
"""

import logging, sys
from PULSED.data.mltrace import MLTrace
from PULSED.data.mlstream import MLStream
from PULSED.module._base import _BaseMod
from PULSED.data.mltrigger import Trigger, GaussTrigger, QuantTrigger, Pick2KMsg
from obspy.signal.trigger import trigger_onset
import numpy as np

Logger = logging.getLogger(__name__)

class BuffTriggerMod(_BaseMod):
    """
    A unit module that conducts incremental triggering on :class:`~PULSED.data.mltracebuff.MLTraceBuff` objects
    using their *fold* attribute and window overlap lengths to only trigger on data that will not receive further
    updates from subsequent windows appended to the MLTraceBuff's. 

    The :meth:`~PULSED.module.trigger.BuffTriggerMod._unit_process` method for this class conducts the following steps:
        1) scans across each :class:`~PULSED.data.mltracebuff.MLTracBuff entry in a :class:`~PULSED.data.mlstream.MLStream`
        2) runs :meth:`~obspy.signal.trigger.trigger_onset` on samples with fold :math:`\\geq` **threshold**
        3) converts triggers to :class:`~PULSED.data.trigger.Trigger`-type objects (see **method** below)
        4) appends Trigger-type objects objects to this BuffTriggerMod's *output* attribute
        5) sets analyzed samples' fold in a given :class:`~PULSED.data.mltracebuff.MLTraceBuff` to 0
            WARNING: Step 5) is an in-place alteration to *fold* values 

    :param module_id: module number for output pick2kmsg objects
    :type module_id: int
    :param installation_id: earthworm installation ID number, defaults to 255
    :type installation_id: int, optional
    :param trigger_level: triggering threshold passed to :meth:`~obspy.signal.trigger.trigger_onset, defaults to 0.2
    :type trigger_level: float, optional
    :param trigger_bounds: minimum and maximum number of samples for a trigger
    :type trigger_bounds: list of int
    :param fold_threshold: minimum fold required for picking on prediction samples, defaults to 1
    :type fold_threshold: int, optional
    :param leading_mute: number of trailing samples (chronologically last) to ignore when triggering (should match window overlap), defaults to 1800
    :type leading_mute: int, optional
    :param pick_method: name of method to use for picking the arrival time, defaults to 'max'
            Supported:
                - 'max' -- represent triggers as :class:`~PULSED.data.trigger.Trigger` objects and use their `tmax` attribute for pick time and `pmax` for quality mapping
                - 'gau' -- represent triggers as :class:`~PULSED.data.trigger.GaussTrigger` objects and use their `mean` attribute for pick time and `scale` attribute for quality mapping
                - 'med' -- represent triggers as :class:`~PULSED.data.trigger.QuantTrigger` objects and use their `tmed` attribute for pick time and `pmed` attribute for quality mapping
    :type pick_method: str, optional
    :param prob2qual_map: mapping of pick quality (keys) to probabilities (values), defaults to {0:0.9, 1:0.7, 2:0.5, 3: 0.3, 4:0.2}
    :type prob2qual_map: dict, optional
    :param phases_to_pick: list of phase names (prediction labels) to pick, defaults to ['P','S']
    :type phases_to_pick: list, optional
    :param phase2comp_map: component names to assign to PICK2K messages for each phase, defaults to {'P': 'Z', 'S':'E'}
    :type phase2comp_map: dict, optional
    :param max_pulse_size: number of sweeps to conduct across, defaults to 1
    :type max_pulse_size: int, optional

    """

    def __init__(
            self,
            module_id,
            installation_id,
            trigger_level=0.2,
            trigger_bounds=[5, 200],
            fold_threshold=1,
            leading_mute=1800,
            pick_method='max',
            prob2qual_map={0:0.9, 1:0.7, 2:0.5, 3: 0.3, 4:0.2},
            phases_to_pick=['P','S'],
            phase2comp_map = {'P': 'Z', 'S':'E'},
            max_pulse_size=1,
            meta_memory=3600,
            report_period=False,
            **trigger_opts):
        """Initialize a BuffTriggerMod object

  
        """        
        # Inherit from Wyrm
        super().__init__(max_pulse_size=max_pulse_size,
                         meta_memory=meta_memory,
                         report_period=report_period)

        # Compatability check for module_id
        if isinstance(module_id, int):
            if 0 < module_id < 256:
                self.module_id = module_id
            else:
                self.Logger.critical('Invalid module_id value: must be in [1, 255]')
                sys.exit(1)
        else:
            self.Logger.critical('Invalid module_id type. Must be type int')
            sys.exit(1)
        
        # Compatability check for installation_id
        if isinstance(installation_id, int):
            if 0 < installation_id < 256:
                self.installation_id = installation_id
            else:
                self.Logger.critical(f'Invalid installtion_id value {installation_id} not in [1, 255]')
                sys.exit(10)
        elif installation_id is None:
            self.installation_id == 255
        else:
            self.Logger.critical(f'Invalid installation_id type {type(installation_id)}')
            sys.exit(1)

        # Compatability check for trigger_level
        if isinstance(trigger_level, float):
            if 0 < trigger_level < 1:
                self.trigger_level = trigger_level
            else:
                raise ValueError('trigger_level must be \in (0, 1)')
        else:
            raise TypeError('trigger_level must be type float')
        
        if isinstance(trigger_bounds, (list, tuple)):
            if len(trigger_bounds) == 2:
                if all(isinstance(t_, int) for t_ in trigger_bounds):
                    if all(t_ >= 0 for t_ in trigger_bounds):
                        self.trigger_bounds = trigger_bounds
                        self.trigger_bounds.sort
                    else:
                        self.Logger.critical('One or both trigger_bounds values are negative. Must be non-negative')
                        sys.exit(1)
                else:
                    self.Logger.critical('Not all trigger_bounds values are type int - must be type int')
                    sys.exit(1)
            else:
                self.Logger.critical('trigger_bounds must have precisely 2 elements')
                sys.exit(1)
        else:
            self.Logger.critical(f'trigger_bounds must be a list or tuple. Not {type(trigger_bounds)}')
            sys.exit(1)
        
        if isinstance(leading_mute, int):
            if leading_mute >= 0:
                if leading_mute > 6000:
                    self.Logger.warning(f'leading_mute of {leading_mute}')
                    self.Logger.warning('may prevent any data from being assessed')
                    self.Logger.warning('(coded limit = 6000 samples)')
                self.leading_mute = leading_mute
                
            else:
                self.Logger.critical(f'leading_mute must be non-negative')
                sys.exit(1)
        else:
            self.Logger.critical(f'leading_mute must be type int. Not type {type(leading_mute)}')
            sys.exit(1)

        # Compatability Check for fold_threshold
        if isinstance(fold_threshold, (float, int)):
            if np.isfinite(fold_threshold):
                if fold_threshold > 0:
                    self.fold_threshold = fold_threshold
                else:
                    self.Logger.critical('fold_threshold must be positive')
            else:
                self.Logger.critical('fold_threshold must be finite')
        else:
            self.Logger.critical('fold_threshold must be type float or int')
                
        # Compatability check for pick_method
        if pick_method.lower() in ['max', 'gau', 'med']:
            self.TriggerClass = {'max': Trigger,
                                 'gau': GaussTrigger,
                                 'med': QuantTrigger}[pick_method]
        else:
            raise ValueError(f'pick_method {pick_method} not supported')
        
        # Compatability check for prob2qual_map
        if isinstance(prob2qual_map, dict):
            if all(_k in prob2qual_map.keys() for _k in [0,1,2,3,4]):
                if all(prob2qual_map[i_] > prob2qual_map[i_+1] for i_ in range(4)):
                    self.prob2qual_map = prob2qual_map
                else:
                    raise ValueError('Quality values must be monotonic decreasing')
            else:
                raise KeyError('incomplete mapping, requires keys 0, 1, 2, 3, 4 (int values)')
        else:
            raise TypeError('prob2qual_map must be type dict')
        
        # Compatability check for phases_to_pick
        if isinstance(phases_to_pick, list):
            if all(isinstance(e, str) for e in phases_to_pick):
                self.phases_to_pick = phases_to_pick
            else:
                raise TypeError('phases_to_pick entries must all be type str')
        elif isinstance(phases_to_pick, str):
            self.phases_to_pick = [phases_to_pick]
        else:
            raise TypeError('phases_to_pick must be a single phase name string or a list of phase name strings')
        
        # Compatability check for phase2comp_map
        if isinstance(phase2comp_map, dict):
            if all(_k in self.phases_to_pick for _k in phase2comp_map.keys()):
                if all(isinstance(_v, str) for _v in phase2comp_map.values()):
                    if all(len(_v) == 1 for _v in phase2comp_map.values()):
                        self.phase2comp_map = phase2comp_map
                    else:
                        raise ValueError('All phase2comp_map values must be single character strings')
                else:
                    raise TypeError('All phase2comp_map values must be type str')
            else:
                raise('All phases_to_pick values must be phase2comp_map keys')
            
        # Capture kwargs
        self.trigger_opts = trigger_opts
        # Create index for tracking sequence numbers at each station
        self.index = {}
        # Create _inner_index for tracking the index of the last station analyzed
        self._next_index = 0
        self._last_key = None


    def _should_this_iteration_run(self, input, input_size, iter_number):
        """
        POLYMORPHIC
        Last updated with :class:`~PULSED.module.trigger.BuffTriggerMod`

        Signal early stopping (status = False) if:
         - type(input) != :class:`~PULSED.data.mlstream.MLStream
         - input_size == 0
         - iter_number < input_size

        I.e., If input is a MLStream, 
              there are MLTrace-like objects in input, 
              and the iteration counter is less than the number of MLTrace-like objects

        :param input: input MLStream
        :type input: PULSED.data.mlstream.MLStream
        :param input_size: number of MLTrace-like objects in mlstream (len(input))
        :type input_size: int
        :param iter_number: iteration number
        :type iter_number: int
        :returns:
            - **status** (*bool*) -- should this iteration be run?
        """
        status = False
        if input_size > 0:
            if isinstance(input, MLStream):
                if iter_number < input_size:
                    status = True
                    self._inner_index=iter_number
        return status
    
    def _unit_input_from_input(self, input):
        """
        POLYMORPHIC
        Last update with :class:`~PULSED.module.trigger.BuffTriggerMod`

        Get a view of a single MLTrace-like object from input

        :param input: input MLStream
        :type input: PULSED.data.mlstream.MLStream
        :returns:
            - **unit_input** (*PULSED.data.mltrace.MLTrace) -- view of a single MLTrace-like object in input

        """        
        if not isinstance(input, MLStream):
            Logger.critical('input for BuffTriggerMod.pulse must be type PULSED.data.mlstream.MLStream')
            sys.exit(1)
    
        # If the next item would fall outside length of input, wrap
        if len(input) == self._next_index:
            self._next_index = 0

        # Get MLTrace
        _mlt = input[self._next_index]

        # If MLTrace ID is in the index keys
        if _mlt.id in self.index.keys():
            # Pass as unit_output (already passed checks below)
            unit_output = _mlt
            # Increment up _next_index
            self._next_index += 1
            # Update _last_key
            self._last_key = _mlt.id
        # If this is an unrecognized ID where the component (predicted label)
        # is in the phases_to_pick list
        elif _mlt.comp in self.phases_to_pick:
            # Create new index entry
            self.index.update({_mlt.id: 0})
            # Pass as unit_output
            unit_output = _mlt
            # Increment up _next_index
            self._next_index += 1
            # Update _last_key
            self._last_key = _mlt.id
        # If unrecognized ID and label is not a phase to pick
        else:
            unit_output = _mlt
            self._next_index += 1

        return unit_output

    
    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last update with :class:`~PULSED.module.trigger.BuffTriggerMod`

        Iterate across each MLTrace-like object in **unit_input** and conduct triggering and picking
        on traces with predicted probability labels that match self.phases_to_pick entries and do the 
        following:
            1) determine if there are any non-0-fold samples in the trace, if so, proceed
            1) with :meth:`~obspy.signal.trigger.trigger_onset`, trigger using self.trigger_level = thresh1 = thresh2
            2) for each trigger

        The right-most self.leading_mute samples are ignored in this process, as described in the
        :class:`~PULSED.module.trigger.BuffTriggerMod` class docstring.



        :param unit_input: input view of MLStream
        :type unit_input: PULSED.data.mlstream.MLStream
        :returns:
            - **unit_output** (*)
        :rtype: _type_
        """
        # initialize unit_output
        unit_output = []
        # If unit_input ID was rejected due to mismatch label
        if unit_input.id != self._last_key:
            Logger.debug(f'Skipping mltrace ID {unit_input.id} - component code rejected')
        # If unit_input passed, but no data have non-0 fold
        elif not any(unit_input.fold[:-self.leading_mute] > 0):
            Logger.debug(f'Skipping mltrace ID {unit_input.id} - no non-zero, non-muted fold samples')
        # Otherwise, proceed to triggering
        else:
            # Run Triggering
            triggers = trigger_onset(
                unit_input.data,
                thres1=self.trigger_level,
                thres2=self.trigger_level,
                max_len=max(self.trigger_bounds),
                max_len_delete=True
            )
            # Get index where muting begins
            mute_index = unit_input.stats.npts - self.leading_mute
            # Filter for additional trigger consideration requirements
            for _t, trigger in enumerate(triggers):
                # Skip trigger if any sample is in the mute zone
                if trigger[0] > mute_index or trigger[1] > mute_index:
                    Logger.debug(f'Skipping trigger {_t}/{len(triggers)} - touches mute area')
                    continue
                # Skip trigger if it is too short (too-long triggers should be handled by trigger_onset)
                elif trigger[1] - trigger[0] < min(self.trigger_bounds):
                    Logger.debug(f'Skipping trigger {_t}/{len(triggers)} - trigger duration too small')
                    continue
                # Skip trigger if any samples in the trigger have fold below the specified threshold
                elif any(unit_input.fold[trigger[0]:trigger[1]] < self.fold_threshold):
                    Logger.debug(f'Skipping trigger {_t}/{len(triggers)} - insufficiently high fold')
                    continue
                # Continue with trigger object conversion otherise
                else:
                    # Convert into Specified TriggerClass
                    trigger = self.TriggerClass(
                        unit_input,
                        trigger,
                        self.trigger_level,
                        **self.trigger_opts)
                    
                    # Generate Pick2KMsg object
                    pickmsg = Pick2KMsg(
                        mod_id = self.module_id,
                        inst_id = self.installation_id,
                        seq_no = self.index[self._last_key],
                        net = unit_input.stats.network,
                        sta = unit_input.stats.station,
                        comp = self.get_comp(unit_input),
                        phz = unit_input.comp,
                        qual = self.get_qual(trigger.pref_pick_prob),
                        time = trigger.pref_pick_time)
                    # Attach Pick2KMsg message to trigger
                    trigger.set_pick2k(pickmsg)
                    # Append pick/trigger object to unit_output
                    unit_output.append(trigger)
                    # Increment sub-index for seq_no
                    if self.index[unit_input.id] == 9999:
                        self.index[unit_input.id] = 0
                    else:
                        self.index[unit_input.id] += 1
        return unit_output


    
    def _capture_unit_output(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class:`~PULSED.module.trigger.BuffTriggerMod`

        Return None - capture is handled in _unit_process

        :param unit_output: list-like set of :class:`~PULSED.data.pick.Trigger`-type objects
        :type unit_output: list

        """   
        if len(unit_output) > 0:
            if all(isinstance(e, Trigger) for e in unit_output):
                self.output += unit_output
            else:
                Logger.critical('unit_process is outputting non-Trigger-type objects')
                sys.exit(1)

    
    def get_comp(self, mlt):
        if isinstance(mlt, MLTrace):
            comp = mlt.stats.channel[:2] + self.phase2comp_map[mlt.comp]
        else:
            raise TypeError
        return comp

    def get_qual(self, value):
        for _k, _v in self.prob2qual_map.items():
            if value > _v:
                qual = _k
                break
            else:
                qual = 4
        return qual