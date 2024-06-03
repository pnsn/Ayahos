"""
:module:`~ayahos.wyrms.pickwyrm`
:auth: Nathan T. Stevens
:email: ntsteve (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose: Defines the PickWyrm class
"""
import logging, sys
from ayahos import Wyrm, DictStream, MLTrace
from ayahos.core.pick import Trigger, GaussTrigger, QuantTrigger, Pick2KMsg
from obspy.signal.trigger import trigger_onset
import numpy as np

Logger = logging.getLogger(__name__)

class PickWyrm(Wyrm):
    """
    This :class:`~ayahos.wyrms.wyrm.Wyrm` child-class provides the following alterations:

    :meth:`~ayahos.wyrms.pickwyrm.PickWyrm.unit_process` 
        1) scans across each entry in a :class:`~ayahos.core.dictstream.DictStream`
        2) runs :meth:`~obspy.signal.trigger.trigger_onset` on samples with fold :math:`\\geq` threshold
        3) converts triggers to :class:`~ayahos.core.pick.Pick2KMsg` objects
        4) appends :class:`~ayahos.core.pick.Pick2KMsg` objects to :attribute:`output`
        5) sets used samples' fold to 0 - WARNING: IN-PLACE OPERATION ON SOURCE DATA

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
                - 'max' -- represent triggers as :class:`~ayahos.core.trigger.Trigger` objects and use their `tmax` attribute for pick time and `pmax` for quality mapping
                - 'gau' -- represent triggers as :class:`~ayahos.core.trigger.GaussTrigger` objects and use their `mean` attribute for pick time and `scale` attribute for quality mapping
                - 'med' -- represent triggers as :class:`~ayahos.core.trigger.QuantTrigger` objects and use their `tmed` attribute for pick time and `pmed` attribute for quality mapping
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
            installation_id=255,
            trigger_level=0.2,
            trigger_bounds=[5, 200],
            fold_threshold=1,
            leading_mute=1800,
            pick_method='max',
            prob2qual_map={0:0.9, 1:0.7, 2:0.5, 3: 0.3, 4:0.2},
            phases_to_pick=['P','S'],
            phase2comp_map = {'P': 'Z', 'S':'E'},
            max_pulse_size=1,
            **trigger_opts):
        """Initialize a PickWyrm object

  
        """        
        # Inherit from Wyrm
        super().__init__(max_pulse_size=max_pulse_size)

        # Compatability check for module_id
        if isinstance(module_id, int):
            if 0 < module_id < 256:
                self.module_id = module_id
            else:
                raise ValueError('module_id must be \in [1, 255]')
        else:
            raise TypeError('module_id must be type int')
        
        # Compatability check for installation_id
        if isinstance(installation_id, int):
            if 0 < installation_id < 256:
                self.installation_id = installation_id
            else:
                raise ValueError
        elif installation_id is None:
            self.installation_id == 255
        else:
            raise TypeError

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
                        raise ValueError
                else:
                    raise TypeError
            else:
                raise ValueError
        else:
            raise TypeError
        
        if isinstance(leading_mute, int):
            if leading_mute >= 0:
                if leading_mute > 6000:
                    Logger.warning(f'leading_mute of {leading_mute}')
                    Logger.warning('may prevent any data from being assessed')
                    Logger.warning('(coded limit = 6000 samples)')
                self.leading_mute = leading_mute
                
            else:
                raise ValueError
        else:
            raise TypeError


        # Compatability Check for fold_threshold
        if isinstance(fold_threshold, (float, int)):
            if np.isfinite(fold_threshold):
                if fold_threshold > 0:
                    self.fold_threshold = fold_threshold
                else:
                    raise ValueError
            else:
                raise ValueError
        else:
            raise TypeError
                
        # Compatability check for pick_method
        if pick_method.lower() in ['max', 'gau', 'med']:
            self.pick_method = pick_method.lower()
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
        self._inner_index = 0


    def _should_this_iteration_run(self, input, input_size, iter_number):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.pickwyrm.PickWyrm`

        Signal early stopping (status = False) if:
         - type(input) != :class:`~ayahos.core.dictstream.DictStream
         - input_size == 0
         - iter_number < input_size

        I.e., If input is a DictStream, 
              there are MLTrace-like objects in input, 
              and the iteration counter is less than the number of MLTrace-like objects

        :param input: input DictStream
        :type input: ayahos.core.dictstream.DictStream
        :param input_size: number of MLTrace-like objects in dictstream (len(input))
        :type input_size: int
        :param iter_number: iteration number
        :type iter_number: int
        :returns:
            - **status** (*bool*) -- should this iteration be run?
        """
        status = False
        if input_size > 0:
            if isinstance(input, DictStream):
                if iter_number < input_size:
                    status = True
                    self._inner_index=iter_number
        return status
    
    def _unit_input_from_input(self, input):
        """
        POLYMORPHIC
        Last update with :class:`~ayahos.wyrms.pickwyrm.PickWyrm`

        Get a view of a single MLTrace-like object from input

        :param input: input DictStream
        :type input: ayahos.core.dictstream.DictStream
        :returns:
            - **unit_input** (*ayahos.core.mltrace.MLTrace) -- view of a single MLTrace-like object in input

        """        
        if isinstance(input, DictStream):
            unit_input = input[self._inner_index]
            return unit_input
        else:
            Logger.critical('Passing non-DictStream object to PickWyrm - not allowed')
            sys.exit(1)
    
    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last update with :class:`~ayahos.wyrms.pickwyrm.PickWyrm`

        Iterate across each MLTrace-like object in **unit_input** and conduct triggering and picking
        on traces with predicted probability labels that match self.phases_to_pick entries and do the 
        following:
            1) determine if there are any non-0-fold samples in the trace, if so, proceed
            1) with :meth:`~obspy.signal.trigger.trigger_onset`, trigger using self.trigger_level = thresh1 = thresh2
            2) for each trigger

        The right-most self.leading_mute samples are ignored in this process, as described in the
        :class:`~ayahos.wyrms.pickwyrm.PickWyrm` class docstring.



        :param unit_input: input view of DictStream
        :type unit_input: ayahos.core.dictstream.DictStream
        :returns:
            - **unit_output** (*)
        :rtype: _type_
        """        
        # Iterate across MLTrace objects
        for _id, _mlt in unit_input.traces.items():
            # Hard safety catch for required input type
            if not isinstance(_mlt, MLTrace):
                Logger.critical('Passing non-MLTrace object to PickWyrm - not allowed')
                sys.exit(1)
            # Filter for just prediction traces
            if _mlt.comp not in self.phases_to_pick:
                continue
            else:
                pass
            # Run triggering on whole trace (quick & cheap)
            crf = _mlt.data
            triggers = trigger_onset(
                crf,
                thresh1=self.trigger_level,
                thresh2=self.trigger_level,
                max_len=max(self.trigger_bounds),
                max_len_delete=True
            )
            # Get index of where muting begins
            mute_samp = _mlt.stats.npts - self.leading_mute
            # Iterate across triggers
            for trigger in triggers:
                trig_fold = _mlt.fold[self.iON: self.iOFF]
                # Reject if trigger touches the mute zone
                if trigger[0] > mute_samp or trigger[1] > mute_samp:
                    continue
                # Reject if too small
                if max(trigger) - min(trigger) < min(self.trigger_bounds):
                    continue
                # Reject if any fold is below threshold
                if any(trig_fold < self.fold_threshold):
                    continue
                
                # Convert trigger indices into Trigger object
                if self.pick_method == 'max':
                    trigger = Trigger(
                        _mlt,
                        trigger,
                        self.trigger_level,
                        **self.trigger_opts)
                    tpick = trigger.tmax
                    ppick = trigger.pmax
                elif self.pick_method == 'gau':
                    trigger = GaussTrigger(
                        _mlt,
                        trigger,
                        self.trigger_level,
                        **self.trigger_opts)
                    tpick = self.mean
                    ppick = self.scale
                elif self.pick_method == 'med':
                    trigger = QuantTrigger(
                        _mlt,
                        trigger,
                        self.trigger_level,
                        **self.trigger_opts
                    )
                    tpick = self.tmed
                    ppick = self.pmed

                else:
                    raise NotImplementedError('self.pick_method somehow got altered after __init__...')
                
                # Check index for sequence number
                if _id in self.index.keys():
                    if self.index[_id] < 9999:
                        self.index[_id] += 1
                    else:
                        self.index[_id] = 0
                        Logger.warning(f'resetting sequence index for {_id}')
                else:
                    self.index.update({_id:0})

                # Generate Pick2KMsg object
                pickmsg = Pick2KMsg(
                    mod_id = self.module_id,
                    inst_id = self.installation_id,
                    seq_no = self.index[_id],
                    net = _mlt.stats.network,
                    sta = _mlt.stats.station,
                    comp = self.get_comp(_mlt),
                    phase = _mlt.comp,
                    qual = self.get_qual(ppick),
                    time = tpick
                )
                trigger.set_pick2k(pickmsg)
                # Append to output
                self.output.append(trigger)
                nnew += 1
            # Zero out fold in assessed region
            _mlt.fold[:mute_samp] = 0
        unit_output = nnew
        return unit_output
    
    def _capture_unit_output(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.pickwyrm.PickWyrm`

        Return None - capture is handled in _unit_process

        :param unit_output: output of 
        :type unit_output: _type_
        :return: _description_
        :rtype: _type_
        """        
        return None
    
    
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
            