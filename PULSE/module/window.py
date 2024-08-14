"""
:module: PULSE.module.window
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This conatins the class definition for a module that facilitates data window generation

Classes
-------
:class:`~PULSE.module.window.WindowMod`
"""
import sys
import seisbench.models as sbm
from collections import deque
from obspy import UTCDateTime
from PULSE.data.mltrace import MLTrace
from PULSE.data.dictstream import DictStream
from PULSE.data.window import Window
from PULSE.module._base import _BaseMod


class WindowMod(_BaseMod):
    """
    The WindowMod class takes windowing information from an input seisbench.models.WaveformModel object and user-defined component
    mapping and data completeness metrics and provides a pulse method that iterates across entries in an input DictStream object and
    generates Window copies of sampled data that pass data completeness requirements. 

    :param component_aliases: aliases for standard component names used in SeisBench, defaults to {"Z": "Z3", "N": "N1", "E": "E2"}
    :type component_aliases: dict, optional
    :param reference_component: reference component used for channel fill rules, defaults to 'Z'
    :type reference_component: str, optional
    :param reference_completeness_threshold: completeness fraction needed for reference component traces to pass QC, defaults to 0.95
    :type reference_completeness_threshold: float, optional
    :param model_name: name of the ML model the windows are intended for, defaults to "EQTransformer"
    :type model_name: str, optional
    :param reference_sampling_rate: target sampling rate after preprocessing, defaults to 100.0
    :type reference_sampling_rate: float, optional
    :param reference_npts: target number of samples per window after preprocessing, defaults to 6000
    :type reference_npts: int, optional
    :param reference_overlap: target number of overlapping samples between windows after preprocessing, defaults to 1800
    :type reference_overlap: int, optional
    :param fnfilter: fnmatch filter string to use for subsetting channel inputs, default is None
    :type fnfilter: None or str
        also see :meth:`~PULSE.data.dictstream.DictStream.fnselect`
    :param pulse_type: style of running pulse, defaults to 'network'
        Supported values:
            'network' - attempt to create one window from each instrument per pulse
            vv NOT IMPLEMENTED vv
            'site' - create a window/windows for a given site per pulse iteration, tracking
                    which site last generated windowed data. Permits for list wrapping  
            'instrument' - under development       
    :type pulse_type: str, optional
    :param max_pulse_size: maximum number of iterations for a given pulse, defaults to 1
    :type max_pulse_size: int, optional
    :param **options: key-word argument collector to pass to Trace.trim() via TraceBuffer.trim_copy() methods
    :type **options: kwargs
    """

    def __init__(
        self,
        component_aliases={"Z": "Z3", "N": "N1", "E": "E2"},
        reference_component='Z',
        reference_completeness_threshold=0.95,
        other_completeness_threshold=0.8,
        model_name="EQTransformer",
        reference_sampling_rate=100.0,
        reference_npts=6000,
        reference_overlap=1800,
        fnfilter=None,
        pulse_type='network',
        stance='patient',
        stagger_start_sec=1,
        max_pulse_size=1,
        meta_memory=3600,
        report_period=False,
        max_output_size=1e9,
        **options):
        """Initialize a WindowMod object that samples a Modictstreamream of TraceBuffer
        objects and generates Window copies of windowed data if a reference
        component for a given instrument in the Modictstreamream is present and has
        sufficient data to meet windowing requirements

        :param component_aliases: aliases for standard component names used in SeisBench, defaults to {"Z": "Z3", "N": "N1", "E": "E2"}
            values in the component_aliases should be an iterable that produces single character strings when iterated
        :type component_aliases: dict, optional
        :param reference_component: reference component code to use for channel fill rules, defaults to 'Z'.
            Must match a key in **component_aliases**
        :type reference_component: str, optional
        :param reference_completeness_threshold: completeness fraction needed for reference component traces to pass QC, defaults to 0.95
            Specified value must be :math:`\in [0,1]`. This value is used when assessing if a window is ready to be generated
            Also see
                :meth:`~PULSE.module.window.WindowMod.update_window_tracker`
                :meth:`~PULSE.data.window.Window.apply_fill_rule`
        :type reference_completeness_threshold: float, optional
        :param other_completeness_threshold: completeness fraction needed for "other" component traces to pass QC, defaults to 0.8
            Specified value must be :math:`\in [0,1]`. This value is only used to populate **Window.stats** attributes
            Also see :meth:`~PULSE.data.window.Window.__init__`
        :type other_completeness_threshold: float, optional
        :param model_name: name of the ML model the windows are intended for, defaults to "EQTransformer"
        :type model_name: str, optional
        :param reference_sampling_rate: target sampling rate after preprocessing, defaults to 100.0
            Used to determine the endtime for window generation and window advance starttimes for sequential windows
        :type reference_sampling_rate: float, optional
        :param reference_npts: target number of samples per window after preprocessing, defaults to 6000
            Used to determine the endtime for window generation
        :type reference_npts: int, optional
        :param reference_overlap: target number of overlapping samples between windows after preprocessing, defaults to 1800
            Used to determine window starttimes for sequential windows
        :type reference_overlap: int, optional
        :param fnfilter: fnmatch filter string to use for subsetting channel inputs, default is None.
            also see :meth:`~PULSE.data.dictstream.DictStream.fnselect`
        :type fnfilter: NoneType or str
        :param pulse_type: style of running pulse, defaults to 'network'
            Supported values:
                'network' - attempt to create one window from each instrument per iteration in a call of :meth:`~PULSE.module.window.WindowMod.pulse`
                TODO: Implement these or remove this feature
                vv NOT IMPLEMENTED vv
                'site' - create a window/windows for a given site per pulse iteration, tracking
                        which site last generated windowed data. Permits for list wrapping  
                'instrument' - under development       
        :type pulse_type: str, optional
        :param stance: window generation rule, defaults to 'patient'
            Supported values:
                'patient' - windows are not flagged as ready until the endtime of a reference component trace 
                    exceeds the end of the window to be generated
                'eager' - windows are flagged as ready as soon as sufficient data are present in the reference component trace,
                    even if the endtime of the trace does not exceed the endtime of the window to be generated.
                    This can produce windows sooner in exchange for potentially omitting trailing data.
        :param max_pulse_size: maximum number of iterations for a given pulse, defaults to 1
        :type max_pulse_size: int, optional
        :param **options: key-word argument collector to pass to :meth:`~PULSE.data.mltrace.MLTrace.trim_copy`
        :type **options: kwargs
        """
        # Initialize/inherit from _BaseMod
        super().__init__(max_pulse_size=max_pulse_size,
                         meta_memory=meta_memory,
                         report_period=report_period,
                         max_output_size=max_output_size)

        # Compatability checks for `component_aliases`
        if not isinstance(component_aliases, dict):
            self.raise_log('TypeError', 'component_aliases must be type dict')
        elif not all(isinstance(_v, str) and _k in _v for _k, _v in component_aliases.items()):
            self.raise_log('SyntaxError','component_aliases values must be type str and include the key value')
        else:
            self.aliases = component_aliases

        # Compatability check for `reference_component`
        if reference_component in self.aliases.keys():
            refc = reference_component
        else:
            self.raise_log('ValueError','reference_component does not appear as a key in component_aliases')
        
        # Compatability check for `reference_completeness_threshold`
        if isinstance(reference_completeness_threshold, (float, int)):
            if 0 <= reference_completeness_threshold <= 1:
                reft = reference_completeness_threshold
            else:
                self.raise_log('ValueError','reference_completeness_threshold out of valid range [0, 1]')
        else:
            self.raise_log('TypeError','reference_completeness_threshold must be type int or float')
        
        # Compatability check for `other_completeness_threshold`
        if isinstance(other_completeness_threshold, (float, int)):
            if 0 <= other_completeness_threshold <= 1:
                othert = other_completeness_threshold
            else:
                self.raise_log('ValueError','other_completeness_threshold out of range')
                sys.exit(1)
        else:
            self.raise_log('TypeError','other_completeness_threshold out of range')
            sys.exit(1)
        
        # Compatability check for `model_name`
        if not isinstance(model_name, str):
            self.raise_log('TypeError','model_name must be type str')
            sys.exit(1)
        else:
            self.model_name = model_name

        # Compatability check for `reference_sampling_rate`
        if isinstance(reference_sampling_rate, (int, float)):
            if 0 < reference_sampling_rate < 1e9:
                refsr = reference_sampling_rate
            else:
                self.raise_log('ValueError','reference_sampling_rate must be in the range (0, 1e9)')
                sys.exit(1)
        else:
            self.raise_log('TypeError','reference_sampling_rate must be float-like')
            sys.exit(1)
        
        # Compatability check for `reference_npts`
        if isinstance(reference_npts, int):
            if 0 < reference_npts < 1e9:
                refn = reference_npts
            else:
                self.raise_log('ValueError','reference_npts must be in the range (0, 1e9)')
                sys.exit(1)
        else:
            self.raise_log('TypeError','reference_npts must be type int')
            sys.exit(1)
        
        # Compatability check for `reference_overlap`
        if isinstance(reference_overlap, int):
            if 0 <= reference_overlap < refn:
                refo = reference_overlap
            elif reference_overlap >= refn:
                self.raise_log('ValueError','reference_overlap must be less than reference_npts')
                sys.exit(1)
            elif reference_overlap < 0:
                self.raise_log('ValueError','reference_overlap must be non-negative')
                sys.exit(1)
        else:
            self.raise_log('TypeError','reference_overlap must be type int')
            sys.exit(1)

        # Compatability check for `fnfilter`
        if isinstance(fnfilter, str):
            self.fnfilter = fnfilter
        elif fnfilter is None:
            self.fnfilter = '*'
        else:
            self.raise_log('TypeError','fnfilter must be type str or NoneType')
            sys.exit(1)

        # Compatability check for `stance`
        if isinstance(stance, str):
            if stance.lower() in ['patient','eager']:
                self.stance = stance.lower()
            else:
                self.raise_log('ValueError','')
        else:
            self.raise_log('TypeError','')
            sys.exit(1)

        # Compatability check for pulse_type
        if pulse_type.lower() in ['network']: #,'site','instrument']:
            self.pulse_type = pulse_type.lower()
            self.next_code = None
        else:
            self.raise_log(f'ValueError','pulse_type "{pulse_type}" not supported.')
            sys.exit(1)
        # Compose reference dictionary
        self.ref = {
            'sampling_rate': refsr,
            'overlap': refo,
            'npts': refn,
            'component': refc,
            'thresholds': {'ref':reft, 'other': othert}
        }
         # Set Defaults and Derived Attributes
        # Calculate window length, window advance, and blinding size in seconds
        self.window_sec = (self.ref['npts'] - 1)/self.ref['sampling_rate']
        self.advance_npts = self.ref['npts'] - self.ref['overlap']
        self.advance_sec = (self.ref['npts'] - self.ref['overlap'])/self.ref['sampling_rate']
        self.options = options
        # Create dict for holding instrument window starttime values
        self.window_tracker = {}
        # Create holder stagger_start
        if isinstance(stagger_start_sec, (int, float)):
            if 0 <= stagger_start_sec < self.window_sec:
                self.stagger_start_sec = float(stagger_start_sec)
            else:
                self.raise_log('ValueError','stagger_start_sec must be in the range [0, self.window_sec)')
                sys.exit(1)
        else:
            self.raise_log('TypeError','stagger_start_sec must be float-like')
            sys.exit(1)
        self.next_available_start = None
        # self.Logger.info('WindowMod initialized!')

    #######################################
    # Parameterization Convenience Method #
    #######################################
        
    def update_from_seisbench(self, model):
        """
        Helper method for (re)setting the window-defining attributes for this
        WindowMod object from a seisbench.models.WaveformModel object:

            self.model_name = model.name
            self.ref['sampling_rate'] = model.sampling_rate
            self.ref['npts'] = model.in_samples
            self.ref['overlap'] = model._annotate_args['overlap'][1]
            self.reference_completeness_threshold = (model.in_samples - model._blinding[1][0])/model.in_samples
        
        :param model: seisbench model to scrape windowing parameters from
        :type model: seisbench.models.WaveformModel
        """
        if not isinstance(model, sbm.WaveformModel):
            self.raise_log('TypeError',''
            sys.exit(1)
        elif model.name != 'WaveformModel':
            if model.sampling_rate is not None:
                self.ref.update({'sampling_rate': model.sampling_rate})
            if model.in_samples is not None:
                self.ref.update({'npts': model.in_samples})
            self.ref.update({'overlap': model._annotate_args['overlap'][1]})
            self.model_name = model.name
        
            # self.ref['threshold']['ref'] = (model.in_samples - model._annotate_args['blinding'][1][0])/model.in_samples

        else:
            self.raise_log('TypeError','seisbench.models.WaveformModel base class does not provide the necessary update information')
            sys.exit(1)

    #################################
    # PULSE POLYMORPHIC SUBROUTINES #
    #################################
    def pulse(self, input):
        """Explicit definition of inherited pulse method from :class:`~PULSE.module._base._BaseMod`
        included for readability and updated documentation

        This pulse method iterates across unique site, instrument, model, and ML model weight code keys for a
        :class:`~PULSE.data.dictstream.DictStream` object containing :class:`~PULSE.data.mltrace.MLTrace` or 
        :class:`~PULSE.data.mltracebuff.MLTraceBuff` objects and uses their metadata to assesses if there are
        sufficient data to generate a new :class:`~PULSE.data.window.Window` object. If a new Window object is
        generated, the data and metadata are copied from the input MLTrace object(s) and metadata regarding window
        generation are updated to reflect the next expected window.

        For more information on this method's behaviors, see :meth:`~PULSE.module.window.WindowMod._unit_process`

        :param input: 
        :type input: _type_
        :return: _description_
        :rtype: _type_
        """        
        output = super().pulse(input)
        return output


    def _should_this_iteration_run(self, input, input_measure, iter_number):
        """
        POLYMORPHIC METHOD
        Last updated with :class:`~PULSE.module.window.WindowMod`

        unconditional pass - early stopping is handled by
        :meth:`~PULSE.module.window.WindowMod._should_next_iteration_run`

        :param input: standard input
        :type input: PULSE.data.dictstream.DictStream
        :param iter_number: iteration number, unused
        :type iter_number: int
        :return status: should iterations continue in pulse, always True
        :rtype: bool
        """        
        status = True
        return status
    
    def _unit_input_from_input(self, input):
        """
        POLYMORPHIC METHOD
        Last updated with :class:`~PULSE.module.window.WindowMod

        unit_input is a view of the input and a sanity check is done to make sure
        that `input` is type :class:`~PULSE.data.dictstream.DictStream`

        :param input: input to :meth:`~PULSE.module.window.WindowMod.pulse`
        :type input: PULSE.data.dictstream.DictStream
        :return unit_input: view of input to :meth:`~PULSE.module.window.WindowMod.pulse`
        :rtype: PULSE.data.dictstream.DictStream
        """
        if isinstance(input, DictStream):
            unit_input = input
            return unit_input
        else:
            self.raise_log('TypeError',' input is not type DictStream')
            sys.exit(1)
    
    def _unit_process(self, unit_input):
        """
        POLYMORPHIC METHOD
        Last updated with :class:`~PULSE.module.window.WindowMod`
        Conducts the following steps
        1)  Scans across all MLTrace(Buff) entries in a :class:`~PULSE.data.dictstream.DictStream`
            and update entries in the **WindowMod.window_tracker** attribute using their metadata.
                uses :meth:`~PULSE.module.window.WindowMod.__update_window_tracker`

        2)  nested iteration across site codes, instrument codes, and model codes in **WindowMod.window_tracker**
            to see if a given site/instrument/model combination in the tracker is flagged as 'ready'.

        3)  for 'ready' codes, generate a :class:`~PULSE.data.window.Window` object from the 
            matching entries

        Update the window_tracker with the contents of unit_input and then
        generate one window for each instrument in unit_input that has a 'ready'
        flag in window_tracker

        Newly generated windows are appended to WindowMod.output

        :param unit_input: view of a DictStream containing MLTrace(Buff)
        :type unit_input: PULSE.data.dictstream.DictStream
        """        
        unit_output = deque()
        # Update window tracker
        self._update_window_tracker(unit_input)
        # Conduct network-wide pulse
        # Iterate across site-level dictionary entries
        for site_key, site_dict in self.window_tracker.items():
            # Iterate across inst-level dictionary entries
            for inst_key, inst_dict in site_dict.items():
                # skip the t0:UTCDateTime entry
                if isinstance(inst_dict, dict):
                    # Iterate across mod-level dictionary entries
                    for mod_key, value in inst_dict.items():
                        # If this instrument record is ready to produce a window
                        if value['ready']:
                            fnstring = f'{site_key}.{inst_key}?.{mod_key}'
                            # Logger.info(f'generating window for {fnstring}')
                            next_window_ti = value['ti']
                            next_window_tf = next_window_ti + self.window_sec
                            # Subset to all traces for this instrument
                            _dictstream = unit_input.fnselect(fnstring)
                            # Create copies of trimmed views of Trace(Buffers)
                            traces = []
                            # Iterate over mltracebuff's
                            for _mltb in _dictstream:
                                # Create copies
                                mlt = _mltb.view_copy(
                                    starttime = next_window_ti,
                                    endtime = next_window_tf,
                                    pad=True,
                                    fill_value=None
                                )
                                # Update stats with the model name relevant to the windowing
                                mlt.stats.model = self.model_name
                                # Append mlt to traces
                                traces.append(mlt)
                            # Populate window traces and metadata
                            window = Window(
                                traces = traces,
                                header = {'reference_starttime': next_window_ti,
                                          'reference_sampling_rate': self.ref['sampling_rate'],
                                          'reference_npts': self.ref['npts'],
                                          'thresholds': self.ref['thresholds'],
                                          'processing': [[self.__name__(full=False), UTCDateTime()]]},
                                ref_component=self.ref['component']
                            )
                            # Append window to output
                            unit_output.append(window)
                            # Advance window start time in window_tracker
                            old_ti = self.window_tracker[site_key][inst_key][mod_key]['ti']
                            self.window_tracker[site_key][inst_key][mod_key]['ti'] += self.advance_sec
                            new_ti = self.window_tracker[site_key][inst_key][mod_key]['ti']
                            self.Logger.debug(f'New window for {window.stats.common_id} at {old_ti} += next at {new_ti}')
                            # Set ready flag to false for this site
                            self.window_tracker[site_key][inst_key][mod_key].update({'ready': False})
        return unit_output

                        
    def _capture_unit_output(self, unit_output):
        """
        POLYMORPHIC METHOD
        Last updated with :class:`~PULSE.module.window.WindowMod`

        Attaches the unit_output to **WindowMod.output** using :meth:`~collections.deque.__iadd__`
        and assesses if **WindowMod.output** is oversized (i.e., len(self.output) > self.max_output_size).

        If oversized, entries in **WindowMod.output** are removed using :meth:`~collections.deque.popleft`

        :param unit_out: deque of new :class:`~PULSE.data.window.Window` objects generated in _unit_process
        :type unit_out: collections.deque
        """  
        self.output += unit_output
        extra = len(self.output) - self.max_output_size
        # if extra > 0:
        #     Logger.warning(f'{self.__class__.__name__} object reached max_output_size. Deleting {extra} oldest values')
        while len(self.output) > self.max_output_size:
            self.output.popleft()


    def _should_next_iteration_run(self, unit_output):
        """
        POLYMORPHIC METHOD
        Last updated with :class:`~PULSE.module.window.WindowMod`

        Assess if any new windows were generated by the last call of :meth:`~PULSE.module.window.WindowMod._unit_process`
        and signal continuation of :meth:`~PULSE.module.window.WindowMod.pulse` iterations if at least one new window was generated
        I.e., len(unit_output) > 0 --> status = True

        :param unit_output: output from last call of _unit_process
        :type unit_output: collections.deque
        :return: status (*bool*) -- continuation signal
        """
        if len(unit_output) > 0:
            status = True
        else:
            status = False
        return status

    def update_window_tracker(self, dictstream):
        """
        Core subroutine for :meth:`~PULSE.module.window.Window._unit_process` used to update windowing metadata
        held in **WindowMod.window_tracker**

        Scans across MLTrace(Buff)s in an input DictStream with a component code matching the self.ref['component']
        attribute of this WindowMod and populate new branches in the self.window_index attribute if new site.inst
        (Network.Station.Location.BandInstrument{ref}) Traces are found

        :: INPUT ::
        :param dictstream: DictStream object containing :class:`~PULSE.data.trace.Trace`-type objects
        :type dictstream: PULSE.data.dictstream.DictStream
        """
        # Subset using fnfilter
        fdictstream = dictstream.fnselect(self.fnfilter)
        # Iterate across subset
        for mltrace in fdictstream.traces.values():
            if not isinstance(mltrace, MLTrace):
                self.raise_log('TypeError','this build of WindowMod only works with PULSE.data.mltrace.MLTrace objects')
                sys.exit(1)
            # Get site, instrument, mod, and component codes from Trace
            site = mltrace.site
            inst = mltrace.inst
            comp = mltrace.comp
            mod = mltrace.mod
            ### WINDOW TRACKER NEW BRANCH SECTION ###
            # First, check if this is a reference component
            if comp not in self.ref['component']:
                # If not a match, continue to next mltrace
                continue
            # Otherwise proceed
            else:
                pass
            # If new site in window_tracker
            if site not in self.window_tracker.keys():
                # TODO: WIP
                # # Check if this is the very first entry
                # if self.window_tracker == {}:
                #     self.next_available_start = mltrace.stats.starttime
                # else:
                #     # Get number of windows difference between trace start and next_available starttime
                #     dwindow = (mltrace.stats.starttime - self.next_available_start)//self.window_sec
                #     if dwindow > 0:
                #         self.next_available_start += dwindow*self.window_sec
                # Populate t0's
                self.window_tracker.update({site: 
                                                {inst: 
                                                    {mod: 
                                                        {'ti': mltrace.stats.starttime,
                                                         'ref': mltrace.id,
                                                         'ready': False}},
                                                't0': mltrace.stats.starttime}})
                #self.Logger.info(f'Added buffer tree for {site} - triggered by {mltrace.id}')
            # If site is in window_tracker
            else:
                # If inst is not in this site subdictionary
                if inst not in self.window_tracker[site].keys():
                    self.window_tracker[site].update({inst:
                                                        {mod:
                                                            {'ti': self.window_tracker[site]['t0'],
                                                             'ref': mltrace.id,
                                                             'ready': False}}})
                    #self.Logger.info(f'Added buffer branch {inst} to {site} tree - triggered by {mltrace.id}')
                # If inst is in this site subdictionary
                else:
                    # If mod is not in this inst sub-subdictionary
                    if mod not in self.window_tracker[site][inst].keys():
                        self.window_tracker[site][inst].update({mod:
                                                                    {'ti': self.window_tracker[site]['t0'],
                                                                     'ref': mltrace.id,
                                                                     'ready': False}})
                        
                        
            ### WINDOW TRACKER TIME INDEXING/WINDOWING STATUS CHECK SECTION ###   
            # Get window edge times
            next_window_ti = self.window_tracker[site][inst][mod]['ti']
            next_window_tf = next_window_ti + self.window_sec
            # If the endpoint exists in the buffer
            status = False
            # If eager, just need the window endtime in the scope of the trace
            if self.stance == 'eager':
                status = next_window_tf <= mltrace.stats.endtime
            # If patient, require at least one non-zero-fold sample past the end of the window
            elif self.stance == 'patient':
                if next_window_tf <= mltrace.stats.endtime:
                    _, fold_view = mltrace.get_subset_view(starttime=next_window_tf)
                    status = any(fold_view > 0)

            # If the mltrace has reached or exceeded the endpoint of the next window
            if status:
                # Get valid fraction for proposed window in Trace
                fv = mltrace.get_fvalid_subset(starttime=next_window_ti, endtime=next_window_tf)
                # If threshold passes
                if fv >= self.ref['thresholds']['ref']:
                    # set (window) ready flag to True
                    self.window_tracker[site][inst][mod].update({'ready': True})
                    # And continue to next mltrace
                    continue
                # If threshold fails
                else:
                    # If the window was previously approved, recind approval
                    if self.window_tracker[site][inst][mod]['ready']:
                        self.window_tracker[site][inst][mod].update({'ready': False})

                    # If data start after the proposed window, increment window index to catch up
                    if next_window_ti < mltrace.stats.starttime:
                        # Determine the number of advances that should be applied to catch up
                        nadv = 1 + (mltrace.stats.starttime - next_window_ti)//self.advance_sec
                        # Apply advance
                        self.window_tracker[site][inst][mod]['ti'] += nadv*self.advance_sec
                        next_window_ti += nadv*self.advance_sec
                        next_window_tf += nadv*self.advance_sec
                        # If the new window ends inside the current data
                        if mltrace.stats.endtime >= next_window_tf and self.stance == 'eager':
                            # Consider re-approving the window for copying + trimming
                            fv = mltrace.get_fvalid_subset(starttime=next_window_ti,
                                                        endtime=next_window_tf)
                            # If window passes threshold, re-approve
                            if fv >= self.ref['thresholds']['ref']:
                                self.window_tracker[site][inst][mod].update({'ready': True})
                        # Otherwise preserve dis-approval of window generation (ready = False) for now


            

    def __repr__(self):
        """
        Provide a user-friendly string representation of this WindowMod's parameterization
        and state.
        """
        rstr = f'WindowMod for model architecture "{self.model_name}"\n'
        rstr += 'Ref.:'
        for _k, _v in self.ref.items():
            if _k in ['overlap', 'npts']:
                rstr += f' {_k}:{_v} samples |'
            elif _k == "sampling_rate":
                rstr += f' {_k}: {_v} sps |'
            elif _k == 'component':
                rstr += f' {_k}:"{_v}" |'
            elif _k == 'threshold':
                rstr += f' {_k}:{_v:.3f}'
        rstr += '\nAliases '
        for _k, _v in self.aliases.items():
            rstr += f' "{_k}":{[__v for __v in _v]}'
        rstr += f'\nIndex: {len(self.window_tracker)} sites |'
        _ni = 0
        for _v in self.window_tracker.values():
            _ni += len(_v) - 1
        types = {}
        for _x in self.output:
            if type(_x).__name__ not in types.keys():
                types.update({type(_x).__name__: 1})
            else:
                types[type(_x).__name__] += 1
        rstr += f' {_ni} instruments'
        rstr += '\nQueue: '
        for _k, _v in types.items():
            rstr += f'{_v} {_k}(s) | '
        if len(types) > 0:
            rstr = rstr[:-3]
        else:
            rstr += 'Nothing'
            
        return rstr