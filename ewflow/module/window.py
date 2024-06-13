"""
:module: camper.module.window
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This conatins the class definition for a module that facilitates data window generation

Classes
-------
:class:`~camper.module.window.WindowMod`
"""
import logging, sys
import seisbench.models as sbm
from collections import deque
from ewflow.data.mltrace import Trace
from ewflow.data.mlstream import Stream, Window
from ewflow.module._base import _BaseMod

Logger = logging.getLogger(__name__)

class WindowMod(_BaseMod):
    """
    The WindowMod class takes windowing information from an input
    seisbench.models.WaveformModel object and user-defined component
    mapping and data completeness metrics and provides a pulse method
    that iterates across entries in an input Stream object and
    generates Window copies of sampled data that pass data completeness
    requirements. 
    """

    def __init__(
        self,
        component_aliases={"Z": "Z3", "N": "N1", "E": "E2"},
        reference_component='Z',
        reference_completeness_threshold=0.95,
        model_name="EQTransformer",
        reference_sampling_rate=100.0,
        reference_npts=6000,
        reference_overlap=1800,
        fnfilter=None,
        pulse_type='network',
        max_pulse_size=1,
        meta_memory=3600,
        report_period=False,
        max_output_size=1e9,
        **options):
        """Initialize a WindowMod object that samples a Mostreamream of TraceBuffer
        objects and generates Window copies of windowed data if a reference
        component for a given instrument in the Mostreamream is present and has
        sufficient data to meet windowing requirements

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
            also see :meth:`~camper.data.stream.Stream.fnselect`
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
        # Initialize/inherit from Mod
        super().__init__(max_pulse_size=max_pulse_size,
                         meta_memory=meta_memory,
                         report_period=report_period,
                         max_output_size=max_output_size)


        if pulse_type.lower() in ['network']: #,'site','instrument']:
            self.pulse_type = pulse_type.lower()
            self.next_code = None
        else:
            raise ValueError(f'pulse_type "{pulse_type}" not supported.')

        # Compatability checks for component_aliases
        if not isinstance(component_aliases, dict):
            raise TypeError('component_aliases must be type dict')
        elif not all(isinstance(_v, str) and _k in _v for _k, _v in component_aliases.items()):
            raise SyntaxError('component_aliases values must be type str and include the key value')
        else:
            self.aliases = component_aliases

        # Compatability check for reference_component
        if reference_component in self.aliases.keys():
            refc = reference_component
        else:
            raise ValueError('reference_component does not appear as a key in component_aliases')
        
        # Compatability check for reference_completeness_threshold
        if isinstance(reference_completeness_threshold, (float, int)):
            if 0 <= reference_completeness_threshold <= 1:
                reft = reference_completeness_threshold
            else:
                raise ValueError
        else:
            raise TypeError

        if not isinstance(model_name, str):
            raise TypeError('model_name must be type str')
        else:
            self.model_name = model_name

        # Target sampling rate compat check
        if isinstance(reference_sampling_rate, (int, float)):
            if 0 < reference_sampling_rate < 1e9:
                refsr = reference_sampling_rate
            else:
                raise ValueError('reference_sampling_rate must be in the range (0, 1e9)')
        else:
            raise TypeError('reference_sampling must be float-like')
        
        if isinstance(reference_npts, int):
            if 0 < reference_npts < 1e9:
                refn = reference_npts
            else:
                raise ValueError('reference_npts must be in the range (0, 1e9)')
        else:
            raise TypeError('reference_npts must be type int')
        
        if isinstance(reference_overlap, int):
            if 0 < reference_overlap < 1e9:
                refo = reference_overlap
            else:
                raise ValueError('reference_overlap must be in the range (0, 1e9)')
        else:
            raise TypeError('reference_overlap must be type int')

        if isinstance(fnfilter, str):
            self.fnfilter = fnfilter
        elif fnfilter is None:
            self.fnfilter = '*'
        else:
            raise TypeError('fnfilter must be type str or NoneType')

        self.ref = {'sampling_rate': refsr, 'overlap': refo, 'npts': refn, 'component': refc, 'threshold': reft}
         # Set Defaults and Derived Attributes
        # Calculate window length, window advance, and blinding size in seconds
        self.window_sec = (self.ref['npts'] - 1)/self.ref['sampling_rate']
        self.advance_npts = self.ref['npts'] - self.ref['overlap']
        self.advance_sec = (self.ref['npts'] - self.ref['overlap'])/self.ref['sampling_rate']
        self.options = options
        # Create dict for holding instrument window starttime values
        self.window_tracker = {}

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
            raise TypeError
        elif model.name != 'WaveformModel':
            if model.sampling_rate is not None:
                self.ref.update({'sampling_rate': model.sampling_rate})
            if model.in_samples is not None:
                self.ref.update({'npts': model.in_samples})
            self.ref.update({'overlap': model._annotate_args['overlap'][1]})
            self.model_name = model.name
            self.ref['threshold'] = (model.in_samples - model._annotate_args['blinding'][1][0])/model.in_samples
        else:
            raise TypeError('seisbench.models.WaveformModel base class does not provide the necessary update information')

    #################################
    # PULSE POLYMORPHIC SUBROUTINES #
    #################################


    def _should_this_iteration_run(self, input, input_measure, iter_number):
        """
        POLYMORPHIC
        Last updated with :class:`~camper.module.window.WindowMod`

        unconditional pass - early stopping is handled by
        :meth:`~camper.module.window.WindowMod._should_next_iteration_run`

        :param input: standard input
        :type input: camper.data.stream.Stream
        :param iter_number: iteration number, unused
        :type iter_number: int
        :return status: should iterations continue in pulse, always True
        :rtype: bool
        """        
        status = True
        return status
    
    def _unit_input_from_input(self, input):
        """_get_obj_from_input for WindowMod

        obj is a view of input

        :param input: standard input
        :type input: camper.data.stream.Stream
        :return: _description_
        :rtype: _type_
        """
        if isinstance(input, Stream):
            unit_input = input
            return unit_input
        else:
            Logger.error('TypeError - input is not type Stream')
            sys.exit(1)
    
    def _unit_process(self, unit_input):
        """_unit_process for WindowMod

        Update the window_tracker with the contents of unit_input and then
        generate one window for each instrument in unit_input that has a 'ready'
        flag in window_tracker

        Newly generated windows are appended to WindowMod.output

        :param unit_input: view of a Stream containing waveforms
        :type unit_input: camper.data.stream.Stream
        """        
        unit_output = deque()
        # Update window tracker
        self.__update_window_tracker(unit_input)
        # Conduct network-wide pulse
        # Iterate across site-level dictionary entries
        for site, site_dict in self.window_tracker.items():
            # Iterate across inst-level dictionary entries
            for inst, inst_dict in site_dict.items():
                # skip the t0:UTCDateTime entry
                if isinstance(inst_dict, dict):
                    # Iterate across mod-level dictionary entries
                    for mod, value in inst_dict.items():
                        # If this instrument record is ready to produce a window
                        if value['ready']:
                            fnstring = f'{site}.{inst}?.{mod}'
                            # Logger.info(f'generating window for {fnstring}')
                            next_window_ti = value['ti']
                            next_window_tf = next_window_ti + self.window_sec
                            # Subset to all traces for this instrument
                            _stream = unit_input.fnselect(fnstring)
                            # Create copies of trimmed views of Trace(Buffers)
                            traces = []
                            # Iterate over tracebuffers
                            for _mltb in _stream:
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
                                          'reference_npts': self.ref['npts']},
                                ref_component=self.ref['component']
                            )
                            # Append window to output
                            unit_output.append(window)
                            # Advance window start time in window_tracker
                            old_ti = self.window_tracker[site][inst][mod]['ti']
                            self.window_tracker[site][inst][mod]['ti'] += self.advance_sec
                            new_ti = self.window_tracker[site][inst][mod]['ti']
                            Logger.debug(f'New window for {window.stats.common_id} at {old_ti} += next at {new_ti}')
                            # Set ready flag to false for this site
                            self.window_tracker[site][inst][mod].update({'ready': False})
        return unit_output

                        
    def _capture_unit_output(self, unit_output):
        """_capture_unit_out for WindowMod

        data capture is handled in _unit_process

        This method signals early stopping to pulse() if _unit_process
        did not generate new windows (i.e., unit_out == 0)
        
        :param unit_out: number of new windows generated in _unit_process
        :type unit_out: int
        :return status: should iterations continue?
        :rtype status: bool
        """  
        self.output += unit_output
        extra = len(self.output) - self.max_output_size
        # if extra > 0:
        #     Logger.warning(f'{self.__class__.__name__} object reached max_output_size. Deleting {extra} oldest values')
        while len(self.output) > self.max_output_size:
            self.output.popleft()


    def _should_next_iteration_run(self, unit_output):
        if len(unit_output) > 0:
            status = True
        else:
            status = False
        return status

    def __update_window_tracker(self, stream):
        """
        PRIVATE METHOD

        Scan across Traces in an input Dictionary Stream 
        with a component code matching the self.ref['component']
        attribute of this WindowMod and populate new branches in
        the self.window_index attribute if new site.inst
        (Network.Station.Location.BandInstrument{ref}) Traces are found

        :: INPUT ::
        :param stream: Stream object containing :class:`~camper.data.trace.Trace`-type objects
        :type stream: camper.data.stream.Stream
        """
        # Subset using fnfilter
        fstream = stream.fnselect(self.fnfilter)
        # Iterate across subset
        for trace in fstream.traces.values():
            if not isinstance(trace, Trace):
                raise TypeError('this build of WindowMod only works with camper.data.trace.Trace objects')
            # Get site, instrument, mod, and component codes from Trace
            site = trace.site
            inst = trace.inst
            comp = trace.comp
            mod = trace.mod
            ### WINDOW TRACKER NEW BRANCH SECTION ###
            # First, check if this is a reference component
            if comp not in self.ref['component']:
                # If not a match, continue to next trace
                continue
            # Otherwise proceed
            else:
                pass
            
            # If new site in window_tracker
            if site not in self.window_tracker.keys():
                # Populate t0's
                self.window_tracker.update({site: 
                                                {inst: 
                                                    {mod: 
                                                        {'ti': trace.stats.starttime,
                                                         'ref': trace.id,
                                                         'ready': False}},
                                                't0': trace.stats.starttime}})
                Logger.info(f'Added buffer tree for {site} - triggered by {trace.id}')
            # If site is in window_tracker
            else:
                # If inst is not in this site subdictionary
                if inst not in self.window_tracker[site].keys():
                    self.window_tracker[site].update({inst:
                                                        {mod:
                                                            {'ti': self.window_tracker[site]['t0'],
                                                             'ref': trace.id,
                                                             'ready': False}}})
                    Logger.info('Added buffer branch {inst} to {site} tree - triggered by {trace.id}')
                # If inst is in this site subdictionary
                else:
                    # If mod is not in this inst sub-subdictionary
                    if mod not in self.window_tracker[site][inst].keys():
                        self.window_tracker[site][inst].update({mod:
                                                                    {'ti': self.window_tracker[site]['t0'],
                                                                     'ref': trace.id,
                                                                     'ready': False}})
                        
                        
            ### WINDOW TRACKER TIME INDEXING/WINDOWING STATUS CHECK SECTION ###   
            # Get window edge times
            next_window_ti = self.window_tracker[site][inst][mod]['ti']
            next_window_tf = next_window_ti + self.window_sec
            # If the trace has reached or exceeded the endpoint of the next window
            if next_window_tf <= trace.stats.endtime:
                # Get valid fraction for proposed window in Trace
                fv = trace.get_fvalid_subset(starttime=next_window_ti, endtime=next_window_tf)
                # If threshold passes
                if fv >= self.ref['threshold']:
                    # set (window) ready flag to True
                    self.window_tracker[site][inst][mod].update({'ready': True})
                    # And continue to next trace
                    continue
                # If threshold fails
                else:
                    # If the window was previously approved, recind approval
                    if self.window_tracker[site][inst][mod]['ready']:
                        self.window_tracker[site][inst][mod].update({'ready': False})

                    # If data start after the proposed window, increment window index to catch up
                    if next_window_ti < trace.stats.starttime:
                        # Determine the number of advances that should be applied to catch up
                        nadv = 1 + (trace.stats.starttime - next_window_ti)//self.advance_sec
                        # Apply advance
                        self.window_tracker[site][inst][mod]['ti'] += nadv*self.advance_sec
                        next_window_ti += nadv*self.advance_sec
                        next_window_tf += nadv*self.advance_sec
                        # If the new window ends inside the current data
                        if trace.stats.endtime >= next_window_tf:
                            # Consider re-approving the window for copying + trimming
                            fv = trace.get_fvalid_subset(starttime=next_window_ti,
                                                        endtime=next_window_tf)
                            # If window passes threshold, re-approve
                            if fv >= self.ref['threshold']:
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