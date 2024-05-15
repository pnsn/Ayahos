"""
:module: wyrm.processing.window
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module hosts the definition for a Wyrm class that produces windowed
    copies of input obspy.Trace-like data objects organized by seismic instrument
    and for a specified duration and overlap to match expected data bounds
    for windowed analysis of waveforms conducted by machine learning models

    WindowWyrm - a submodule for sampling a waveform buffer and generating forward-marching
                windowed copies of these data as they become available
                    PULSE
                        input: WyrmStream holding (ML)Trace(Buffer) objects
                        output: deque holding WindowStream objects

"""

import time
import seisbench.models as sbm
from collections import deque
from ayahos.core.trace.mltrace import MLTrace
from ayahos.core.wyrms.wyrm import Wyrm
from ayahos.core.stream.dictstream import DictStream
from ayahos.core.stream.windowstream import WindowStream
from ayahos.util.input import bounded_floatlike, bounded_intlike


class WindowWyrm(Wyrm):
    """
    The WindowWyrm class takes windowing information from an input
    seisbench.models.WaveformModel object and user-defined component
    mapping and data completeness metrics and provides a pulse method
    that iterates across entries in an input DictStream object and
    generates WindowStream copies of sampled data that pass data completeness
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
        pulse_type='network',
        max_pulse_size=1,
        **options
    ):
        """Initialize a WindowWyrm object that samples a WyrmStream of MLTraceBuffer
        objects and generates WindowStream copies of windowed data if a reference
        component for a given instrument in the WyrmStream is present and has
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
        :param pulse_type: style of running pulse, defaults to 'network'
            Supported values:
                'network' - attempt to create one window from each instrument per pulse
                'site' - create a window/windows for a given site per pulse iteration, tracking
                        which site last generated windowed data. Permits for list wrapping  
                'instrument' - under development       
        :type pulse_type: str, optional
        :param max_pulse_size: maximum number of iterations for a given pulse, defaults to 1
        :type max_pulse_size: int, optional
        :param **options: key-word argument collector to pass to MLTrace.trim() via MLTraceBuffer.trim_copy() methods
        :type **options: kwargs
        """
        # Initialize/inherit from Wyrm
        super().__init__(max_pulse_size=max_pulse_size)


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
        reft = bounded_floatlike(
            reference_completeness_threshold,
            name = "reference_completeness_threshold",
            minimum=0,
            maximum=1,
            inclusive=True
        )
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

    
        self.ref = {'sampling_rate': refsr, 'overlap': refo, 'npts': refn, 'component': refc, 'threshold': reft}
         # Set Defaults and Derived Attributes
        # Calculate window length, window advance, and blinding size in seconds
        self.window_sec = (self.ref['npts'] - 1)/self.ref['sampling_rate']
        self.advance_npts = self.ref['npts'] - self.ref['overlap']
        self.advance_sec = (self.ref['npts'] - self.ref['overlap'])/self.ref['sampling_rate']
        self.options = options
        # Create dict for holding instrument window starttime values
        self.window_tracker = {}

    def unit_process(self, x, i_):
        nnew = 0
        if self.pulse_type == 'network':
            # Iterate across each site
            for site, _sv in self.window_tracker.items():
                # Iterate across each instrument at a site
                for inst, _ssv in _sv.items():
                    # Skip non-instrument entries
                    if isinstance(_ssv, dict):
                        # Iterate across model-weight codes (usually just one)
                        for mod in _ssv.keys():
                            # Sample a new window
                            nnew += self._sample_window(x, site, inst, mod)
            self._update_window_tracker(x)
            if nnew == 0:
                status = False
            else:
                status = True

        elif self.pulse_type == 'site':
            raise NotImplementedError('under development')
            # sites = list(self.window_tracker.keys())
            # for site in sites:
            #     if self.next_code is None:
            #         self.next_code = site
            #     if site == self.next_code:
            #         pass
            #     else:
                    # continue
        elif self.pulse_type == 'instrument':
            raise NotImplementedError('under development')
    
        return status

    # INHERIT PULSE FROM WYRM



    def _update_window_tracker(self, dst):
        """
        PRIVATE METHOD

        Scan across MLTraces in an input Dictionary Stream 
        with a component code matching the self.ref['component']
        attribute of this WindowWyrm and populate new branches in
        the self.window_index attribute if new site.inst
        (Network.Station.Location.BandInstrument{ref}) MLTraces are found

        :: INPUT ::
        :param dst: [wyrm.data.WyrmStream.WyrmStream] containing 
                        wyrm.data.mltrace.MLTrace type objects
        """
        for mltr in dst.traces.values():
            if not isinstance(mltr, MLTrace):
                raise TypeError('this build of WindowWyrm only works with wyrm.data.mltrace.MLTrace objects')
            # Get site, instrument, mod, and component codes from MLTrace
            site = mltr.site
            inst = mltr.inst
            comp = mltr.comp
            mod = mltr.mod
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
                                                        {'ti': mltr.stats.starttime,
                                                         'ref': mltr.id,
                                                         'ready': False}},
                                                't0': mltr.stats.starttime}})
            # If site is in window_tracker
            else:
                # If inst is not in this site subdictionary
                if inst not in self.window_tracker[site].keys():
                    self.window_tracker[site].update({inst:
                                                        {mod:
                                                            {'ti': self.window_tracker[site]['t0'],
                                                             'ref': mltr.id,
                                                             'ready': False}}})
                # If inst is in this site subdictionary
                else:
                    # If mod is not in this inst sub-subdictionary
                    if mod not in self.window_tracker[site][inst].keys():
                        self.window_tracker[site][inst].update({mod:
                                                                    {'ti': self.window_tracker[site]['t0'],
                                                                     'ref': mltr.id,
                                                                     'ready': False}})
                        
            ### WINDOW TRACKER TIME INDEXING/WINDOWING STATUS CHECK SECTION ###   
            # Get window edge times
            next_window_ti = self.window_tracker[site][inst][mod]['ti']
            next_window_tf = next_window_ti + self.window_sec
            # If the trace has reached or exceeded the endpoint of the next window
            if next_window_tf <= mltr.stats.endtime:
                # Get valid fraction for proposed window in MLTrace
                fv = mltr.get_fvalid_subset(starttime=next_window_ti, endtime=next_window_tf)
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
                    if next_window_ti < mltr.stats.starttime:
                        # Determine the number of advances that should be applied to catch up
                        nadv = 1 + (mltr.stats.starttime - next_window_ti)//self.advance_sec
                        # Apply advance
                        self.window_tracker[site][inst][mod]['ti'] += nadv*self.advance_sec
                        next_window_ti += nadv*self.advance_sec
                        next_window_tf += nadv*self.advance_sec
                        # If the new window ends inside the current data
                        if mltr.stats.endtime >= next_window_tf:
                            # Consider re-approving the window for copying + trimming
                            fv = mltr.get_fvalid_subset(starttime=next_window_ti,
                                                        endtime=next_window_tf)
                            # If window passes threshold, re-approve
                            if fv >= self.ref['threshold']:
                                self.window_tracker[site][inst][mod].update({'ready': True})
                        # Otherwise preserve dis-approval of window generation (ready = False) for now
                                

    def _sample_window(self, dst, site, inst, mod):
        """
        PRIVATE METHOD
        """
        nnew = 0
        _ssv = self.window_tracker[site][inst][mod]
        if _ssv['ready']:
            # if self._timestamp:
            #     start_entry = ['WindowWyrm','_sample_window','start',time.time()]
            self.logger.info(f'generating window for {site}.{inst}?.{mod}')

            next_window_ti = _ssv['ti']
            next_window_tf = next_window_ti + self.window_sec
            # Subset data view
            _dst = dst.fnselect(f'{site}.{inst}?.{mod}')
            # Copy/trim traces from view
            traces = []
            for _mltb in _dst.traces.values():
                mlt = _mltb.trimmed_copy(starttime=next_window_ti,
                                        endtime=next_window_tf,
                                        pad=True,
                                        fill_value=None)
                mlt.stats.model = self.model_name
                traces.append(mlt)
            # Compose WindowStream copy
            cst = WindowStream(traces=traces,
                                    header={'reference_starttime': next_window_ti,
                                            'reference_sampling_rate': self.ref['sampling_rate'],
                                            'reference_npts': self.ref['npts'],
                                            'aliases': self.aliases},
                                    ref_component=self.ref['component'])
            # if self._timestamp:
            #     # TODO: Figure out why this hard reset is needed for stats.processing...
            #     cst.stats.processing = []
            #     cst.stats.processing.append(start_entry)
            #     cst.stats.processing.append(['WindowWyrm','_sample_window','end',time.time()])
            # Append to queue
            self.output.append(cst.copy())
            del cst
            # Update window with an advance for this instrument
            self.window_tracker[site][inst][mod]['ti'] += self.advance_sec
            # Set ready flag to False for this new window (_update_window_tracker will handle re-readying)
            self.window_tracker[site][inst][mod].update({'ready': False})
            # Add to nneww
            nnew += 1
        return nnew
    
    def _reset(self, attr='window_tracker', safety_catch=True):
        if safety_catch:
            if attr in ['window_tracker','queue']:
                answer = input(f'About to delete contents of WindowWyrm.{attr} | Proceed? [Y]/[n]')
            elif attr == 'both':
                answer = input('About to delete contents of WindowWyrm.window_tracker and WindowWyrm.queue | Proceed? [Y]/[n]')
            if answer == 'Y':
                proceed = True
            else:
                proceed = False
        else:
            proceed = True
        if proceed:
            if attr in ['window_tracker', 'both']:
                self.window_tracker = {}
                if safety_catch:
                    print('WindowWyrm.window_tracker reset to empty dictionary')
            if attr in ['queue','both']:
                self.output = deque()
                if safety_catch:
                    print('WindowWyrm.queue reset to empty deque')

            
    def update_from_seisbench(self, model):
        """
        Helper method for (re)setting the window-defining attributes for this
        WindowWyrm object from a seisbench.models.WaveformModel object:

            self.model_name = model.name
            self.ref['sampling_rate'] = model.sampling_rate
            self.ref['npts'] = model.in_samples
            self.ref['overlap'] = model._annotate_args['overlap'][1]

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
        else:
            raise TypeError('seisbench.models.WaveformModel base class does not provide the necessary update information')

    def __repr__(self):
        """
        Provide a user-friendly string representation of this WindowWyrm's parameterization
        and state.
        """
        rstr = f'WindowWyrm for model architecture "{self.model_name}"\n'
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
    
    # def __str__(self):
    #     """
    #     Provide a string representation of this WindowWyrm's initialization
    #     """
    #     rstr = 'wyrm.core.process.WindowWyrm('
    #     rstr += f'component_aliases={self.aliases}, '
    #     rstr += f'reference_component="{self.ref["component"]}", '
    #     rstr += f'reference_completeness_threshold={self.ref["threshold"]}, '
    #     rstr += f'model_name="{self.model_name}", '
    #     rstr += f'reference_sampling_rate={self.ref["sampling_rate"]}, '
    #     rstr += f'reference_npts={self.ref["npts"]}, '
    #     rstr += f'reference_overlap={self.ref["overlap"]}, '
    #     rstr += f'max_pulse_size={self.max_pulse_size}, '
    #     rstr += f'debug={self.debug}'
    #     for _k, _v in self.options.items():
    #         if isinstance(_v, str):
    #             rstr += f', {_k}="{_v}"'
    #         else:
    #             rstr += f', {_k}={_v}'
    #     rstr += ')'
    #     return rstr


    # def pulse(self, x):
    #     """
    #     Primary method of WindowWyrm that executes up to max_pulse_size
    #     iterations of the following subroutines:
    #         _update_window_tracker
    #             --> Iterates across traces in `x` and updates the window_index
    #                 attribute with windowing parameters and assesses if windows
    #                 can be generated based on trace metadata
    #         _sample_windows
    #             --> Creates trimmed copies of traces in `x` housed in
    #                 WindowStream objects, appending these to self.queue
    #     This pulse has an early stopping clause if no new windows are generated by
    #     _sample_windows.

    #     :: INPUT ::
    #     :param x: [wyrm.data.WyrmStream.WyrmStream] comprising MLTrace(Buffer) objects

    #     :: OUTPUT ::
    #     :return y: [collections.deque] access to this WindowWyrm's self.queue attribute
    #     """
    #     if not isinstance(x, WyrmStream):
    #         raise TypeError
    #     # Update window tracker with WyrmStream metadata
    #     self._update_window_tracker(x)
    #     # Run pulse based on pulse_type
    #     if self.pulse_type == 'network':
    #         # if self.debug:
    #         #     print(f'∂∂∂ Window∂ - qlen {len(x)} - network pulse ∂∂∂')
    #         self._network_pulse(x)
    #     elif self.pulse_type == 'site':
    #         # if self.debug:
    #         #     print(f'∂∂∂ Window∂ - qlen {len(x)} - site pulse ∂∂∂')
    #         self._sitewise_pulse(x)

    #     y = self.queue
    #     return y
        
    # def _network_pulse(self, x):
    #     nnew = 0
    #     # Execute number of network-wide sweeps
    #     for _ in range(self.max_pulse_size):
    #         # Iterate across sites
    #         for site, _sv in self.window_tracker.items():
    #             # if self.debug:
    #             #     print(f'   {site}')
    #             # Iterate across instruments
    #             for inst, _ssv in _sv.items():
    #                 # Skip t0 reference
    #                 if isinstance(_ssv, dict):
    #                     # Iterate across model-weights
    #                     for mod in _ssv.keys():
    #                         nnew += self._sample_window(x, site, inst, mod)
    #         self._update_window_tracker(x)
    #         # If network sweep does not produce new windows, execute early stopping
    #         if nnew == 0:
    #             break



    # def _sitewise_pulse(self, x):
    #     niter = 0
    #     # Get current list of sites
    #     sites = list(self.window_tracker.keys())
    #     # Iterate across site names
    #     for _i, site in enumerate(sites):
    #         # Handle very first site-wise looping catch on last_code
    #         if self.next_code is None:
    #             self.next_code = site
            
    #         if site == self.next_code:
    #             pass
    #         else:
    #             continue

    #         # if self.debug:
    #         #     print(f'    processing {site}')
    #         # If you're still here, iterate across instruments
    #         for inst, _ssv in self.window_tracker[site].items():
    #             # Skip over 't0' reference entry in window_tracker[site]
    #             if not isinstance(_ssv, dict):
    #                 continue
    #             for mod in _ssv.keys():
    #                 # Attempt to sample window (if approved)
    #                 nnew = self._sample_window(x, site, inst, mod)
    #                 # Get the site code of the next site in the list
    #                 if _i + 1 < len(sites):
    #                     self.next_code = sites[_i + 1]
    #                 # If we hit the end of the list, loop over to the beginning
    #                 else:
    #                     self.next_code = sites[0]
    #                 # if self.debug:
    #                 #     print(f'   next_code is {self.next_code}')
    #         # Increase the iteration counter
    #         niter += 1
    #         if niter > self.max_pulse_size:
    #             return
                        

    