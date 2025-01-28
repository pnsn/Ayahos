"""
:module: PULSE.mod.window
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This conatins the class definition for a lineage of modules that facilitate
    regularly spaced generation of time-windowed, subset samples 
    :class:`~.DictStream` objects.

    Lineage

    (BaseMod)
        |
    SampleMod - Generate subset views of DictStream contents
        |       and return those views (i.e. aliases to source [meta]data)
        |
    WindowMod - Generate subset copies of DictStream contents
                and generate :class:`~.Window` objects

# TODO: Need to include up-incrementing windows
"""
from collections import deque
from warnings import warn

from obspy import UTCDateTime
import seisbench.models as sbm

from PULSE.util.header import MLStats
from PULSE.data.dictstream import DictStream
from PULSE.data.window import Window
from PULSE.util.header import WindowStats
from PULSE.mod.base import BaseMod

class SamplingMod(BaseMod):
    
    def __init__(
            self,
            length=59.99,
            step=42.,
            delay=0.,
            min_valid_frac=0.9,
            fold_thr = 1.,
            ref_val='Z',
            ref_key='component',
            split_key='instrument',
            eager=False,
            blind_after_sampling=False,
            max_pulse_size=1,
            maxlen=None,
            name=None):
        """_summary_

        :param length: sampling window length in seconds, defaults to 60.
        :type length: float, optional
        :param step: sampling step size in seconds, defaults to 42.
        :type step: float, optional
        :param min_valid_frac: minimum valid fraction of data to consider
            a given data view ready to generate a new windowed data view,
            defaults to 0.9
        :type min_valid_frac: float, optional
        :param fold_thr: fold threshold that data must meet or exceed to
            be considered "valid", defaults to 1.
        :type fold_thr: float, optional
        :param ref_val: the reference value used to determine
            if a view can be sampled from a given subset, defaults to 'Z'
        :type ref_val: str, optional
        :param ref_key: the id key corresponding to **ref_val**
            , defaults to 'component'
        :type ref_key: str, optional
        :param split_key: The id key on which input dictstream
            objects are split , defaults to 'instrument'
        :type split_key: str, optional
        :param mode: what mode should this SampleMod be run in?
            defaults to 'normal'
             - **eager** - generate a set of views as soon as the reference trace
                has enough data to satisfy **min_valid_frac**
             - **normal** - generate a set of views only once the endtime 
             - **patient** - 
        :type mode: str, optional
        :param max_pulse_size: _description_, defaults to 1
        :type max_pulse_size: int, optional
        :param maxlen: _description_, defaults to None
        :type maxlen: _type_, optional
        :param name: _description_, defaults to None
        :type name: _type_, optional
        """        
        # Inherit from BaseMod
        super().__init__(max_pulse_size=max_pulse_size,
                         maxlen=maxlen,
                         name=name)
        self.length = length
        self.step = step
        self.delay = delay
        self.min_valid_frac = min_valid_frac
        self.fold_thr = fold_thr
        self.split_key = split_key
        self.ref_key = ref_key
        self.ref_val = ref_val
        # BOOL Arguments
        self.eager = eager
        self.blind = blind_after_sampling
        # Generate index object
        self.index = {}
        # Set input_types
        self._input_types = [DictStream]

        # Compatability Catch
        if self.delay > 0. and self.eager:
            raise NotImplementedError('cannot use a non-zero delay for "eager" mode')

    def __setattr__(self, key, value):
        """Farm out attribute setting safety checks to this
        method, rather than re-writing them in multiple method
        headers!

        :param key: _description_
        :type key: _type_
        :param value: _description_
        :type value: _type_
        :raises TypeError: _description_
        :raises ValueError: _description_
        :raises NotImplementedError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises KeyError: _description_
        :raises TypeError: _description_
        :raises AttributeError: _description_
        """
        # FLOAT-LIKE NON-NEGATIVE ATTRIBUTES    
        if key in ['length','step', 'delay', 'min_valid_frac', 'fold_thr']:
            if isinstance(value, (int, float)):
                value = float(value)
            else:
                raise TypeError
            if key == 'delay': 
                if value < 0:
                    raise ValueError
            elif value <= 0:
                raise ValueError
            
        # SAFETY CATCHES FOR MAX VALUES
        if key in ['length','delay']:
            if value > 1e6:
                raise NotImplementedError 
        if key == 'step':
            if self.length < value:
                raise ValueError 
        if key == 'min_valid_frac':
            if value > 1:
                raise ValueError

        # ID_KEYS VALUES ONLY
        if key in ['ref_key', 'split_key']:
            if value in dict(MLStats().id_keys).keys():
                pass
            else:
                raise KeyError
            
        # REF_VAL MUST BE SET
        if key == 'ref_val':
            if isinstance(value, str):
                value = set(value)
            elif hasattr(value, '__iter__'):
                if all(isinstance(_e, str) for _e in value):
                    value = set(value)
                else:
                    raise TypeError
            else:
                raise TypeError
            
        # BOOL ATTR
        if key in ['eager','blind']:
            if not isinstance(value, bool):
                raise TypeError
            
        # APPLY
        super().__setattr__(key, value)
    
    def get_unit_input(self, input: DictStream) -> dict:
        unit_input = {}
        for _val, _ds in input.split_on(self.split_key).items():
            self._assess_new_entry(_val, _ds)
            if _val in self.index.keys():
                _ds_view = self._assess_entry_readiness(_val, _ds)
                if isinstance(_ds_view, DictStream):
                    unit_input.update({_val: _ds_view})
        return unit_input
    

    def _assess_new_entry(self, _val, _ds):
        """Subprocess for :meth:`~.SamplingMod.get_unit_input`
        that assesses if a specific subset slice of a :class:`~.DictStream`
        object and its split_key value warrant generating a new entry in the
        **index** attribute of this :class:`~.SamplingMod` object.

        :param _val: split_key value corresponding to the id_key
            type assigned to **split_value** that is common to all
            :class:`~.FoldTrace` objects contained in **_ds**
        :type _val: str
        :param _ds: subset slice of a :class:`~.DictStream` containing
            only :class:`~.FoldTrace` objects that share the same
            **_val** for their id_key's for **split_value**
        :type _ds: PULSE.data.dictstream.DictStream
        """        
        # New Entry Generation
        if _val not in self.index.keys():
            primaries = set()
            secondaries = set()
            for _id, _ft in _ds.traces.items():
                if _ft.id_keys[self.ref_key] in self.ref_val:
                    primaries.add(_id)
                else:
                    secondaries.add(_id)
            # If there is one or more primaries, generate new entry
            if len(primaries) >= 1:
                # Set t0 to NOW
                t0 = UTCDateTime()
                # Find the earliest starttime in the DictStream view
                for _p in primaries:
                    if _ds[_p].stats.starttime < t0:
                        t0 = _ds[_p].stats.starttime
                # Populate new_entry
                new_entry = {'p_ids': primaries,
                                's_ids': secondaries,
                                'ready': False,
                                'ti': t0,
                                'tf': t0 + self.length}
                # Register new_entry in **index**
                self.index.update({_val: new_entry})


    def _assess_entry_readiness(self, _val, _ds):
        """Subprocess for :meth:`~.SamplingMod.get_unit_input`
        that checks if there are new or dropped key values in the
        **keys** of **_ds** that correspond to registered keys for
        entry **_val** in the **index** attribute of this 

        :param _val: _description_
        :type _val: _type_
        :param _ds: _description_
        :type _ds: _type_
        :raises RuntimeError: _description_
        """
        ready = False     
        # Entry Readiness Assessment
        if _val in self.index.keys():
            _index = self.index[_val]

            ## CATCH-UP SECTION ##
            for _pid in _index['p_ids']:
                # If the next window endt falls before the start of the trace
                if _index['tf'] <= _ds[_pid].stats.starttime:
                    # Shift starttime to the new starttime
                    _index['ti'] = _ds[_pid].stats.startime
                    _index['tf'] = _ds[_pid].stats.starttime + self.length
            
            ## PULL VIEW FOR CURRENT PROPOSED WINDOW ##
            _ds_view = _ds.view(starttime=_index['ti'],
                                endtime=_index['tf'],
                                keep_empty_traces=False)

            ## REGISTERING SECTION ##
            # Iterate across all traces and IDs
            for _id, _ft in _ds_view.traces.items():
                # Check if ids are registered
                if _id in _index['p_ids']:
                    pass
                elif _id in _index['s_ids']:
                    pass
                # Catch new ids coming in
                else:
                    if _ft.key_attr[self.ref_key] in self.ref_val:
                        _index['p_ids'].add(_id)
                    else:
                        _index['s_ids'].add(_id)

            ## DEREGISTERING SECTION ##
            # Check registered traces (i.e. p_ids & s_ids)
            _regset = _index['p_ids'].union(_index['s_ids'])
            # Get _regset - _ds.traces.keys()
            todrop = _regset.difference(_ds.traces.keys())
            # If todrop is nonempty, iterates and drops (or errors out)
            for _r in todrop:
                if _r in _index['p_ids']:
                    _index['p_ids'].remove(_r)
                elif _r in _index['s_ids']:
                    _index['s_ids'].remove(_r)
                else:
                    raise RuntimeError('Not sure how we got here...')
            
            ## READINESS ASSESSMENT SECTION ##
            for _pid in _index['p_ids']:
                _vf = _ds[_pid].get_valid_fraction(
                        starttime=_index['ti'],
                        endtime=_index['tf'],
                        threshold=self.fold_thr)
                # Get primary trace start and endtimes
                _ti = _ds[_pid].stats.starttime
                _tf = _ds[_pid].stats.endtime
                # EAGER BEHAVIOR
                # - only one primary trace needs to satisfy the min_valid_fract
                # - that trace does not need to have the target endtime
                if self.eager:
                    if _vf >= self.min_valid_frac:
                        ready = True
                # NORMAL (NON-EAGER) BEHAVIOR
                # - One primary trace needs to have the window start and end times
                #   present in their scope and pass fvalid
                # - Allows use of a non-zero "delay" value
                else:
                    # Tests
                    # 1) Meets min_valid_frac
                    # 2) trace starttime is less than sample starttime
                    # 3) trace endtime minus delay is greater than sample endtime
                    tests = [_vf >= self.min_valid_frac,
                             _ti <= _index['ti'],
                             _tf - self.delay > _index['tf']]
                    if all(tests):
                        ready = True
            # Update readiness
            _index['ready'] = ready
            if ready:
                return _ds_view
            else:
                return None
        else:
            return None

    def run_unit_process(self, unit_input: dict) -> deque:
        """
        Iterate across data views that meet all specified
        requirements, 
        
        Apply blinding to the source material
        if specified

        Attach (copied) view passed through unit_input
        to unit_output

        :param unit_input: _description_
        :type unit_input: dict
        :return: _description_
        :rtype: deque
        """        
        unit_output = deque()
        for icode, ds_view in unit_input.items():
            _index = self.index[icode]
            # Ensure ready flag is flying
            if _index['ready']:
                # If blinding after sampling
                if self.blind:
                    ds_out = ds_view.copy()
                    for ft in ds_view:
                        ft.fold *= 0
                else:
                    ds_out = ds_view
                # Append view to unit_output
                unit_output.appendleft(ds_out)
                # Update window times
                _index['ti'] += self.step
                _index['tf'] += self.step
                # Un-Ready entry
                _index['ready'] = False        
        return unit_output
    
    def put_unit_output(self, unit_input: deque) -> None:
        for _ in range(len(unit_input)):
            _ds = unit_input.pop()
            self.output.appendleft(_ds)

    def update_from_seisbench(self, model: sbm.WaveformModel, delay_scalar=0):
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError('model must be a seisbench.models.WaveformModel-type object')
        # Append model name to the end of the module name 
        if len(self.name.split('_')) > 1:
            newname = f"{'_'.join(self.name.split('_')[1:])}_{model.name}"
        else:
            newname = model.name
        self.setname(newname)
        # Calculate the new length
        length = model.in_samples / model.sampling_rate
        # Calculate new step
        step = length - (model._annotate_args['overlap'][1] / model.sampling_rate)
        # Safety catch if old step is longer than new length
        if self.step > length:
            self.step = length*0.9
        # update length and step
        self.length = length
        self.step = step
        
        if isinstance(delay_scalar, (int, float)):
            delay_scalar = int(round(delay_scalar))
        else:
            raise TypeError
        
        if delay_scalar < 0:
            raise ValueError
        elif delay_scalar == 0:
            self.delay = 0.
        else:
            self.eager = False
            self.delay = delay_scalar*self.length - (delay_scalar - 1)*self.step
        return self

class WindowingMod(SamplingMod):
    """
    A :class:`~.SamplingMod` specialized for generating :class:`~PULSE.data.window.Window`
    objects for workflows focused on analyzing synchronized time-series recorded by a
    single seismic sensor

    """
    def __init__(
            self,
            target_npts = 6000,
            target_sampling_rate = 100.,
            overlap_npts = 1800,
            primary_threshold = 0.9,
            primary_components = set('Z3'),
            secondary_threshold = 0.8,
            secondary_components = {'NE','12'},
            eager=False,
            max_pulse_size=1,
            maxlen=None,
            name=None):
            # Use WindowStats to QC
            self.window_stats = WindowStats(
                {'target_npts': target_npts,
                 'target_sampling_rate': target_sampling_rate,
                 'pthresh':primary_threshold,
                 'sthresh':secondary_threshold}
            )
            super_kwargs = {'length': (target_npts - 1) / target_sampling_rate,
                            'step': (target_npts - overlap_npts) / target_sampling_rate,
                            'delay': 0.,
                            'min_valid_frac': primary_threshold,
                            'ref_val': primary_components,
                            'ref_key': 'component',
                            'split_key': 'instrument',
                            'eager': eager,
                            'blind_after_sampling': False,
                            'maxlen': maxlen,
                            'max_pulse_size': max_pulse_size,
                            'name': name}
            super().__init__(**super_kwargs)

            self.primary_components = primary_components
            self.secondary_components = secondary_components

    def __setattr__(self, key, value):
        if key in ['primary_components','secondary_components']:
            if isinstance(value, str):
                value = set(value)
            elif hasattr(value, "__iter__"):
                if all(isinstance(_e, str) for _e in value):
                    value = set(value)
                else:
                    raise TypeError(f'{key} must comprise only str entries')
            else:
                raise TypeError(f'{key} must be type str, or an iterable thereof')
        
        if key == 'primary_components':
            if any(len(_e) != 1 for _e in value):
                raise SyntaxError('Primary component codes must be single character strings')

        super().__setattr__(key, value)

    def run_unit_process(self, unit_input):
        unit_output = deque()
        # Iterate across reportedly ready views
        for icode, ds_view in unit_input.items():
            # Get index
            _index = self.index[icode]
            # Confirm ready flag
            if _index['ready']:
                
                # Confirm single p_ids registered
                if len(_index['p_ids']) > 1:
                    # Eventually open this up for Z and P primary channels on OBS + PickBlue
                    raise NotImplementedError('WindowingMod does not yet support multiple primary components')
                # If no registered primary ids - extra safety catch
                elif len(_index['p_ids']) == 0:
                    continue
                else:
                    pass
                # DEBUG: Gappy passage of metadata in Window.__init__
                # Get the relevant primary component code
                _pid = list(_index['p_ids'])[0]
                _pc = ds_view[_pid].stats.component
                # Create a copy of the general window_stats
                _ws = self.window_stats.copy()
                # Set the primary_id
                _ws.primary_id = _pid
                # Set the secondary components in the header input
                _ws.secondary_components = ''.join([_c[-1] for _c in _index['s_ids']])
                # Set target starttime from index entry
                _ws.target_starttime = _index['ti']
                # Create copies of the views
                traces = [_ft.copy() for _ft in ds_view]
                # GENERATE WINDOW
                try:
                    window = Window(traces=traces, header=_ws, primary_component=_pc)
                except ValueError:
                    breakpoint()
                # Attache to unit_output
                unit_output.appendleft(window)

                # Advance index entry
                _index['ready'] = False
                _index['ti'] += self.step
                _index['tf'] += self.step
        return unit_output
    

    def update_from_seisbench(self, model):
        super().update_from_seisbench(model)
        self.window_stats.update({'target_sampling_rate': model.sampling_rate,
                                  'target_npts': model.in_samples})
        return self

#             _stats = _index['stats']
#             if not _index['ready']:
#                 raise RuntimeError('Unready dictstream views are making it past get_unit_input')
#             traces = ds.view(starttime=_stats.target_starttime,
#                              endtime=_stats.target_endtime).copy()
#             header = _stats.copy()
#             # Generate window
#             window = Window(traces=traces, header=header)
#             # Append new window to unit_output
#             unit_output.appendleft(window)
#             # Update index to reflect new window generation
#             _index['ready'] = False
#             _stats.target_starttime += self.advance_dt
#             # Apply blinding if specified
#             if self.blind_after_sampling:
#                 for ft in ds:
#                     # Make a view of the foldtrace
#                     vft = ft.view(starttime=_stats.target_starttime,
#                                   endtime=_stats.target_endtime)
#                     # Set fold to 0 in the view
#                     vft.fold = vft.fold*0
#         return unit_output
                
#     def put_unit_output(self, unit_output: deque) -> None:
#         if not isinstance(unit_output, deque):
#             self.Logger.critical('TypeError: unit_output must be type collections.deque')
#         if not all(isinstance(_e, Window) for _e in unit_output):
#             self.Logger.critical('TypeError: unit_output elements are not all type PULSE.data.window.Window')
#         self.output += unit_output
    

# class WindowingMod(SamplingMod)
            
                


# def get_unit_input(self, input: DictStream) -> dict:
#         """Update the **index** of this :class:`~.WindMod` for new
#         and existing instrument codes in **input**

#         :param input: _description_
#         :type input: DictStream
#         :return: _description_
#         :rtype: dict
#         """        
#         unit_input = {}

#         # Split by instrument
#         for instrument, ds in input.split_on().items():
#             ### POPULATE NEW ENTRIES IN **INDEX** ###
#             # If this is a new instrument code
#             if instrument not in self.index.keys():
#                 new_entry = {'stats': self.window_stats.copy(),
#                              'ready': False}
#                 comps = set([ft.stats.component for ft in ds])
#                 # Check that at least one reference component is present
#                 if self.reference.intersection(comps) != set([]):

#                 # Find primary & secondary keys for this instrument
#                 for ft in ds:
#                     # If this is a primary component, populate primary_id and target_starttime
#                     if ft.stats.component in self.primary_components:
#                         new_entry['stats'].primary_id = ft.id
#                         new_entry['stats'].target_starttime = ft.stats.starttime
#                     # Find secondary keys if there are secondary channels
#                     elif len(ds) > 1:
#                         # Identify the component pairs
#                         for sc in self.secondary_components:
#                             if ft.stats.component in sc:
#                                 new_entry['stats'].update({'secondary_components': sc})
#                                 break
#                     else:
#                         new_entry['stats'].update({'secondary_components': 'NE'})
#                 # If a primary key has been identified, add new entry
#                 if new_entry['stats']['primary_id'] is not None:
#                     self.index.update({instrument: new_entry})

#             ### ASSESS READINESS TO GENERATE NEW WINDOWS ###
#             if instrument in self.index.keys():
#                 _index = self.index[instrument]
#                 _ws = _index['stats']
#                 # Get primary component
#                 _ft = ds[_ws.primary_id]
#                 # Catch large forward jumps (outages)
#                 if _ws.target_endtime < _ft.stats.starttime:
#                     # increment up window until target_starttime is within the
#                     # data time domain
#                     while _ws.target_starttime < ft.stats.starttime:
#                         _ws.target_starttime += self.advance_dt
#                 # Assess readiness for 'normal' and 'eager' windowing
#                 if self.windowing_mode in ['normal','eager']:
#                     fv = _ft.get_valid_fraction(
#                         starttime=_ws.target_starttime,
#                         endtime=_ws.target_endtime) 
#                     # If there are enough valid data to make the next window
#                     if fv > _ws.pthresh:
#                         # If eager, go ahead (even if the target_endtime isn't in the buffer yet)
#                         if self.windowing_mode == 'eager':
#                             _index['ready'] = True
#                         # If normal, make sure the target_endtime exists in the buffer
#                         elif self.windowing_mode == 'normal':
#                             if _ws.target_endtime <= ft.stats.endtime:
#                                 _index['ready'] = True
#                         # If padded, make sure target_endtime is at least a window length behind the buffer end
#                         elif self.windowing_mode == 'padded':
#                             if _ws.target_endtime <= (ft.stats.endtime + self.window_dt):
#                                 _index['ready'] = True
                
#             # Capture ready instruments for windowing
#             if self.index[instrument]['ready']:
#                 unit_input.update({instrument: ds})

#         # Trigger early stopping if no windows are approved
#         if len(unit_input) == 0:
#             self._continue_pulsing = False
#         return unit_input


                        


                    
            # # If there are traces in the instrument DictStream
            # if len(ds) == 0:
            #     continue
            # # If there is a primary component present
            # for ft in ds:
            #     if ft.comp in self.primary_components:
            #         self.index[instrument].update({})
            # if any(ft.comp in self.primary_components for ft in ds):




# class WindowMod(BaseMod):
#     """
#     The WindowMod class takes windowing information from an input seisbench.models.WaveformModel object and user-defined component
#     mapping and data completeness metrics and provides a pulse method that iterates across entries in an input DictStream object and
#     generates Window copies of sampled data that pass data completeness requirements. 

#     :param component_aliases: aliases for standard component names used in SeisBench, defaults to {"Z": "Z3", "N": "N1", "E": "E2"}
#     :type component_aliases: dict, optional
#     :param primary_component: reference component used for channel fill rules, defaults to 'Z'
#     :type primary_component: str, optional
#     :param primary_completeness_threshold: completeness fraction needed for reference component traces to pass QC, defaults to 0.95
#     :type primary_completeness_threshold: float, optional
#     :param model_name: name of the ML model the windows are intended for, defaults to "EQTransformer"
#     :type model_name: str, optional
#     :param target_sampling_rate: target sampling rate after preprocessing, defaults to 100.0
#     :type target_sampling_rate: float, optional
#     :param target_window_npts: target number of samples per window after preprocessing, defaults to 6000
#     :type target_window_npts: int, optional
#     :param target_overlap_npts: target number of overlapping samples between windows after preprocessing, defaults to 1800
#     :type target_overlap_npts: int, optional
#     :param fnfilter: fnmatch filter string to use for subsetting channel inputs, default is None
#     :type fnfilter: None or str
#         also see :meth:`~PULSE.data.dictstream.DictStream.fnselect`
#     :param pulse_type: style of running pulse, defaults to 'network'
#         Supported values:
#             'network' - attempt to create one window from each instrument per pulse
#             vv NOT IMPLEMENTED vv
#             'site' - create a window/windows for a given site per pulse iteration, tracking
#                     which site last generated windowed data. Permits for list wrapping  
#             'instrument' - under development       
#     :type pulse_type: str, optional
#     :param max_pulse_size: maximum number of iterations for a given pulse, defaults to 1
#     :type max_pulse_size: int, optional
#     :param **options: key-word argument collector to pass to Trace.trim() via TraceBuffer.trim_copy() methods
#     :type **options: kwargs
#     """

#     def __init__(
#         self,
#         primary_components='Z3',
#         secondary_components='NE12',
#         primary_completeness_threshold=0.95,
#         secondary_completeness_threshold=0.8,
#         model_name='EQTransformer',
#         target_sampling_rate=100.,
#         target_window_npts=6000,
#         target_overlap_npts=1800,
#         pad=False,
#         fill_value=None,
#         eager_generation=False,
#         stagger_sec=1.,
#         max_pulse_size=1,
#         maxlen=None,
#         name_suffix=None):

#         """Initialize a WindowMod object that samples a Modictstreamream of TraceBuffer
#         objects and generates Window copies of windowed data if a reference
#         component for a given instrument in the Modictstreamream is present and has
#         sufficient data to meet windowing requirements

#         :param component_aliases: aliases for standard component names used in SeisBench, defaults to {"Z": "Z3", "N": "N1", "E": "E2"}
#             values in the component_aliases should be an iterable that produces single character strings when iterated
#         :type component_aliases: dict, optional
#         :param primary_component: reference component code to use for channel fill rules, defaults to 'Z'.
#             Must match a key in **component_aliases**
#         :type primary_component: str, optional
#         :param primary_completeness_threshold: completeness fraction needed for reference component traces to pass QC, defaults to 0.95
#             Specified value must be :math:`\in [0,1]`. This value is used when assessing if a window is ready to be generated
#             Also see
#                 :meth:`~PULSE.mod.window.WindowMod.update_window_tracker`
#                 :meth:`~PULSE.data.window.Window.apply_fill_rule`
#         :type primary_completeness_threshold: float, optional
#         :param secondary_completeness_threshold: completeness fraction needed for "other" component traces to pass QC, defaults to 0.8
#             Specified value must be :math:`\in [0,1]`. This value is only used to populate **Window.stats** attributes
#             Also see :meth:`~PULSE.data.window.Window.__init__`
#         :type secondary_completeness_threshold: float, optional
#         :param model_name: name of the ML model the windows are intended for, defaults to "EQTransformer"
#         :type model_name: str, optional
#         :param target_sampling_rate: target sampling rate after preprocessing, defaults to 100.0
#             Used to determine the endtime for window generation and window advance starttimes for sequential windows
#         :type target_sampling_rate: float, optional
#         :param target_window_npts: target number of samples per window after preprocessing, defaults to 6000
#             Used to determine the endtime for window generation
#         :type target_window_npts: int, optional
#         :param target_overlap_npts: target number of overlapping samples between windows after preprocessing, defaults to 1800
#             Used to determine window starttimes for sequential windows
#         :type target_overlap_npts: int, optional
#         :param fnfilter: fnmatch filter string to use for subsetting channel inputs, default is None.
#             also see :meth:`~PULSE.data.dictstream.DictStream.fnselect`
#         :type fnfilter: NoneType or str
#         :param pulse_type: style of running pulse, defaults to 'network'
#             Supported values:
#                 'network' - attempt to create one window from each instrument per iteration in a call of :meth:`~PULSE.mod.window.WindowMod.pulse`
#                 TODO: Implement these or remove this feature
#                 vv NOT IMPLEMENTED vv
#                 'site' - create a window/windows for a given site per pulse iteration, tracking
#                         which site last generated windowed data. Permits for list wrapping  
#                 'instrument' - under development       
#         :type pulse_type: str, optional
#         :param stance: window generation rule, defaults to 'patient'
#             Supported values:
#                 'patient' - windows are not flagged as ready until the endtime of a reference component trace 
#                     exceeds the end of the window to be generated
#                 'eager' - windows are flagged as ready as soon as sufficient data are present in the reference component trace,
#                     even if the endtime of the trace does not exceed the endtime of the window to be generated.
#                     This can produce windows sooner in exchange for potentially omitting trailing data.
#         :param max_pulse_size: maximum number of iterations for a given pulse, defaults to 1
#         :type max_pulse_size: int, optional
#         :param **options: key-word argument collector to pass to :meth:`~PULSE.data.mltrace.MLTrace.trim_copy`
#         :type **options: kwargs
#         """
#         # Initialize/inherit from BaseMod
#         super().__init__(max_pulse_size=max_pulse_size, maxlen=maxlen, name_suffix=name_suffix)
#         demerits = 0
#         # # Compatability checks for `component_aliases`
#         # if not isinstance(component_aliases, dict):
#         #     self.Logger.critical('TypeError: component_aliases must be type dict')
#         #     demerits += 1
#         # elif not all(isinstance(_v, str) and _k in _v for _k, _v in component_aliases.items()):
#         #     self.Logger.critical('SyntaxError: component_aliases values must be type str and include the key value')
#         #     demerits += 1
#         # else:
#         #     self.aliases = component_aliases

#         # Compatability check for `primary_component`
#         if primary_components in self.aliases.keys():
#             refc = primary_components
#         else:
#             self.Logger.critical('ValueError: primary_component does not appear as a key in component_aliases')
#             demerits += 1
#         # Compatability check for `primary_completeness_threshold`
#         if isinstance(primary_completeness_threshold, (float, int)):
#             if 0 <= primary_completeness_threshold <= 1:
#                 reft = primary_completeness_threshold
#             else:
#                 self.Logger.critical('ValueError: primary_completeness_threshold out of valid range [0, 1]')
#                 demerits += 1
#         else:
#             self.Logger.critical('TypeError: primary_completeness_threshold must be type int or float')
#             demerits += 1
        
#         # Compatability check for `secondary_completeness_threshold`
#         if isinstance(secondary_completeness_threshold, (float, int)):
#             if 0 <= secondary_completeness_threshold <= 1:
#                 othert = secondary_completeness_threshold
#             else:
#                 self.Logger.critical('ValueError: secondary_completeness_threshold out of range')
#                 demerits += 1
#         else:
#             self.Logger.critical('TypeError: secondary_completeness_threshold out of range')
#             demerits += 1
        
#         # Compatability check for `model_name`
#         if not isinstance(model_name, str):
#             self.Logger.critical('TypeError: model_name must be type str')
#             demerits += 1
#         else:
#             self.model_name = model_name

#         # Compatability check for `target_sampling_rate`
#         if isinstance(target_sampling_rate, (int, float)):
#             if 0 < target_sampling_rate < 1e9:
#                 refsr = target_sampling_rate
#             else:
#                 self.Logger.critical('ValueError: target_sampling_rate must be in the range (0, 1e9)')
#                 demerits += 1
#         else:
#             self.Logger.critical('TypeError: target_sampling_rate must be float-like')
#             demerits += 1
        
#         # Compatability check for `target_window_npts`
#         if isinstance(target_window_npts, int):
#             if 0 < target_window_npts < 1e9:
#                 refn = target_window_npts
#             else:
#                 self.Logger.critical('ValueError: target_window_npts must be in the range (0, 1e9)')
#                 demerits += 1
#         else:
#             self.Logger.critical('TypeError: target_window_npts must be type int')
#             demerits += 1
        
#         # Compatability check for `target_overlap_npts`
#         if isinstance(target_overlap_npts, int):
#             if 0 <= target_overlap_npts < refn:
#                 refo = target_overlap_npts
#             elif target_overlap_npts >= refn:
#                 self.Logger.critical('ValueError: target_overlap_npts must be less than target_window_npts')
#                 demerits += 1
#             elif target_overlap_npts < 0:
#                 self.Logger.critical('ValueError: target_overlap_npts must be non-negative')
#                 demerits += 1
#         else:
#             self.Logger.critical('TypeError: target_overlap_npts must be type int')
#             demerits += 1
        
#         ## Compatability checks for kwargs passed to MLTrace.view_copy
#         self._vckwargs={}
#         # Compatability check for `pad`
#         if not isinstance(pad, bool):
#             self.Logger.critical('TypeError: pad must be type bool')
#             demerits += 1
#         else:
#             self._vckwargs.update({'pad': pad})

#         # Compatability check for `fill_value`
#         if not isinstance(pad, (type(None), int, float)):
#             self.Logger.critical('TypeError: fill_value must be type int, float or NoneType')
#             demerits += 1
#         else:
#             self._vckwargs.update({'fill_value': fill_value})


#         # Set Defaults and Derived Attributes
#         # Calculate window length, window advance, and blinding size in seconds
#         self.window_sec = (self.target['npts'] - 1)/self.target['sampling_rate']
#         self.advance_npts = self.target['npts'] - self.target['overlap']
#         self.advance_sec = (self.target['npts'] - self.target['overlap'])/self.target['sampling_rate']
#         # Create dict for holding instrument window starttime values
#         self.window_tracker = {}
#         # Create placeholder for next available start time
#         self.next_available_start = None

#         # Compatability check for stagger_sec
#         if isinstance(stagger_sec, (int, float)):
#             if 0 <= stagger_sec < self.window_sec:
#                 self.stagger_sec = float(stagger_sec)
#             else:
#                 self.Logger.critical('ValueError: stagger_sec must be in the range [0, self.window_sec)')
#                 demerits += 1
#         else:
#             self.Logger.critical('TypeError: stagger_sec must be float-like')
#             demerits += 1


#         # Compatability check for eager_generation
#         if not isinstance(eager_generation, bool):
#             self.Logger.critical('TypeError: eager_generation must be type bool.')
#             demerits += 1
#         else:
#             self.eager = eager_generation
        
#         # Make target values dictionary for look-up convienience
#         self.target = { 'sampling_rate': refsr,
#                         'overlap': refo,
#                         'npts': refn,
#                         'components': refc,
#                         'threshold': reft}

#         if demerits > 0:
#             self.Logger.critical(f'{demerits} errors raised during initialization. Exiting on {os.EX_DATAERR}')
#             sys.exit(os.EX_DATAERR)
#     #######################################
#     # Parameterization Convenience Method #
#     #######################################
        
#     def update_from_seisbench(self, model):
#         """
#         Helper method for (re)setting the window-defining attributes for this
#         WindowMod object from a seisbench.models.WaveformModel object:

#             self.model_name = model.name
#             self.target['sampling_rate'] = model.sampling_rate
#             self.target['npts'] = model.in_samples
#             self.target['overlap'] = model._annotate_args['overlap'][1]
#             self.primary_completeness_threshold = (model.in_samples - model._blinding[1][0])/model.in_samples
        
#         :param model: seisbench model to scrape windowing parameters from
#         :type model: seisbench.models.WaveformModel
#         """
#         if not isinstance(model, sbm.WaveformModel):
#             self.Logger.critical(f'TypeError: model is not a seisbench.models.WaveformModel child class.')
#             sys.exit(os.EX_DATAERR)
#         elif model.name != 'WaveformModel':
#             if model.sampling_rate is not None:
#                 self.target.update({'sampling_rate': model.sampling_rate})
#             if model.in_samples is not None:
#                 self.target.update({'npts': model.in_samples})
#             self.target.update({'overlap': model._annotate_args['overlap'][1]})
#             self.model_name = model.name
        
#             # self.target['threshold']['ref'] = (model.in_samples - model._annotate_args['blinding'][1][0])/model.in_samples

#         else:
#             self.Logger.critical(f'TypeError: seisbench.models.WaveformModel base class does not provide the necessary update information. Exiting on DATAERR ({os.EX_DATAERR})')
#             sys.exit(os.EX_DATAERR)

#     #################################
#     # PULSE POLYMORPHIC SUBROUTINES #
#     #################################
#     def measure_input(self, input):
#         return super().measure_input(input)
    
#     def measure_output(self):
#         return super().measure_output()
    
#     def get_unit_input(self, input):
#         if not isinstance(input, DictStream):
#             self.Logger.critical(f'TypeError: input must be type PULSE.data.dictstream.DictStream. Exiting on {os.EX_DATAERR}')
#             sys.exit(os.EX_DATAERR)
#         else:
#             if self.measure_input(input) == 0:
#                 self._continue_pulsing = False
#                 unit_input = None
#             else:
#                 unit_input = input
#         return unit_input

#     def run_unit_process(self, unit_input):
#         """POLYMORPHIC METHOD

#         Last updated with :class:`~PULSE.mod.window.WindowMod`

#         This method wraps two subroutines:
#         - :meth:`~PULSE.mod.window.Window.update_window_tracker` - scans unit_input metadata and determines if instrument-level
#         groups of MLTrace objects it contains can produce a new window based on window generation parameters set when
#         this :class:`~PULSE.mod.window.WindowMod` object was initialized
#         - :meth:`~PULSE.mod.window.Window.generate_windows` - generates new :class:`~PULSE.data.window.Window` objects that contain
#          instrument-level view copies (via :meth:`~PULSE.data.mltrace.MLTrace.view_copy`) of data from unit_input

#         :param unit_input: dictstream containing MLTrace-type objects
#         :type unit_input: PULSE.data.dictstream.DictStream
#         :return: 
#          - **unit_output** (*collections.deque*) -- double ended queue of :class:`~PULSE.data.window.Window` objects
#         """        
#         # Update window metadata with the update_window_tracker subroutine
#         self.update_window_tracker(unit_input)
#         # Iterate across site keys and associated sub-views of unit_input traces
#         unit_output = self.generate_windows(unit_input)
#         return unit_output
    
#     def capture_unit_output(self, unit_output):
#         """POLYMORPHIC METHOD

#         Last updated with :class:`~PULSE.mod.window.WindowMod`

#         Use the __iadd__ dunder method to append all entries to the **WindowMod.output** :class:`~collections.deque` attribute

#         :param unit_output: collection of :class:`~PULSE.data.window.Window` objects
#         :type unit_output: collections.deque
#         """        
#         if not isinstance(unit_output, deque):
#             self.Logger.critical(f'TypeError: unit_output is not type collections.deque. Exiting')
#             sys.exit(os.EX_DATAERR)
#         else:
#             self.output += unit_output
        
    
#     def update_window_tracker(self, unit_input):
#         """Update the **WindowMod.window_tracker** attribute by scanning over the entries in **unit_input**,
#         grouped by site, instrument, and model codes and determine if the primary trace in each sub-group has 
#         enough data to produce a new data window based on parameters set when this :class:`~PULSE.mod.window.WindowMod` object was initialized.

#         Readiness checks are the following
#         1) Does the mltrace start before or synchronously with the starttime for the next window?
#             If this is true, proceed
#             If the next window starts before the data, the window is advanced until this check is satisfied
#         2) Does the mltrace contain the endtime for the next window?
#             If this is true, proceed
#             If using eager_generation, traces that end before the end of the next window are still considered
#             If not using eager_generation, traces that end before the end of the next window end the assessment as not-ready
#         3) Does the mltrace have enough valid data to meet the threshold?
#             If this is true, mark as ready for window generation
#             Otherwise, mark as not-ready for window generation
            
#         :param unit_input: dictstream containing MLTraces
#         :type unit_input: PULSE.data.dictstream.DictStream
#         """        
#         if not all(isinstance(_e, MLTrace) for _e in unit_input):
#             self.Logger.critical('TypeError: not all elements of unit_input are type PULSE.data.mltrace.MLTrace. Exiting')
#             sys.exit(os.EX_DATAERR)
#         else:
#             pass
#         # Iterate across every mltrace in unit_input
#         for mlt in unit_input:
#             fnstring = f'{mlt.site}.{mlt.inst}'
#             fnstring += f'[{self.primary_components}{self.secondary_components}]'
#             fnstring += f'.{mlt.model}'

#             # Skip over traces that don't have the target component (or aliases thereof)
#             if mlt.comp not in self.target['components']:
#                 continue
#             # Case: Completely new site
#             elif mlt.site not in self.window_tracker.keys():
#                 self.window_tracker.update({mlt.site:
#                                                 {mlt.inst:
#                                                     {mlt.mod:
#                                                         {'ti': mlt.stats.starttime,
#                                                          'fnstr': fnstring,
#                                                          'primary_comp': mlt.comp,
#                                                          'ready': False}},
#                                                 't0': mlt.stats.starttime}})
#             # Case: Site already exists, but new instrument code appears
#             elif mlt.inst not in self.window_tracker[mlt.site].keys():
#                 self.window_tracker[mlt.site].update({mlt.inst:
#                                                         {mlt.mod:
#                                                             {'ti': self.window_tracker[mlt.site]['t0'],
#                                                              'fnstr': fnstring,
#                                                              'primary_comp': mlt.comp,
#                                                              'ready': False}}})
#             # Case: Site and instrument already exist, but new model code appears
#             elif mlt.mod not in self.window_tracker[mlt.site][mlt.inst].keys():
#                 self.window_tracker[mlt.site][mlt.inst].update({mlt.mod:
#                                                                     {'ti': self.window_tracker[mlt.site]['t0'],
#                                                                      'fnstr': fnstring,
#                                                                      'primary_comp': mlt.comp
#                                                                      'ready': False}})
#             # Case: Site-Instrument-Model combination already present in tracker
#             else:
#                 pass
            
#             ## ASSESS WINDOW GENERATION READINESS
#             # Get this site-instrument-model combination tracker
#             tracker = self.window_tracker[mlt.site][mlt.inst][mlt.mod]

#             # Check 1: Do the data start after the next window should start?
#             if mlt.stats.starttime > tracker['ti']:
#                 # Advance the next window starttime until it is >= the data starttime
#                 while mlt.stats.starttime > tracker['ti']:
#                     tracker['ti'] += self.advance_sec
#             # Data include the next window starttime
#             else:
#                 pass

#             # Check 2: Do the data end before the endtime of the next window?
#             if mlt.stats.endtime < tracker['ti'] + self.window_sec:
#                 # If using eager generation - proceed to next check
#                 if self.eager:
#                     pass
#                 # If not using eager generation - make sure ready = False & continue to next MLTrace
#                 else:
#                     tracker.update({'ready': False})
#                     continue
#             # Data include the next window endtime
#             else:
#                 pass
            
#             # Check 3: Does fraction of valid data meet/exceed the primary threshold?
#             fv = mlt.get_vfalid_subset(starttime=tracker['ti'], endtime = tracker['ti'] + self.window_sec)
#             # Threshold is met, update to ready
#             if fv >= self.target['threshold']:
#                 tracker.update({'ready': True})
#             # Otherwise, make sure it is marked as not-ready
#             else:
#                 tracker.update({'ready': False})


#     def generate_windows(self, unit_input):
#         """Using the readiness assessment provided by :meth:`~PULSE.mod.window.WindowMod.update_window_tracker`,
#         generate new :class:`~PULSE.data.window.Window` objects from trace-sets

#         :param unit_input: _description_
#         :type unit_input: _type_
#         :return: _description_
#         :rtype: _type_
#         """        
#         # Create unit_output holder object
#         unit_output = deque()
#         # Iterate across site codes and associated sub-dictionaries
#         for site, site_dict in self.window_tracker.items():
#             # Iterate across instrument codes and sub-dictionaries
#             for instrument, inst_dict in site_dict.items():
#                 # Iterate across model codes and get associated tracker
#                 for model, tracker in inst_dict.items():
#                     # If the tracker indicates enough data are present to make the next window
#                     if tracker['ready']:
#                         next_window_t0 = tracker['ti']
#                         next_window_t1 = next_window_t0 + self.window_sec
#                         _dst = unit_input.fnselect(tracker['fnstr'])
#                         traces = []
#                         for _mltb in _dst:
#                             mlt = _mltb.view_copy(
#                                 starttime = next_window_t0,
#                                 endtime = next_window_t1,
#                                 **self._vckwargs
#                             )
#                             mlt.stats.model = self.model_name
#                             traces.append(mlt)
#                         # Generate a new PULSE.data.window.Window object
#                         # TODO: migrate processing annotations to the Window Class
#                         window = Window(traces = traces,
#                                         primary_component = tracker['primary_comp'],
#                                         target_starttime=next_window_t0,
#                                         target_sampling_rate=self.target['sampling_rate'],
#                                         target_window_npts=self.target['npts'],
#                                         header={'primary_threshold': self.target['threshold']}
#                         # Append new window to unit_output
#                         unit_output.append(window)
#                         # Advance start time for the next window from this instrument & flag as not-ready
#                         self.window_tracker[site][instrument][model]['ti'] += self.advance_sec
#                         self.window_tracker[site][instrument][model].update({'ready': False})
#                     else:
#                         continue
#         return unit_output





    # def _should_this_iteration_run(self, input, input_measure, iter_number):
    #     """
    #     POLYMORPHIC METHOD
    #     Last updated with :class:`~PULSE.mod.window.WindowMod`

    #     unconditional pass - early stopping is handled by
    #     :meth:`~PULSE.mod.window.WindowMod._should_next_iteration_run`

    #     :param input: standard input
    #     :type input: PULSE.data.dictstream.DictStream
    #     :param iter_number: iteration number, unused
    #     :type iter_number: int
    #     :return status: should iterations continue in pulse, always True
    #     :rtype: bool
    #     """        
    #     status = True
    #     return status
    
    # def _unit_input_from_input(self, input):
    #     """
    #     POLYMORPHIC METHOD
    #     Last updated with :class:`~PULSE.mod.window.WindowMod

    #     unit_input is a view of the input and a sanity check is done to make sure
    #     that `input` is type :class:`~PULSE.data.dictstream.DictStream`

    #     :param input: input to :meth:`~PULSE.mod.window.WindowMod.pulse`
    #     :type input: PULSE.data.dictstream.DictStream
    #     :return unit_input: view of input to :meth:`~PULSE.mod.window.WindowMod.pulse`
    #     :rtype: PULSE.data.dictstream.DictStream
    #     """
    #     if isinstance(input, DictStream):
    #         unit_input = input
    #         return unit_input
    #     else:
    #         self.Logger.critical('TypeError',' input is not type DictStream')
    #         sys.exit(1)
    
    # def _unit_process(self, unit_input):
    #     """
    #     POLYMORPHIC METHOD
    #     Last updated with :class:`~PULSE.mod.window.WindowMod`
    #     Conducts the following steps
    #     1)  Scans across all MLTrace(Buff) entries in a :class:`~PULSE.data.dictstream.DictStream`
    #         and update entries in the **WindowMod.window_tracker** attribute using their metadata.
    #             uses :meth:`~PULSE.mod.window.WindowMod.__update_window_tracker`

    #     2)  nested iteration across site codes, instrument codes, and model codes in **WindowMod.window_tracker**
    #         to see if a given site/instrument/model combination in the tracker is flagged as 'ready'.

    #     3)  for 'ready' codes, generate a :class:`~PULSE.data.window.Window` object from the 
    #         matching entries

    #     Update the window_tracker with the contents of unit_input and then
    #     generate one window for each instrument in unit_input that has a 'ready'
    #     flag in window_tracker

    #     Newly generated windows are appended to WindowMod.output

    #     :param unit_input: view of a DictStream containing MLTrace(Buff)
    #     :type unit_input: PULSE.data.dictstream.DictStream
    #     """        
    #     unit_output = deque()
    #     # Update window tracker
    #     self._update_window_tracker(unit_input)
    #     # Conduct network-wide pulse
    #     # Iterate across site-level dictionary entries
    #     for site_key, site_dict in self.window_tracker.items():
    #         # Iterate across inst-level dictionary entries
    #         for inst_key, inst_dict in site_dict.items():
    #             # skip the t0:UTCDateTime entry
    #             if isinstance(inst_dict, dict):
    #                 # Iterate across mod-level dictionary entries
    #                 for mod_key, value in inst_dict.items():
    #                     # If this instrument record is ready to produce a window
    #                     if value['ready']:
    #                         fnstring = f'{site_key}.{inst_key}?.{mod_key}'
    #                         # Logger.info(f'generating window for {fnstring}')
    #                         next_window_ti = value['ti']
    #                         next_window_tf = next_window_ti + self.window_sec
    #                         # Subset to all traces for this instrument
    #                         _dictstream = unit_input.fnselect(fnstring)
    #                         # Create copies of trimmed views of Trace(Buffers)
    #                         traces = []
    #                         # Iterate over mltracebuff's
    #                         for _mltb in _dictstream:
    #                             # Create copies
    #                             mlt = _mltb.view_copy(
    #                                 starttime = next_window_ti,
    #                                 endtime = next_window_tf,
    #                                 pad=True,
    #                                 fill_value=None
    #                             )
    #                             # Update stats with the model name relevant to the windowing
    #                             mlt.stats.model = self.model_name
    #                             # Append mlt to traces
    #                             traces.append(mlt)
    #                         # Populate window traces and metadata
    #                         window = Window(
    #                             traces = traces,
    #                             header = {'reference_starttime': next_window_ti,
    #                                       'target_sampling_rate': self.target['sampling_rate'],
    #                                       'target_window_npts': self.target['npts'],
    #                                       'thresholds': self.target['thresholds'],
    #                                       'processing': [[self.__name__(full=False), UTCDateTime()]]},
    #                             ref_component=self.target['component']
    #                         )
    #                         # Append window to output
    #                         unit_output.append(window)
    #                         # Advance window start time in window_tracker
    #                         old_ti = self.window_tracker[site_key][inst_key][mod_key]['ti']
    #                         self.window_tracker[site_key][inst_key][mod_key]['ti'] += self.advance_sec
    #                         new_ti = self.window_tracker[site_key][inst_key][mod_key]['ti']
    #                         self.Logger.debug(f'New window for {window.stats.common_id} at {old_ti} += next at {new_ti}')
    #                         # Set ready flag to false for this site
    #                         self.window_tracker[site_key][inst_key][mod_key].update({'ready': False})
    #     return unit_output

                        
    # def _capture_unit_output(self, unit_output):
    #     """
    #     POLYMORPHIC METHOD
    #     Last updated with :class:`~PULSE.mod.window.WindowMod`

    #     Attaches the unit_output to **WindowMod.output** using :meth:`~collections.deque.__iadd__`
    #     and assesses if **WindowMod.output** is oversized (i.e., len(self.output) > self.max_output_size).

    #     If oversized, entries in **WindowMod.output** are removed using :meth:`~collections.deque.popleft`

    #     :param unit_out: deque of new :class:`~PULSE.data.window.Window` objects generated in _unit_process
    #     :type unit_out: collections.deque
    #     """  
    #     self.output += unit_output
    #     extra = len(self.output) - self.max_output_size
    #     # if extra > 0:
    #     #     Logger.warning(f'{self.__class__.__name__} object reached max_output_size. Deleting {extra} oldest values')
    #     while len(self.output) > self.max_output_size:
    #         self.output.popleft()


    # def _should_next_iteration_run(self, unit_output):
    #     """
    #     POLYMORPHIC METHOD
    #     Last updated with :class:`~PULSE.mod.window.WindowMod`

    #     Assess if any new windows were generated by the last call of :meth:`~PULSE.mod.window.WindowMod._unit_process`
    #     and signal continuation of :meth:`~PULSE.mod.window.WindowMod.pulse` iterations if at least one new window was generated
    #     I.e., len(unit_output) > 0 --> status = True

    #     :param unit_output: output from last call of _unit_process
    #     :type unit_output: collections.deque
    #     :return: status (*bool*) -- continuation signal
    #     """
    #     if len(unit_output) > 0:
    #         status = True
    #     else:
    #         status = False
    #     return status

    # def update_window_tracker(self, unit_input):
    #     """
    #     Core subroutine for :meth:`~PULSE.mod.window.Window._unit_process` used to update windowing metadata
    #     held in **WindowMod.window_tracker**

    #     Scans across MLTrace(Buff)s in an input DictStream with a component code matching the self.target['component']
    #     attribute of this WindowMod and populate new branches in the self.window_index attribute if new site.inst
    #     (Network.Station.Location.BandInstrument{ref}) Traces are found

    #     :param unit_input: DictStream object containing :class:`~PULSE.data.trace.Trace`-type objects
    #     :type unit_input: PULSE.data.dictstream.DictStream
    #     :
    #     """
    #     # Iterate across subset
    #     for mltrace in unit_input.traces.values():
    #         if not isinstance(mltrace, MLTrace):
    #             self.Logger.critical('TypeError','this build of WindowMod only works with PULSE.data.mltrace.MLTrace objects')
    #             sys.exit(1)
    #         # Get site, instrument, mod, and component codes from Trace
    #         site = mltrace.site
    #         inst = mltrace.inst
    #         comp = mltrace.comp
    #         mod = mltrace.mod
    #         ### WINDOW TRACKER NEW BRANCH SECTION ###
    #         # First, check if this is a reference component
    #         if comp not in self.target['component']:
    #             # If not a match, continue to next mltrace
    #             continue
    #         # Otherwise proceed
    #         else:
    #             pass
    #         # If new site in window_tracker
    #         if site not in self.window_tracker.keys():
    #             # TODO: WIP
    #             # # Check if this is the very first entry
    #             # if self.window_tracker == {}:
    #             #     self.next_available_start = mltrace.stats.starttime
    #             # else:
    #             #     # Get number of windows difference between trace start and next_available starttime
    #             #     dwindow = (mltrace.stats.starttime - self.next_available_start)//self.window_sec
    #             #     if dwindow > 0:
    #             #         self.next_available_start += dwindow*self.window_sec
    #             # Populate t0's
    #             self.window_tracker.update({site: 
    #                                             {inst: 
    #                                                 {mod: 
    #                                                     {'ti': mltrace.stats.starttime,
    #                                                      'ref': mltrace.id,
    #                                                      'ready': False}},
    #                                             't0': mltrace.stats.starttime}})
    #             #self.Logger.info(f'Added buffer tree for {site} - triggered by {mltrace.id}')
    #         # If site is in window_tracker
    #         else:
    #             # If inst is not in this site subdictionary
    #             if inst not in self.window_tracker[site].keys():
    #                 self.window_tracker[site].update({inst:
    #                                                     {mod:
    #                                                         {'ti': self.window_tracker[site]['t0'],
    #                                                          'ref': mltrace.id,
    #                                                          'ready': False}}})
    #                 #self.Logger.info(f'Added buffer branch {inst} to {site} tree - triggered by {mltrace.id}')
    #             # If inst is in this site subdictionary
    #             else:
    #                 # If mod is not in this inst sub-subdictionary
    #                 if mod not in self.window_tracker[site][inst].keys():
    #                     self.window_tracker[site][inst].update({mod:
    #                                                                 {'ti': self.window_tracker[site]['t0'],
    #                                                                  'ref': mltrace.id,
    #                                                                  'ready': False}})
                        
                        
    #         ### WINDOW TRACKER TIME INDEXING/WINDOWING STATUS CHECK SECTION ###   
    #         # Get window edge times
    #         next_window_ti = self.window_tracker[site][inst][mod]['ti']
    #         next_window_tf = next_window_ti + self.window_sec
    #         # If the endpoint exists in the buffer
    #         status = False
    #         # If eager, just need the window endtime in the scope of the trace
    #         if self.stance == 'eager':
    #             status = next_window_tf <= mltrace.stats.endtime
    #         # If patient, require at least one non-zero-fold sample past the end of the window
    #         elif self.stance == 'patient':
    #             if next_window_tf <= mltrace.stats.endtime:
    #                 _, fold_view = mltrace.get_subset_view(starttime=next_window_tf)
    #                 status = any(fold_view > 0)

    #         # If the mltrace has reached or exceeded the endpoint of the next window
    #         if status:
    #             # Get valid fraction for proposed window in Trace
    #             fv = mltrace.get_vfalid_subset(starttime=next_window_ti, endtime=next_window_tf)
    #             # If threshold passes
    #             if fv >= self.target['thresholds']['ref']:
    #                 # set (window) ready flag to True
    #                 self.window_tracker[site][inst][mod].update({'ready': True})
    #                 # And continue to next mltrace
    #                 continue
    #             # If threshold fails
    #             else:
    #                 # If the window was previously approved, recind approval
    #                 if self.window_tracker[site][inst][mod]['ready']:
    #                     self.window_tracker[site][inst][mod].update({'ready': False})

    #                 # If data start after the proposed window, increment window index to catch up
    #                 if next_window_ti < mltrace.stats.starttime:
    #                     # Determine the number of advances that should be applied to catch up
    #                     nadv = 1 + (mltrace.stats.starttime - next_window_ti)//self.advance_sec
    #                     # Apply advance
    #                     self.window_tracker[site][inst][mod]['ti'] += nadv*self.advance_sec
    #                     next_window_ti += nadv*self.advance_sec
    #                     next_window_tf += nadv*self.advance_sec
    #                     # If the new window ends inside the current data
    #                     if mltrace.stats.endtime >= next_window_tf and self.stance == 'eager':
    #                         # Consider re-approving the window for copying + trimming
    #                         fv = mltrace.get_vfalid_subset(starttime=next_window_ti,
    #                                                     endtime=next_window_tf)
    #                         # If window passes threshold, re-approve
    #                         if fv >= self.target['thresholds']['ref']:
    #                             self.window_tracker[site][inst][mod].update({'ready': True})
    #                     # Otherwise preserve dis-approval of window generation (ready = False) for now


            

    # def __repr__(self):
    #     """
    #     Provide a user-friendly string representation of this WindowMod's parameterization
    #     and state.
    #     """
    #     rstr = f'WindowMod for model architecture "{self.model_name}"\n'
    #     rstr += 'Ref.:'
    #     for _k, _v in self.target.items():
    #         if _k in ['overlap', 'npts']:
    #             rstr += f' {_k}:{_v} samples |'
    #         elif _k == "sampling_rate":
    #             rstr += f' {_k}: {_v} sps |'
    #         elif _k == 'component':
    #             rstr += f' {_k}:"{_v}" |'
    #         elif _k == 'threshold':
    #             rstr += f' {_k}:{_v:.3f}'
    #     rstr += '\nAliases '
    #     for _k, _v in self.aliases.items():
    #         rstr += f' "{_k}":{[__v for __v in _v]}'
    #     rstr += f'\nIndex: {len(self.window_tracker)} sites |'
    #     _ni = 0
    #     for _v in self.window_tracker.values():
    #         _ni += len(_v) - 1
    #     types = {}
    #     for _x in self.output:
    #         if type(_x).__name__ not in types.keys():
    #             types.update({type(_x).__name__: 1})
    #         else:
    #             types[type(_x).__name__] += 1
    #     rstr += f' {_ni} instruments'
    #     rstr += '\nQueue: '
    #     for _k, _v in types.items():
    #         rstr += f'{_v} {_k}(s) | '
    #     if len(types) > 0:
    #         rstr = rstr[:-3]
    #     else:
    #         rstr += 'Nothing'
            
    #     return rstr