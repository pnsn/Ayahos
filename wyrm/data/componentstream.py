import fnmatch, logging, torch, copy
import numpy as np
import pandas as pd
import seisbench.models as sbm
from obspy import Trace, Stream, UTCDateTime
from wyrm.data.dictstream import DictStream, DictStreamStats, _add_processing_info
from wyrm.data.mltrace import MLTrace, MLTraceBuffer
from wyrm.util.pyew import wave2mltrace

###############################################################################
# Component Stream Header Class Definition ####################################
###############################################################################

class ComponentStreamStats(DictStreamStats):
    # NTS: Deepcopy is necessary to not overwrite _types and defaults for parent class
    _types = copy.deepcopy(DictStreamStats._types)
    _types.update({'aliases': dict,
            'reference_starttime': (UTCDateTime, type(None)),
            'reference_npts': (int, type(None)),
            'reference_sampling_rate': (float, type(None))})
    defaults = copy.deepcopy(DictStreamStats.defaults)
    defaults.update({'aliases': {},
                    'reference_starttime': None,
                    'reference_sampling_rate': None,
                    'reference_npts': None})
    
    def __init__(self, header={}):
        # Initialize super + updates to class attributes
        super(ComponentStreamStats, self).__init__()
        # THEN update self with header inputs
        self.update(header)

    def __str__(self):
        prioritized_keys = ['reference_id',
                            'reference_starttime',
                            'reference_sampling_rate',
                            'reference_npts',
                            'aliases',
                            'processing']

        hidden_keys = ['min_starttime',
                        'max_starttime',
                        'min_endtime',
                        'max_endtime']
        return self._pretty_str(prioritized_keys, hidden_keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
    
###############################################################################
# Component Stream Class Definition ###########################################
###############################################################################
        
class ComponentStream(DictStream):

    def __init__(self, traces=None, header={}, component_aliases={'Z':'Z3', 'N':'N1', 'E':'E2'}, **options):
        """
        Initialize a wyrm.core.dictstream.ComponentStream object

        :: INPUTS ::
        :param traces: [obspy.core.trace.Trace] or [list]/[Stream] thereof
                        trace-type object(s) to append to this ComponentStream, if they pass validation cross
                        checks with the component_aliases' contents
        :param header: [dict] inputs to pass to ComponentStreamStats.__init__
        :param component_aliases: [dict] mapping between aliases (keys) and native component codes (characters in each value)
                        that are mapped to those aliases
        :param **options: [kwargs] collector for key-word arguments to pass to the ComponentStream.__add__ method
                        inherited from DictStream when 
        """
        super().__init__()
        self.stats = ComponentStreamStats(header=header)

        # component_aliases compatability checks
        if isinstance(component_aliases, dict):
            if all(isinstance(_k, str) and len(_k) == 1 and isinstance(_v, str) for _k, _v in component_aliases.items()):
                self.stats.aliases = component_aliases
            else:
                raise TypeError('component_aliases keys and values must be type str')
        else:
            raise TypeError('component aliases must be type dict')

        if traces is not None:
            if isinstance(traces, Trace):
                traces = [traces]
            elif isinstance(traces, (Stream, list)):
                ref_type = type(traces[0])
                if not all(isinstance(_tr, ref_type) for _tr in traces):
                    raise TypeError('all input traces must be of the same type')
                else:
                    self.ref_type = ref_type
            else:
                raise TypeError("input 'traces' must be Trace-like, Stream-like, or list-like")
            # Run validate_ids and continue if error isn't kicked
            self.validate_ids(traces)
            # Add traces using the ComponentStream __add__ method that converts non MLTrace objects into MLTrace objects
            self.__add__(traces, **options)

    def __add__(self, other, **options):
        """
        Wrapper around the inherited DictStream.__add__ method that fixes
        the key_attr to 'component'

        also see wyrm.core.dictstream.DictStream.__add__()
        """
        super().__add__(other, key_attr='component', **options)

    def __repr__(self, extended=False):
        """
        Provide a user-friendly string representation of the contents and key parameters of this
        ComponentStream object. 

        :: INPUTS ::
        :param extended: [bool] - option to show an extended form of the ComponentStream should 
                            there be a large number of unique component codes (an uncommon use case)
        """
        rstr = self.stats.__str__()
        if len(self.traces) > 0:
            id_length = max(len(_tr.id) for _tr in self.traces.values())
        else:
            id_length=0
        if len(self.traces) > 0:
            rstr += f'\n{len(self.traces)} {type(self[0]).__name__}(s) in {type(self).__name__}\n'
        else:
            rstr += f'\nNothing in {type(self).__name__}\n'
        if len(self.traces) <= 20 or extended is True:
            for _l, _tr in self.traces.items():
                rstr += f'{_l:} : {_tr.__str__(id_length)}\n'
        else:
            _l0, _tr0 = list(self.traces.items())[0]
            _lf, _trf = list(self.traces.items())[-1]
            rstr += f'{_l0:} : {_tr0.__repr__(id_length=id_length)}\n'
            rstr += f'...\n({len(self.traces) - 2} other traces)\n...\n'
            rstr += f'{_lf:} : {_trf.__repr__(id_length=id_length)}\n'
            rstr += f'[Use "print({type(self).__name__}.__repr__(extended=True))" to print all labels and MLTraces]'
        return rstr

    def _add_trace(self, other, **options):
        # If potentially appending a wave
        if isinstance(other, dict):
            other = wave2mltrace(other)
        # If appending a trace-type object
        elif isinstance(other, Trace):
            # If it isn't an MLTrace, __init__ one from data & header
            if not isinstance(other, MLTrace):
                other = MLTrace(data=other.data, header=other.stats)
            else:
                pass
        # Otherwise
        else:
            raise TypeError(f'other {type(other)} not supported.')
        # Ensure that the trace is converted to MLTrace
        if isinstance(other, MLTrace):
            # Get other's component code
            comp = other.comp
            # Get other's model code
            mod = other.mod
            key = f'{comp}.{mod}'
            # If the component code is not in alias keys
            if comp not in dict(self.stats.aliases).keys():
                # Iterate across alias keys and aliases
                for _k, _v in dict(self.stats.aliases).items():
                    # If a match is found
                    if comp in _v:
                        # And the alias is not in self.traces.keys() - use update
                        if _k not in self.traces.keys():
                            self.traces.update({_k: other})
                        # Otherwise try to add traces together - allowing MLTrace.__add__ to handle the error raising
                        else:
                            self.traces[_k].__add__(other, **options)
                        self.stats.update_time_range(other)
            else:
                if comp not in self.traces.keys():
                    self.traces.update({comp: other})
                else:
                    self.traces[comp].__add__(other, **options)
                self.stats.update_time_range(other)

    def enforce_alias_keys(self):
        """
        Enforce aliases
        """
        for _k in self.traces.keys():
            if _k not in self.stats.aliases.keys():
                for _l, _w in self.stats.aliases.items():
                    if _k in _w:
                        _tr = self.traces.pop(_k)
                        self.traces.update({_l: _tr})

    def validate_ids(self, traces):
        """
        Check id strings for traces against WindowStream.stats.reference_id
        :: INPUTS ::
        :param traces: [list-like] of [obspy.core.trace.Trace-like] or individual
                        objects thereof
        """
        # Handle case where a single trace-type object is passed to validate_ids
        if isinstance(traces, Trace):
            traces = [traces]
        # if there is already a non-default reference_id, use that as reference
        if self.stats.reference_id != self.stats.defaults['reference_id']:
            ref = self.stats.reference_id
        # Otherwise use the first trace in traces as the template
        else:
            tr0 = traces[0]
            # If using wyrm.core.trace.MLTrace(Buffer) objects, as above with the 'mod' extension
            if isinstance(tr0, MLTrace):
                ref = f'{tr0.site}.{tr0.inst}?.{tr0.mod}'
                        # If using obspy.core.trace.Trace objects, use id with "?" for component char
            elif isinstance(tr0, Trace):
                ref = tr0.id[:-1]+'?'
        # Run match on all trace ids
        matches = fnmatch.filter([_tr.id for _tr in traces], ref)
        # If all traces conform to ref
        if all(_tr.id in matches for _tr in traces):
            # If reference_id 
            if self.stats.reference_id == self.stats.defaults['reference_id']:
                self.stats.reference_id = ref
        
        else:
            raise KeyError('Trace id(s) do not conform to reference_id: "{self.stats.reference_id}"')

    ###############################################################################
    # FILL RULE METHODS ###########################################################
    ###############################################################################
    @_add_processing_info
    def apply_fill_rule(self, rule='zeros', ref_component='Z', other_components='NE', ref_thresh=0.9, other_thresh=0.8):
        if ref_component not in self.traces.keys():
            raise KeyError('reference component {ref_component} is not present in traces')
        else:
            thresh_dict = {ref_component: ref_thresh}
            thresh_dict.update({_c: other_thresh for _c in other_components})
        # Check if data meet requirements before triggering fill rule
        checks = []
        # Check if all components are present in traces
        checks.append(self.traces.keys() == thresh_dict.keys())
        # Check if all valid data fractions meet/exceed thresholds
        checks.append(all(self[_k].fvalid >= thresh_dict[_k] for _k in self.trace.keys()))
        if all(checks):
            pass
        elif rule == 'zeros_wipe':
            self._apply_zeros(ref_component, thresh_dict, method='wipe')
        elif rule == 'zeros_fill':
            self._apply_zeros(ref_component, thresh_dict, method='fill')
        elif rule == 'clone_ref':
            self._apply_clone_ref(ref_component, thresh_dict)
        elif rule == 'clone_other':
            self._apply_clone_other(ref_component, thresh_dict)
        else:
            raise ValueError(f'rule {rule} not supported. Supported values: "zeros", "clone_ref", "clone_other"')

    @_add_processing_info
    def _apply_zeros(self, ref_component, thresh_dict, method='fill'):
        """
        Apply the channel filling rule "zero" (e.g., Retailleau et al., 2022)
        where both "other" (horzontal) components are set as zero-valued traces
        if one or both are missing/overly gappy.

        0-valued traces are assigned fold values of 0 to reflect the absence of
        added information.

        :: INPUTS ::
        :param ref_component: [str] single character string corresponding to
                        the KEYED (aliased) component code of the reference trace
        :param thresh_dict: [dir] directory with keys matching keys in 
                        self.traces.keys() (i.e., alised component characters)
                        and values \in [0, 1] representing fractional completeness
                        thresholds below which the associated component is rejected
        """
        ref_tr = self[ref_component]
        if ref_tr.fvalid < thresh_dict[ref_component]:
            raise ValueError('insufficient valid data in reference trace')
        else:
            pass
        for _k in thresh_dict.keys():
            if method == 'wipe':
                tr0 = ref_tr.copy().to_zero(method='both').set_comp(_k)
                self.traces.update({_k: tr0})
            elif method == 'fill':
                if _k in self.traces.keys():
                    if self.traces[_k].fvalid < thresh_dict[_k]:
                        tr0 = ref_tr.copy().to_zero(method='both').set_comp(_k)
                        self.traces.update({_k: tr0})


    @_add_processing_info
    def _apply_clone_ref(self, ref_component, thresh_dict):
        """
        Apply the channel filling rule "clone reference" (e.g., Ni et al., 2023)
        where the reference channel (vertical component) is cloned onto both
        horizontal components if one or both horizontal (other) component data
        are missing or are sufficiently gappy. 
        
        Cloned traces are assigned fold values of 0 to reflect the absence of
        additional information contributed by this trace.

        :: INPUTS ::
        :param ref_component: [str] single character string corresponding to
                        the KEYED (aliased) component code of the reference trace
        :param thresh_dict: [dir] directory with keys matching keys in 
                        self.traces.keys() (i.e., alised component characters)
                        and values \in [0, 1] representing fractional completeness
                        thresholds below which the associated component is rejected
        """
        ref_tr = self[ref_component]
        if ref_tr.fvalid < thresh_dict[ref_component]:
            raise ValueError('insufficient valid data in reference trace')
        else:
            pass
        for _k in thresh_dict.keys():
            trC = ref_tr.copy().to_zero(method='fold').set_comp(_k)
            self.traces.update({_k: trC})


    @_add_processing_info    
    def _apply_clone_other(self, ref_component, thresh_dict):
        """
        Apply the channel filling rule "clone other" (e.g., Lara et al., 2023)
        where the reference channel (vertical component) is cloned onto both
        horizontal components if both "other" component traces (horizontal components)
        are missing or are sufficiently gappy, but if one "other" component is present
        and valid, clone that to the missing/overly-gappy other "other" component.
        
        Cloned traces are assigned fold values of 0 to reflect the absence of
        additional information contributed by this trace.

        :: INPUTS ::
        :param ref_component: [str] single character string corresponding to
                        the KEYED (aliased) component code of the reference trace
        :param thresh_dict: [dir] directory with keys matching keys in 
                        self.traces.keys() (i.e., alised component characters)
                        and values \in [0, 1] representing fractional completeness
                        thresholds below which the associated component is rejected
        """
        # Run through each component and see if it passes thresholds
        pass_dict = {}
        for _k, _tr in self.traces.items():
            pass_dict.update({_k: _tr.fvalid >= thresh_dict[_k]})

        # If the reference component is present but fails to pass, kick error
        if ref_component in pass_dict.keys():
            if not pass_dict[ref_component]:
                raise ValueError('insufficient valid data in reference trace')
            else:
                pass
        # If the reference component is absent, kick error
        else:
            raise KeyError("reference component is not in this ComponentStream's keys")
        
        # If all expected components are present
        if pass_dict.keys() == thresh_dict.keys():
            # If all components pass thresholds
            if all(pass_dict.values()):
                # Do nothing
                pass

            # If at least one "other" component passed checks
            elif any(_v for _k, _v in pass_dict.items() if _k != ref_component):
                # Iterate across components in 
                for _k, _v in pass_dict.items():
                    # If not the reference component and did not pass
                    if _k != ref_component and not _v:
                        # Grab component code that will be cloned over
                        cc = _k
                    # If not the reference component and did pass
                    if _k != ref_component and _v:
                        # Create a clone of the passing "other" component
                        trC = _v.copy()
                # Zero out the fold of the cloned component and overwrite it's component code
                trC.to_zero(method='fold').set_comp(cc)
                # Write cloned, relabeled "other" trace to the failing trace's position
                self.traces.update({cc: trC})

            # If only the reference trace passed, run _apply_clone_ref() method instead
            else:
                self._apply_clone_ref(ref_component, thresh_dict)

        # If at least one "other" component is present
        elif ref_component in pass_dict.keys():
            # If both ref and at "other" pass, create clone of "other"
            if all(pass_dict.items()):
                for _c in thresh_dict.keys():
                    if _c not in pass_dict.keys():
                        cc = _c
                    elif _c != ref_component:
                        trC = self[_c].copy()
                trC.to_zero(method='fold').set_comp(cc)
                self.traces.update({cc: trC})
            # If the single "other" trace does not pass, use _apply_clone_ref method
            else:
                self._apply_clone_ref(ref_component, thresh_dict)
        # If only the reference component is present & passing
        else:
            self._apply_clone_ref(ref_component, thresh_dict)
    
    ###############################################################################
    # Synchronization Methods #####################################################
    ###############################################################################
            
    def check_windowing_status(self,
                               reference_starttime=None,
                               reference_sampling_rate=None,
                               reference_npts=None,
                               mode='summary'):
        """
        Check if the data timing and sampling in this ComponentStream are synchronized
        with the reference_* [starttime, sampling_rate, npts] attributes in its Stats object
        or those specified as arguments in this check_sync() call. Options are provided for
        different slices of the boolean representation of trace-attribute-reference sync'-ing

        This method also checks if data are masked, using the truth of np.ma.is_masked(tr.data)
        as the output

        :: INPUTS ::
        :param reference_starttime: None - use self.stats.reference_starttime
                                    UTCDateTime - use this value for sync check on starttime
        :param reference_sampling_rate: None - use self.stats.reference_sampling_rate
                                    float - use this value for sync check on sampling_rate
        :param reference_npts: None - use self.stats.reference_npts
                                    int - use this value for sync check on npts
        :param mode: [str] output mode
                        'summary' - return the output of bool_array.all()
                        'trace' - return a dictionary of all() outputs of elements 
                                subset by trace
                        'attribute' - return a dictionary of all() outputs of elements
                                subset by attribute
                        'full' - return a pandas DataFrame with each element, labeled
                                with trace component codes (columns) and 
                                attribute names (index)
        :: OUTPUT ::
        :return status: [bool] - for mode='summary'
                        [dict] - for mode='trace' or 'attribute'
                        [DataFrame] - for mode='full'
        """
        ref = {}
        if reference_starttime is not None:
            ref.update({'starttime': reference_starttime})
        elif self.stats.reference_starttime is not None:
            ref.update({'starttime': self.stats.reference_starttime})
        else:
            raise ValueError('Neither stats.reference_starttime or kwarg reference_starttime are assigned')
        
        if reference_sampling_rate is not None:
            ref.update({'sampling_rate': reference_sampling_rate})
        elif self.stats.reference_sampling_rate is not None:
            ref.update({'sampling_rate': self.stats.reference_sampling_rate})
        else:
            raise ValueError('Neither stats.reference_sampling_rate or kwarg reference_sampling_rate are assigned')
             
        if reference_npts is not None:
            ref.update({'npts': reference_npts})
        elif self.stats.reference_npts is not None:
            ref.update({'npts': self.stats.reference_npts})
        else:
            raise ValueError('Neither stats.reference_npts or kwarg reference_npts are assigned')
        
        holder = []
        for _tr in self.traces.values():
            line = []
            for _k, _v in ref.items():
                line.append(getattr(_tr.stats,_k) == _v)
            line.append(np.ma.is_masked(_tr.data))
            holder.append(line)
        
        bool_array = np.array(holder)
        if mode == 'summary':
            status=bool_array.all()
        elif mode == 'attribute':
            status = dict(zip(list(ref.keys()) + ['masked'], bool_array.all(axis=0)))
        elif mode == 'trace':
            status = dict(zip(self.traces.keys(), bool_array.all(axis=1)))
        elif mode == 'full':
            status = pd.DataFrame(bool_array,
                                  columns=list(ref.keys()) + ['masked'],
                                  index=self.traces.keys()).T
        return status

    def treat_gaps(self,
                   filterkw={'type': 'bandpass', 'freqmin': 1, 'freqmax': 45},
                   detrendkw={'type': 'linear'},
                   resample_method='resample',
                   resamplekw={},
                   taperkw={},
                   mergekw={},
                   trimkw={}):
        # Get reference values from header
        ref = {}
        for _k in ['reference_starttime','reference_npts','reference_sampling_rate']:
            ref.update({'_'.join(_k.split("_")[1:]): self.stats[_k]})
        resamplekw.update({'sampling_rate':ref['sampling_rate']})
        trimkw.update({'starttime': ref['starttime'],
                        'endtime': ref['starttime'] + (ref['npts'] - 1)/ref['sampling_rate'],
                        'fill_value': 0})
        for tr in self.traces.values():
            tr.treat_gaps(filterkw = filterkw,
                          detrendkw = detrendkw,
                          resample_method=resample_method,
                          resamplekw=resamplekw,
                          taperkw=taperkw,
                          mergekw=mergekw,
                          trimkw=trimkw)
        return self
            

    # def treat_gaps(self):
    #     return None
    
    # def sync_sampling(self,
    #                   resampling_method='resample',
    #                   resampling_kw={},
    #                   )

    ###############################################################################
    # ComponentStream to Tensor Methods ###########################################
    ###############################################################################
            
    def ready_to_burn(self, model):
        """
        Assess if the data contents of this ComponentStream are ready
        to convert into a torch.Tensor given a particular seisbench model

        NOTE: This inspects that the dimensionality, timing, sampling, completeness,
             and component aliasing of the contents of this ComponentStream are
             compliant with reference values in the metadata and in the input `model`'s
             metadata
              
             It does not check if the data have been processed: filtering, tapering, 
                normalization, etc. 

        :: INPUT ::
        :param model: [seisbench.models.WaveformModel] initialized model
                        object that prediction will be run on 
                        NOTE: This is really a child-class of sbm.WaveformModel
        :: OUTPUT ::
        :return status: [bool] - is this ComponentStream ready for conversion
                                using ComponentStream.to_torch(model)?
        """
        # model compatability check
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError
        elif model.name == 'WaveformModel':
            raise ValueError('WaveformModel baseclass objects do not provide a viable prediciton method - use a child class thereof')
        
        # Check that data are not masked
        if any(isinstance(_tr.data, np.ma.MaskedArray) for _tr in self):
            status = False
        # Check starttime sync
        elif not all(_tr.stats.starttime == self.stats.reference_starttime for _tr in self):
            status = False
        # Check npts is consistent
        elif not all(_tr.stats.npts == model.in_samples for _tr in self):
            status = False
        # Check that all components needed in model are represented in the aliases
        elif not all(_k in model.component_order for _k in self.keys()):
            status = False
        # Check that sampling rate is sync'd
        elif not all(_tr.stats.sampling_rate == self.stats.reference_sampling_rate for _tr in self):
            status = False
        # Passing (or really failing) all of the above
        else:
            status = True
        return status
    
    def to_tensor(self, model):
        """
        Convert the data contents of this ComponentStream into a numpy array that
        conforms to the component ordering required by a seisbench WaveformModel
        object
        :: INPUT ::
        :param model: []
        """
        if not self.ready_to_burn(model):
            raise ValueError('This ComponentStream is not ready for conversion to a torch.Tensor')
        
        npy_array = np.c_[[self[_c].data for _c in model.component_order]]
        tensor = torch.Tensor(npy_array)
        return tensor

    def collapse_fold(self):
        """
        Collapse fold vectors into a single vector by addition, if all traces have equal npts
        """
        npts = self[0].stats.npts
        if all(_tr.stats.npts == npts for _tr in self):
            addfold = np.sum(np.c_[[_tr.fold for _tr in self]], axis=0)
            return addfold
        else:
            raise ValueError('not all traces in this ComponentStream have matching npts')