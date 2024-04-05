import fnmatch, logging, torch, copy
import numpy as np
import pandas as pd
import seisbench.models as sbm
from obspy import Trace, Stream, UTCDateTime
from wyrm.data.dictstream import DictStream, DictStreamStats, _add_processing_info
from wyrm.data.mltrace import MLTrace
from wyrm.util.pyew import wave2mltrace

###############################################################################
# Component Stream Header Class Definition ####################################
###############################################################################
## TODO: Nix primary_components for now - too general.

class ComponentStreamStats(DictStreamStats):
    # NTS: Deepcopy is necessary to not overwrite _types and defaults for parent class
    _types = copy.deepcopy(DictStreamStats._types)
    _types.update({'primary_component': str,
                   'aliases': dict,
                   'reference_starttime': (UTCDateTime, type(None)),
                   'reference_npts': (int, type(None)),
                   'reference_sampling_rate': (float, type(None))})
    defaults = copy.deepcopy(DictStreamStats.defaults)
    defaults.update({'primary_component': 'Z',
                     'aliases': {'Z': 'Z3',
                                 'N': 'N1',
                                 'E': 'E2'},
                     'reference_starttime': None,
                     'reference_sampling_rate': None,
                     'reference_npts': None})
    
    def __init__(self, header={}):
        # Initialize super + updates to class attributes
        super(ComponentStreamStats, self).__init__()
        # THEN update self with header inputs
        self.update(header)

    def __str__(self):
        prioritized_keys = ['primary_component',
                            'common_id',
                            'aliases',
                            'reference_starttime',
                            'reference_sampling_rate',
                            'reference_npts',
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

    def __init__(
            self,
            traces,
            primary_component='Z',
            header={},
            **options):
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
                        inherited from DictStream 
        """
        # Initialize & inherit from DictStream
        super().__init__()
        # Initialize Stream Header
        self.stats = ComponentStreamStats(header=header)
        if isinstance(primary_component, str):
            if primary_component in self.stats.aliases.keys():
                self.stats.primary_component = primary_component
            else:
                raise ValueError(f'primary_component must be a key value in ComponentStream.stats.aliases')
        else:
            raise TypeError('primary_component must be type str')
        
        if isinstance(traces, Trace):
            traces = [traces]
        elif isinstance(traces, (Stream, list, tuple)):
            if not all(isinstance(tr, Trace) for tr in traces):
                raise TypeError('all input traces must be type MLTrace')
        else:
            raise TypeError("input 'traces' must be a single MLTrace or iterable set of MLTrace objects")
        # Add traces using the ComponentStream __add__ method that converts non MLTrace objects into MLTrace objects
        # if self.validate_trace_ids(self, other=traces)
        self.extend(traces, **options)
        self.stats.common_id = self.get_common_id()
        # if self.ref['component'] in self.traces.keys():
        #     self.stats.reference_id = self.traces[self.ref['component']].id

    def extend(self, traces, **options):
        for tr in traces:
            if not isinstance(tr, MLTrace):
                tr = MLTrace(tr)
            comp = tr.stats.component
            # If component is a primary key
            if comp in self.stats.aliases.keys():
                # And that key is not in the current holdings
                if comp not in self.traces.keys():
                    self.traces.update({comp: tr})
                else:
                    self.traces[comp].__add__(tr, **options)
                self.stats.update_time_range(self.traces[comp])
            # If component is an aliased key    
            elif comp in ''.join(self.stats.aliases.values()):
                # Get the matching alias/key pair
                for _k, _v in self.stats.aliases.items():
                    if comp in _v:
                        # If primary key not in current holdings
                        if _k not in self.traces.keys():
                            self.traces.update({_k: tr})
                        else:
                            self.traces[_k].__add__(tr, **options)
                        self.stats.update_time_range(self.traces[_k])
                        break
                    else:
                        pass

            else:
                raise ValueError('component code for {tr.id} is not in the self.stats.aliases dictionary')


    # def validate_traces(self, other=None):
    #     if len(self.traces) == 0:
    #         if other is None:
    #             return True
    #         elif isinstance(other, MLTrace):
    #             return True
    #         elif isinstance(other, (Stream, list, tuple)):
    #             if not all(isinstance(tr, MLTrace) for tr in other):
    #                 return False
    #             else:
    #                 bool_list = []
    #                 for _i, itr in enumerate(other):
    #                     for _j, jtr in enumerate(other):
    #                         if _i > _j:
    #                             bool_line = [getattr(itr, _k) == getattr(jtr, _k) for _k in ['site','inst','mod']]
    #                             bool_list.append(all(bool_line))
    #                 return all(bool_list)
    #     if len(self.traces) > 0:
    #         bool_list = []
    #         for _i, itr in self.traces.items():
    #             for _j, jtr in self.traces.items():
    #                 if _i > _j:
    #                     bool_line = [getattr(itr, _k) == getattr(jtr, _k) for _k in ['site','inst','mod']]
    #                     bool_list.append(all(bool_line))
    #         internal_status = all(bool_list)
        
    #         if other is None:
    #             return internal_status
    #         elif isinstance(other, MLTrace):


        
    #     if not all(bool_list):
    #         raise ValueError('Input trace IDs (excluding component code) mismatch')
    #     else:
    #         self.extend(traces, **options)


            
    # def extend(self, other, **options):
    #     """
    #     Wrapper around the inherited MLTrace.__add__ method that fixes
    #     the key_attr to 'component'. 

    #     NOTE: Inheritance has an intermediate step where
    #         DictStream.__add__() wraps MLTrace.__add__

    #     also see wyrm.data.mltrace.MLTrace.__add__()
    #     """

    #     super().extend(other, key_attr='component', **options)
    
    # def __add__(self, other, **options):
    #     self.extend(other, **options)


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

    

    # def validate_ids(self, traces):
    #     """
    #     Check id strings for traces against ComponentStream.stats.reference_id
    #     :: INPUTS ::
    #     :param traces: [list-like] of [obspy.core.trace.Trace-like] or individual
    #                     objects thereof
    #     """
    #     # Handle case where a single trace-type object is passed to validate_ids
    #     if isinstance(traces, Trace):
    #         traces = [traces]
    #         # TODO: Shift reference_id use here to common_id, and then add cross-checks for new def of reference_id
    #     # if there is already a non-default reference_id, use that as reference
    #     if self.stats.common_id != self.stats.defaults['common_id']:
    #         ref = self.stats.common_id
    #     # Otherwise use the first trace in traces as the template
    #     else:
    #         tr0 = traces[0]
    #         # If using wyrm.core.trace.MLTrace(Buffer) objects, as above with the 'mod' extension
    #         if isinstance(tr0, MLTrace):
    #             ref = f'{tr0.site}.{tr0.inst}?.{tr0.mod}'
    #                     # If using obspy.core.trace.Trace objects, use id with "?" for component char
    #         elif isinstance(tr0, Trace):
    #             ref = tr0.id[:-1]+'?'
    #     # Run match on all trace ids
    #     matches = fnmatch.filter([_tr.id for _tr in traces], ref)
    #     # If all traces conform to ref
    #     if all(_tr.id in matches for _tr in traces):
    #         # If reference_id 
    #         if self.stats.reference_id == self.stats.defaults['reference_id']:
    #             self.stats.reference_id = ref
        
    #     else:
    #         raise KeyError('Trace id(s) do not conform to reference_id: "{self.stats.reference_id}"')

    ###############################################################################
    # FILL RULE METHODS ###########################################################
    ###############################################################################
    # @_add_processing_info
    def apply_fill_rule(self, rule='zeros', ref_thresh=0.9, other_thresh=0.8):
        thresh_dict = {}
        for _k in self.stats.aliases.keys():
            if _k == self.stats.primary_component:
                thresh_dict.update({_k: ref_thresh})
            else:
                thresh_dict.update({_k: other_thresh})

        # Check if all expected components are present and meet threshold
        checks = [thresh_dict.keys() == self.traces.keys(),
                  all(_v.get_fvalid_subset() >= thresh_dict[_k] for _k, _v in self.traces.items())]
        # If so, do nothing
        if all(checks):
            pass
        # Otherwise apply rule
        elif rule == 'zeros':
            self._apply_zeros(thresh_dict)
        elif rule == 'clone_ref':
            self._apply_clone_ref(thresh_dict)
        elif rule == 'clone_other':
            self._apply_clone_other(thresh_dict)
        else:
            raise ValueError(f'rule {rule} not supported. Supported values: "zeros", "clone_ref", "clone_other"')

    # @_add_processing_info
    def _apply_zeros(self, thresh_dict): #, method='fill'):
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

        (OBSOLITED)
        # :param method: [str] method for filling behavior
        #             Supported:
        #                 'wipe' - without exception, convert all "other" component
        #                         traces into 0-data 0-fold MLTrace objects
        #                 'fill' - if a given existing "other" trace falls below
        #                             the threshold, convert it into a 0-data 0-fold
        #                             MLTrace object
        """
        ref_comp = self.stats.primary_component
        # Get reference trace
        ref_tr = self.traces[ref_comp]
        # Safety catch that at least the reference component does have enough data
        if ref_tr.get_fvalid_subset() < thresh_dict[ref_comp]:
            raise ValueError('insufficient valid data in reference trace')
        else:
            pass
        # Iterate across entries in the threshold dictionary
        for _k in thresh_dict.keys():
            # For "other" components
            if _k != ref_comp:
                # Create copy of reference trace
                tr0 = ref_tr.copy()
                # Relabel
                tr0.set_comp(_k)
                # Set data and fold to zero
                tr0.to_zero(method='both')
                # Update 
                self.traces.update({_k: tr0})

    # @_add_processing_info
    def _apply_clone_primary(self, thresh_dict):
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
        # Get reference trace
        ref_comp = self.stats.primary_component
        ref_tr = self[ref_comp]
        # Get 
        if ref_tr.fvalid < thresh_dict[ref_comp]:
            raise ValueError('insufficient valid data in reference trace')
        else:
            pass
        for _k in thresh_dict.keys():
            if _k != ref_comp:
                # Create copy of reference trace
                trC = ref_tr.copy()
                # Relabel component
                trC.set_comp(_k)
                # Zero-out fold on copies
                trC.to_zero(method='fold')
                # Update zero-fold copies as new "other" component(s)
                self.traces.update({_k: trC})


    # @_add_processing_info    
    def _apply_clone_other(self, thresh_dict):
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
        ref_comp = self.stats.primary_component
        # Run through each component and see if it passes thresholds
        pass_dict = {}
        for _k, _tr in self.traces.items():
            pass_dict.update({_k: _tr.fvalid >= thresh_dict[_k]})

        # If the reference component is present but fails to pass, kick error
        if ref_comp in pass_dict.keys():
            if not pass_dict[ref_comp]:
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
            elif any(_v for _k, _v in pass_dict.items() if _k != ref_comp):
                # Iterate across components in 
                for _k, _v in pass_dict.items():
                    # If not the reference component and did not pass
                    if _k != ref_comp and not _v:
                        # Grab component code that will be cloned over
                        cc = _k
                    # If not the reference component and did pass
                    if _k != ref_comp and _v:
                        # Create a clone of the passing "other" component
                        trC = _v.copy()
                # Zero out the fold of the cloned component and overwrite it's component code
                trC.to_zero(method='fold').set_comp(cc)
                # Write cloned, relabeled "other" trace to the failing trace's position
                self.traces.update({cc: trC})

            # If only the reference trace passed, run _apply_clone_ref() method instead
            else:
                self._apply_clone_ref(thresh_dict)

        # If ref and one "other" component are present
        elif ref_comp in pass_dict.keys():
            # ..and they both pass checks
            if all(pass_dict.items()):
                # Iterate across all expected components
                for _c in thresh_dict.keys():
                    # to find the missing component 
                    # (case where all components are present & passing is handled above) 
                    # catch the missing component code
                    if _c not in pass_dict.keys():
                        cc = _c
                    # and use the present "other" as a clone template
                    elif _c != ref_comp:
                        trC = self[_c].copy()
                # Stitch results from the iteration loop together
                trC.set_comp(cc)
                # Zero out fold
                trC.to_zero(method='fold')
                # And update traces with clone
                self.traces.update({cc: trC})
            # If the single "other" trace does not pass, use _apply_clone_ref method
            else:
                self._apply_clone_ref(thresh_dict)
        # If only the reference component is present & passing
        else:
            self._apply_clone_ref(thresh_dict)
    
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
            line.append(not np.ma.is_masked(_tr.data))
            holder.append(line)
        
        bool_array = np.array(holder)
        if mode == 'summary':
            status=bool_array.all()
        elif mode == 'attribute':
            status = dict(zip(list(ref.keys()) + ['mask_free'], bool_array.all(axis=0)))
        elif mode == 'trace':
            status = dict(zip(self.traces.keys(), bool_array.all(axis=1)))
        elif mode == 'full':
            status = pd.DataFrame(bool_array,
                                  columns=list(ref.keys()) + ['mask_free'],
                                  index=self.traces.keys()).T
        return status

    def treat_gaps(self,
                   filterkw={'type': 'bandpass', 'freqmin': 1, 'freqmax': 45},
                   detrendkw={'type': 'linear'},
                   resample_method='resample',
                   resamplekw={},
                   taperkw={},
                   mergekw={},
                   trimkw={'pad': True, 'fill_value':0}):
        # Get reference values from header
        ref = {}
        for _k in ['reference_starttime','reference_npts','reference_sampling_rate']:
            ref.update({'_'.join(_k.split("_")[1:]): self.stats[_k]})
        resamplekw.update({'sampling_rate':ref['sampling_rate']})
        trimkw.update({'starttime': ref['starttime'],
                        'endtime': ref['starttime'] + (ref['npts'] - 1)/ref['sampling_rate']})
        for tr in self:
            tr.treat_gaps(
                filterkw = filterkw,
                detrendkw = detrendkw,
                resample_method=resample_method,
                resamplekw=resamplekw,
                taperkw=taperkw,
                mergekw=mergekw,
                trimkw=trimkw)
            if tr.stats.sampling_rate != ref['sampling_rate']:
                breakpoint()
        return self
    
    def sync_to_reference(self, fill_value=0., **kwargs):
        starttime = self.stats.reference_starttime
        if starttime is None:
            raise ValueError('reference_starttime must be specified in this ComponentStream\'s `stats`')
        npts = self.stats.reference_npts
        if npts is None:
            raise ValueError('reference_npts must be specified in this ComponentStream\'s `stats`')
        sampling_rate = self.stats.reference_sampling_rate
        if sampling_rate is None:
            raise ValueError('reference_sampling_rate must be specified in this ComponentStream\'s `stats`')

        endtime = starttime + (npts-1)/sampling_rate

        # If any checks against windowing references fail, proceed with interpolation/padding
        if not self.check_windowing_status(mode='summary'):
            # df_full = self.check_windowing_status(mode='full')
            # breakpoint()
            for _tr in self:
                _tr.sync_to_window(starttime=starttime, endtime=endtime, fill_value=fill_value, **kwargs)

        # Extra sanity check
        if not self.check_windowing_status(mode='summary'):
            breakpoint()

        return self


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
        if any(_tr is None for _tr in self):
            breakpoint()
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
        elif not all(_k in model.component_order for _k in self.traces.keys()):
            status = False
        # Check that sampling rate is sync'd
        elif not all(_tr.stats.sampling_rate == self.stats.reference_sampling_rate for _tr in self):
            status = False
        # Passing (or really failing) all of the above
        else:
            status = True
        return status
    
    def to_npy_tensor(self, model):
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
        return npy_array
        # tensor = torch.Tensor(npy_array)
        # return tensor

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
        

    ## I/O ROUTINES ##
