import copy
import numpy as np
import pandas as pd
import seisbench.models as sbm
from obspy import Trace, Stream, UTCDateTime
from wyrm.core.stream.dictstream import DictStream, DictStreamStats
from wyrm.core.trace.mltrace import MLTrace

###############################################################################
# WindowStreamStats Class Definition ##########################################
###############################################################################

class WindowStreamStats(DictStreamStats):
    # NTS: Deepcopy is necessary to not overwrite _types and defaults for parent class
    _types = copy.deepcopy(DictStreamStats._types)
    _types.update({'ref_component': str,
                   'aliases': dict,
                   'reference_starttime': (UTCDateTime, type(None)),
                   'reference_npts': (int, type(None)),
                   'reference_sampling_rate': (float, type(None))})
    defaults = copy.deepcopy(DictStreamStats.defaults)
    defaults.update({'ref_component': 'Z',
                     'aliases': {'Z': 'Z3',
                                 'N': 'N1',
                                 'E': 'E2'},
                     'reference_starttime': None,
                     'reference_sampling_rate': None,
                     'reference_npts': None})
    
    def __init__(self, header={}):
        """Create a WindowStreamStats object

        :param header: attribute: value pairs to assign in the WindowStreamStats, defaults to {}
        :type header: dict, optional
        """        
        # Initialize super + updates to class attributes
        super(WindowStreamStats, self).__init__()
        # THEN update self with header inputs
        self.update(header)

    def __str__(self):
        prioritized_keys = ['ref_component',
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
        
class WindowStream(DictStream):
    """A child-class of DictStream that only uses trace component codes as keys and
    is postured towards processing a collection of windowed traces from a single
    seismometer. It provides additional class methods extending from DictStream
    that facilitate windowed trace data pre-processing in advance of ML prediction
    using SeisBench WaveformModel type model architectures.
    """
    def __init__(
            self,
            traces,
            ref_component='Z',
            header={},
            **options):
        """Initialize a wyrm.core.DictStream.WindowStream object

        :param traces: ObsPy Trace-like object(s)
        :type traces: obspy.core.trace.Trace or list/obspy.core.stream.Stream thereof
        :param ref_component: reference component code for this window stream, defaults to 'Z'
        :type ref_component: str, optional
        :param header: inputs ot pass to the WindowStreamStats.__init__, defaults to {}
        :type header: dict, optional
        :param **options: collector for key-word arguments passed to WindowStream.__add__ 
                for merging entries with matching component codes
        :type **options: kwargs
        """
        # Initialize & inherit from DictStream
        super().__init__()
        # Initialize Stream Header
        self.stats = WindowStreamStats(header=header)
        if isinstance(ref_component, str):
            if ref_component in self.stats.aliases.keys():
                self.stats.ref_component = ref_component
            else:
                raise ValueError(f'ref_component must be a key value in WindowStream.stats.aliases')
        else:
            raise TypeError('ref_component must be type str')
        
        if isinstance(traces, Trace):
            traces = [traces]
        elif isinstance(traces, (Stream, list, tuple)):
            if not all(isinstance(tr, Trace) for tr in traces):
                raise TypeError('all input traces must be type MLTrace')
        else:
            raise TypeError("input 'traces' must be a single MLTrace or iterable set of MLTrace objects")
        # Add traces using the WindowStream __add__ method that converts non MLTrace objects into MLTrace objects
        # if self.validate_trace_ids(self, other=traces)
        self.extend(traces, **options)
        self.stats.common_id = self.get_common_id()
        # if self.ref['component'] in self.traces.keys():
        #     self.stats.reference_id = self.traces[self.ref['component']].id

    def extend(self, traces, **options):
        """Extend (add) more traces to this WindowStream

        :param traces: set of traces to add to this WindowStream, keying on their component codes
        :type traces: obspy.core.trace.Trace-like

        NOTE: Any true obspy.core.trace.Trace objects are converted into wyrm.core.trace.mltrace.MLTrace
             objects before extending the WindowStream
        """
        # If extending with a single trace object
        if isinstance(traces, Trace):
            traces = [traces]

        # Iterate across traces
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


    def __repr__(self, extended=False):
        """
        Provide a user-friendly string representation of the contents and key parameters of this
        WindowStream object. 

        :: INPUTS ::
        :param extended: option to show an extended form of the WindowStream should 
                         there be a large number of unique component codes (an uncommon use case)
        :type extend: bool, optional

        :return rstr: representative string
        :rtype rstr: str
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

    

    ###############################################################################
    # FILL RULE METHODS ###########################################################
    ###############################################################################
    def apply_fill_rule(self, rule='zeros', ref_thresh=0.9, other_thresh=0.8):
        """Summative class method for assessing if channels have enough data,
        and applying the specified channel fill `rule`

        The thresh(olds) values in this method are compared against a given
        wyrm.core.trace.mltrace.MLTrace object's .get_fvalid_subset() output
            ~also see wyrm.core.trace.mltrace.MLTrace.get_fvalid_subset()
        
        :param rule: channel fill rule to apply to non-reference channels that are
                    missing or fail to meet the `other_thresh` requirement, defaults to 'zeros'
                    Supported Values
                        'zeros' - fill with 0-valued traces
                            ~also see wyrm.core.stream.windowstream.WindowStream._apply_zeros()
                        'clone_ref' - clone the primary trace if any secondary traces are missing
                            ~also see wyrm.core.stream.windowstream.WindowStream._apply_clone_ref()
                        'clone_other' - if 1 `other` trace is missing, clone with the present one
                                        if both `other` traces are missing, clone the `ref` trace
                            ~also see wyrm.core.stream.windowstream.WindowStream._apply_clone_other()
        :type rule: str, optional
        :param ref_thresh: fractional completeness threshold for the reference trace,
                        the `ref` trace must pass this threshold to be considered sufficiently complete,
                        defaults to 0.9
                        See rule-specific methods for failure behavior
        :type ref_thresh: float, optional
        :param other_thresh: fractional completeness threshold for the other traces, 
                        traces are considered missing if they fail to meet/exceed this threshold,
                        defaults to 0.8
        :type other_thresh: float, optional
        :raises ValueError: Raised if `rule` is not a supported value
        """        
        thresh_dict = {}
        for _k in self.stats.aliases.keys():
            if _k == self.stats.ref_component:
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

    def _apply_zeros(self, thresh_dict): 
        """
        Apply the channel filling rule "zero" (e.g., Retailleau et al., 2022)
        where both "other" (horzontal) components are set as zero-valued traces
        if one or both are missing/overly gappy.

        0-valued traces are assigned fold values of 0 to reflect the absence of
        added information.

        :param thresh_dict: directory with keys matching keys in 
                        self.traces.keys() (i.e., alised component characters)
                        and values \in [0, 1] representing fractional completeness
                        thresholds below which the associated component is rejected
        :type thresh_dict: dict
        :raises ValueError: raised if the `ref` component has insufficient data
        """
        ref_comp = self.stats.ref_component
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
    def _apply_clone_ref(self, thresh_dict):
        """
        Apply the channel filling rule "clone reference" (e.g., Ni et al., 2023)
        where the reference channel (vertical component) is cloned onto both
        horizontal components if one or both horizontal (other) component data
        are missing or are sufficiently gappy. 
        
        Cloned traces are assigned fold values of 0 to reflect the absence of
        additional information contributed by this trace.

        :: INPUTS ::
        :param thresh_dict: dictionary with keys matching keys in 
                        self.traces.keys() (i.e., alised component characters)
                        and values \in [0, 1] representing fractional completeness
                        thresholds below which the associated component is rejected
        :type thresh_dict: dict
        :raise ValueError: If ref trace has insufficient data
        """
        # Get reference trace
        ref_comp = self.stats.ref_component
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
        :param thresh_dict: dictionary with keys matching keys in 
                        self.traces.keys() (i.e., alised component characters)
                        and values \in [0, 1] representing fractional completeness
                        thresholds below which the associated component is rejected
        :type thresh_dict: dict

        :raise
        """
        ref_comp = self.stats.ref_component
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
            raise KeyError("reference component is not in this WindowStream's keys")
        
        # If all expected components are present
        if pass_dict.keys() == thresh_dict.keys():
            # If all components pass thresholds
            if all(pass_dict.values()):
                # Do nothing
                pass

            # If at least one "other" component passed checks
            elif any(_v if _k != ref_comp else False for _k, _v in pass_dict.items()):
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
        elif ref_comp in pass_dict.keys() and len(pass_dict) > 1:
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
        Check if the data timing and sampling in this WindowStream are synchronized
        with the reference_* [starttime, sampling_rate, npts] attributes in its Stats object
        or those specified as arguments in this check_sync() call. Options are provided for
        different slices of the boolean representation of trace-attribute-reference sync'-ing

        This method also checks if data are masked, using the truth of np.ma.is_masked(tr.data)
        as the output

        :: INPUTS ::
        :param reference_starttime: if None - use self.stats.reference_starttime
                                    if UTCDateTime - use this value for sync check on starttime
        :type reference_starttime: None or obspy.core.utcdatetime.UTCDateTime
        :param reference_sampling_rate: None - use self.stats.reference_sampling_rate
                                    float - use this value for sync check on sampling_rate
        :type reference_sampling_rate: None or float
        :param reference_npts: None - use self.stats.reference_npts
                                    int - use this value for sync check on npts
        :type reference_npts: None or int
        :param mode: output mode
                        'summary' - return the output of bool_array.all()
                        'trace' - return a dictionary of all() outputs of elements 
                                subset by trace
                        'attribute' - return a dictionary of all() outputs of elements
                                subset by attribute
                        'full' - return a pandas DataFrame with each element, labeled
                                with trace component codes (columns) and 
                                attribute names (index)
        :type mode: str
        :return status: [bool] - for mode='summary'
                        [dict] - for mode='trace' or 'attribute'
                        [DataFrame] - for mode='full'
        :rtype status: bool, dict, or pandas.core.dataframe.DataFrame
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
        """Execute a wyrm.core.trace.mltrace.MLTrace.treat_gaps() method on each trace
        in this WindowStream using common kwargs (see below) and reference sampling data
        in the WindowStream.stats

        filter -> detrend* -> resample* -> taper* -> merge* -> trim

        For expanded explanation of key word arguments, see wyrm.core.trace.mltrace.MLTrace.treat_gaps

        :param filterkw: kwargs to pass to MLTrace.filter(), defaults to {'type': 'bandpass', 'freqmin': 1, 'freqmax': 45}
        :type filterkw: dict, optional
        :param detrendkw: kwargs to pass to MLTrace.detrend(), defaults to {'type': 'linear'}
        :type detrendkw: dict, optional
        :param resample_method: resampling method to use, defaults to 'resample'
                supported values - 'resample','interpolate','decimate'
        :type resample_method: str, optional
        :param resamplekw: kwargs to pass to specified resample_method, defaults to {}
        :type resamplekw: dict, optional
                            obspy.core.trace.Trace.interpolate
                            obspy.core.trace.Trace.decimate
        :param taperkw: kwargs to pass to MLTrace.taper(), defaults to {}
        :type taperkw: dict, optional
        :param mergekw: kwargs to pass to MLTrace.merge, defaults to {}
        :type mergekw: dict, optional
        :param trimkw: kwargs to pass to MLTrace.trim, defaults to {'pad': True, 'fill_value':0}
        :type trimkw: dict, optional
        """        
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

    
    def sync_to_reference(self, fill_value=0., **kwargs):
        """Use a combination of trim and interpolate functions to synchronize
        the sampling of traces contained in this WindowWyrm

        Wraps the wyrm.core.trace.mltrace.MLTrace.sync_to_window() method

        :param fill_value: fill value for, defaults to 0.
        :type fill_value: _type_, optional
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :return: _description_
        :rtype: _type_
        """        
        starttime = self.stats.reference_starttime
        if starttime is None:
            raise ValueError('reference_starttime must be specified in this WindowStream\'s `stats`')
        npts = self.stats.reference_npts
        if npts is None:
            raise ValueError('reference_npts must be specified in this WindowStream\'s `stats`')
        sampling_rate = self.stats.reference_sampling_rate
        if sampling_rate is None:
            raise ValueError('reference_sampling_rate must be specified in this WindowStream\'s `stats`')

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
    # WindowStream to Tensor Methods ###########################################
    ###############################################################################
            
    def ready_to_burn(self, model):
        """
        Assess if the data contents of this WindowStream are ready
        to convert into a torch.Tensor given a particular seisbench model

        NOTE: This inspects that the dimensionality, timing, sampling, completeness,
             and component aliasing of the contents of this WindowStream are
             compliant with reference values in the metadata and in the input `model`'s
             metadata
              
             It does not check if the data have been processed: filtering, tapering, 
                normalization, etc. 

        :: INPUT ::
        :param model: [seisbench.models.WaveformModel] initialized model
                        object that prediction will be run on 
                        NOTE: This is really a child-class of sbm.WaveformModel
        :: OUTPUT ::
        :return status: [bool] - is this WindowStream ready for conversion
                                using WindowStream.to_torch(model)?
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
        Convert the data contents of this WindowStream into a numpy array that
        conforms to the component ordering required by a seisbench WaveformModel
        object
        :: INPUT ::
        :param model: []
        """
        if not self.ready_to_burn(model):
            raise ValueError('This WindowStream is not ready for conversion to a torch.Tensor')
        
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
            raise ValueError('not all traces in this WindowStream have matching npts')
        

    ## I/O ROUTINES ##
