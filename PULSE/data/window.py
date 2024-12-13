"""
:module: PULSE.data.window
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

.. rubric:: Purpose
This module provides the class definitions for :class:`~PULSE.data.window.Window` and :class:`~PULSE.data.window.WindowStats`
that are child classes of :class:`~PULSE.data.dictstream.DictStream` and :class:`~PULSE.data.dictstream.DictStreamStats`, respectively.

The Window class keys :class:`~PULSE.data.mltrace.MLTrace`-type objects by their component code, rather than full **id** attribute
and provides additional class methods for pre-processing one or more MLTrace objects into a data tensor ready for input to a machine
learning model (i.e., those derived from :class:`~seisbench.models.WaveformModel`.

NOTE: This class implicitly assumes data are no more than 3 component. 
If we want to work with something like OBS or DAS data there are some
places that will need modification to include more channels and/or multiple
primary channels (e.g., Z and pressure on an OBS).
"""
import logging, os, sys, warnings, typing
import numpy as np
import pandas as pd
from torch import Tensor
import seisbench.models as sbm
from obspy import Trace, Stream
from PULSE.data.dictstream import DictStream
from PULSE.data.foldtrace import FoldTrace
from PULSE.util.header import WindowStats
    
###############################################################################
# Window Class Definition ###########################################
###############################################################################

Logger = logging.getLogger(__name__)

class Window(DictStream):
    """A child-class of :class:`~PULSE.data.dictstream.DictStream`
    that uses trace component codes as keys and is intended to
    faciliate processing of a collection of windowed traces from
    a single seismometer. 

    :param traces: list of FoldTrace-like objects to ingest at
        initialization, defaults to [].
    :type traces: :class:`~.FoldTrace` or iterable thereof, optional
    :param header: Non-default arguments to pass to :class:`~.WindowStats`,
        defaults to {}.
    :type header: dict, optional
    **options: collector for key-word arguments passed to :meth:`~.Window.__add__` in determining
        how entries in `traces` with matching component codes are merged.
    """
    def __init__(self, traces=[], header=None, primary_component='Z', **options):
        """
        Initialize a PULSE.data.window.Window object

        :param traces: collection of :class:`~.FoldTrace`-like to insert at initialization,
            defaults to [].
        :type traces: list-like, optional
        :param header: non-default values to pass to the header of this Window object, 
            defaults to 'auto'.
            If 'auto' is passed and **traces** contains :class:`~.FoldTrace` objects, the
            header will be populated with target values based on the first "Z" component
            foldtrace encounterd in traces, if one is present.
        :type header: dict, optional
        
        **options: collector for key-word arguments passed to :meth:`~PULSE.data.window.Window.__add__` in determining
            how entries in `traces` with matching component codes are merged.
            also see :meth:`~PULSE.data.dictstream.DictStream.__add__`
        """
        # Initialize & inherit from DictStream as an empty dictstream using 'comp' key attributes
        super().__init__()
        # Overwrite supported keys & set key-attr
        # self.supported_keys = ['comp']
        self.key_attr = 'component'

        # Compatability Check for traces
        if all(isinstance(ft, Trace) for ft in traces):
            pass
        else:
            raise TypeError('not all elements in "traces" is type obspy.Trace')

        # Bug-patch for bleed issue in testing FIXME
        if header is None:
            header = {}
        elif isinstance(header, dict):
            pass
        elif isinstance(header, WindowStats):
            hdr = {}
            for _k, _v in header.items():
                if _k not in header._readonly:
                    hdr.update({_k: _v})
            header = hdr
        else:
            raise TypeError

        # Compatability Check for primary_component
        if isinstance(primary_component, str):
            if len(primary_component) != 1:
                raise ValueError('str-type primary component must be a single-character string')
        elif isinstance(primary_component, int):
            if 10 > int(primary_component) >= 0:
                primary_component = str(primary_component)
            else:
                raise ValueError('int-type primary_component must be in [0, 9]')
        else:
            raise TypeError('primary_component must be type int or str')

        # AUTO HEADER GENERATION
        if 'primary_id' not in header.keys():                
            if len(traces) > 0:
                if 'secondary_components' not in header.keys():
                    c2 = ''
                else:
                    c2 = header['secondary_components']
                for tr in traces:
                    if tr.stats.component == primary_component:
                        header.update({'primary_id': tr.id})
                        for fld in {'starttime','npts','sampling_rate'}:
                            if f'target_{fld}' not in header.keys():
                                header.update({f'target_{fld}': tr.stats[fld]})
                    elif tr.stats.component not in c2:
                        if 'secondary_components' not in header.keys():
                            c2 += tr.stats.component
                        else:
                            raise ValueError(f'{tr.id} is not one of the specified secondary components')
                    else:
                        raise ValueError(f'{tr.id} is a repeat component')
                header.update({'secondary_components': c2})

        # Populate stats
        self.stats = WindowStats(header=header)
        
        # Add traces - safety checks for repeat components handled in extend()
        if len(traces) > 0:

            self.extend(traces, **options)
            # Final check - validate contents if traces are present        
            self._validate()
    
    
    ### DUNDER METHODS ###
    def __repr__(self) -> str:
        """Provide a representative string for the metadata 
        and traces in this :class:`~.Window`. Primary component
        trace is marked with a `(P)`

        :returns: **rstr** (*str*) -- representative string
        """        
        rstr = f'{self.stats.__str__()}\n'
        rstr += f'----------\n'
        rstr += f'{len(self)} FoldTraces in Window\n'
        if len(self) > 0:
            rstr += f'{self.primary.stats.component}: {self.primary} (P)\n'
        for _k, _v in self.traces.items():
            if _k != self.primary.stats.component:
                rstr += f'{_k}: {_v}\n'
        return rstr

    def __eq__(self, other):
        """Allow rich implementation of the == operator
        """        
        if not isinstance(other, Window):
            return False
        if self.traces != other.traces:
            return False
        if self.stats != other.stats:
            return False
        return True

    ### PRIVATE METHODS, PROPERTY ASSIGNMENTS, SUBROUTINES ###
    def _validate(self) -> None:
        """Validate essential elements of this :class:`~.Window` object

        :raises ValueError: If the primary component trace is not present
        :raises ValueError: If not all traces in this Window are type :class:`~.FoldTrace`
        :raises ValueError: If not all traces share the same instrument code
        """        
        # Make sure primary component is present
        if self.stats.get_primary_component() not in self.traces.keys():
            raise ValueError('Primary component is not present')
        # Make sure all traces are type FoldTrace
        if not all(isinstance(ft, FoldTrace) for ft in self):
            raise ValueError('Not all traces are type PULSE.data.foldtrace.FoldTrace')
        # Make sure all traces are from the same instrument
        if not all(self.primary.id_keys['inst'] == ft.id_keys['inst'] for ft in self):
            raise ValueError('Not all traces are from the same instrument')
    
    def _get_primary(self) -> typing.Union[FoldTrace, None]:
        """Return a view of the primary component trace if present
        otherwise, return None

        :returns: **primary** (*FoldTrace* or *None*)
        """        
        pid = self.stats.primary_id
        if isinstance(pid, str):
            pcomp = pid[-1]
            if pcomp in self.keys:
                return self.traces[pcomp]
            else:
                return None
        else:
            return None
        
    primary = property(_get_primary)

    def _get_order(self) -> str:
        """Get the preferred component order specified
        as the primary_id component + the secondary_component codes

        :returns: **order** (*str*) -- string containing ordered component codes
        """        
        pid = self.stats.get_primary_component()
        sids = self.stats.secondary_components
        return pid + sids

    order = property(_get_order)

    def _check_targets(self, comp: str) -> set:
        """Check a given FoldTrace in this :class:`~.Window` object
        meet the target values in **stats**
        :param comp: component code to check
        :type comp: str
        :returns: **result** (*set*) names of attributes that do
            not meet target values
        """        
        ft = self[comp]
        failing = []
        if ft.stats.starttime != self.stats.target_starttime:
            failing.append('starttime')
        if ft.stats.sampling_rate != self.stats.target_sampling_rate:
            failing.append('sampling_rate')
        if ft.stats.npts != self.stats.target_npts:
            failing.append('npts')
        return set(failing)
    
    def _check_starttime_alignment(self, comp: str) -> bool:
        """Check if the specified component's time sampling
        aligns with the target_starttime

        :param comp: component code to check
        :type comp: str
        :return: **result** (*bool*) -- does trace starttime align
            with target time index sampling?
        """        
        ft = self[comp]
        dt = self.stats.target_starttime - ft.stats.starttime
        result = dt*self.stats.target_sampling_rate == \
                 int(dt*self.stats.target_sampling_rate)
        return result
    
    def _get_nearest_starttime(self, comp, roundup=True):
        """Calculate the nearest timestamp to a specified component's
        starttime in the target time sampling index (i.e., the vector
        of times specified by target_(starttime/npts/sampling_rate)).
        
        This method defaults to rounding up the nearest starttime in
        the case of un-aligned sampling to provide a starttime that 
        falls within the time domain of the specified component's data,
        which is required for applying interpolation without introducing
        padding values.

        .. rubric:: Example
        >>> st = read()
        >>> win = Window(st.copy())
        >>> win['Z'].stats.starttime
        2023-10-09T02:20:17.870000Z
        >>> win.stats.target_starttime += 1.0001
        2023-10-09T02:20:18.870100Z
        >>> win._get_nearest_starttime('Z')
        2023-10-09T02:20:17.870100Z

        :param comp: component code to check
        :type comp: str
        :param roundup: should fractional samples be rounded up
            to the nearest integer sample or simply rounded, 
            defaults to `True`
        :type roundup: bool, optional
        :returns: **nearest** (*obspy.UTCDateTime*) -- nearest
            time in the target sampling index to
        """        
        # Get the specified component
        ft = self[comp]
        # Get the time delta: target - current
        dt = self.stats.target_starttime - ft.stats.starttime
        # Convert to samples using the target sampling rate
        # npts = target S/R samples from ft start to target start
        npts = dt*self.stats.target_sampling_rate
        # Round up so starttime is inside domain of **other**
        if roundup:
            npts = np.floor(npts)
        # Or just round to nearest if turned off
        else:
            npts = np.round(npts)
        # Calcluate nearest starttime in the target sampling index
        # to the starttime of the original trace
        nearest = self.stats.target_starttime - npts/self.stats.target_sampling_rate
        return nearest 

    def _check_fvalid(self, comp, tolerance=1e-3, fthresh=0):
        """Check if the fraction of data in a specified component
        within the bounds of the target starttime and endtime
        passes the associated threshold:
        primary >= pthresh
        secondary >= sthresh 

        :param comp: component code to check
        :type comp: str
        :param tolerance: tolerance for small rounding discrepancies
            when assessing a components fvalid agains the associated
            threshold, defaults to 1e-3.
            Must be a value in [0, 0.1)
        :type tolerance: float, optional
        :param fthresh: fold threshold level defining
            valid/invalid data, defaults to 0.
            also see :meth:`~.FoldTrace.get_valid_fraction`
        :type fthresh: float, optional
        :raises ValueError: _description_
        :returns: **result** (*bool*) -- does the specified component
            pass data validity requirements?
        """        
        # Compatability check for tolerance
        if 0 <= tolerance < 0.1:
            pass
        else:
            raise ValueError('tolerance must be in [0.0, 0.1)')
        
        # Get component
        ft = self[comp]
        # Get the fraction of valid data for the target range
        fvalid = ft.get_valid_fraction(
            starttime = self.stats.target_starttime,
            endtime = self.stats.target_endtime,
            threshold=fthresh)
        # Add tolerance
        fvalid += tolerance
        # If assessing the primary component
        if ft.id == self.stats.primary_id:
            return fvalid >= self.stats.pthresh
        # If assessing secondary components
        else:
            return fvalid >= self.stats.sthresh
     
    ######################
    ### PUBLIC METHODS ###
    ######################
    def copy(self):
        """Create a deepcopy of the traces
        and metadata contained in this :class:`~.Window` object

        :return: _description_
        :rtype: _type_
        """        
        window = super().copy()
        window.stats = self.stats.copy()
        # window = self.__class__(traces=[ft.copy() for ft in self],
        #                         header=self.stats.copy())
        return window


    def preprocess_component(
            self,
            comp,
            align={},
            filter={'type': 'bandpass', 'freqmin': 1., 'freqmax': 45.},
            detrend={'type': 'linear'},
            resample={'method':'resample'},
            taper={'max_percentage': None, 'max_length': 0.06},
            trim={'fill_value': 0.}):
        """Apply the following pre-processing pipeline on a specified component
        in this :class:`~.Window` object

        Workflow
        --------
        split component trace into contiguous elements
        using :meth:`~.FoldTrace.split` with **ascopy**=`False`
        for each element:
            align - align the temporal sampling with the target sampling domain
                uses :meth:`~.FoldTrace.align_starttime`
                REQUIRED
            filter - apply a filter to the component trace element
                uses :meth:`~.FoldTrace.filter`
                OPTIONAL (turned of with None input)
            detrend - detrend the component trace element
                uses :meth:`~.FoldTrace.detrend`
                OPTIONAL (turned of with None input)
            demean - remove the mean of the component trace element
                uses :meth:`~.FoldTrace.detrend` with **type** = `demean`
            resample - resample the component trace element to the target sampling_rate
                uses one of the following methods (passed as {'method':'<name>'})
                'resample': uses :meth:`~.FoldTrace.resample`
                'interpolate': uses :meth:`~.FoldTrace.interpolate`
                'decimate': uses :meth:`~.FoldTrace.decimate`
                REQUIRED
            taper - apply a taper to the component trace element on both ends
                uses :meth:`~.FoldTrace.taper`
                OPTIONAL (turned of with None input)
        
        trim - trim / pad the component trace and fill masked values
            uses :meth:`~.FoldTrace.trim`
            hard sets **pad**=`True` and **apply_fill**=`True`
            REQUIRED

        Parameters
        ----------
        :param comp: component code for trace to process
        :type comp: str
        :param align: optional arguments to pass to :meth:`~.FoldTrace.align_starttime`,
            defaults to {}
        :type align: dict, optional
        :param filter: arguments to pass to :meth:`~.FoldTrace.filter`,
            defaults to {'type': 'bandpass', 'freqmin': 1., 'freqmax': 45.}
            Input of None turns off filtering
        :type filter: dict or NoneType, optional
        :param detrend: arguments to pass to :meth:`~.FoldTrace.detrend`,
            defaults to {'type': 'linear'}
            Input of None turns off filtering
        :type detrend: dict or NoneType, optional
        :param resample: arguments to pass to pass to the specified resampling
            method (see above), defaults to {'method':'resample'}
            Key 'method' is required
        :type resample: dict, optional
        :param taper: arguments to pass to :meth:`~.FoldTrace.taper`,
            defaults to {'max_percentage': None, 'max_length': 0.06}
            Input of None turns off tapering
        :type taper: dict or NoneType, optional
        :param trim: arguments to pass to :meth:`~.FoldTrace.trim`,
            defaults to {'fill_value': 0.}
            'fill_value' is a required input
        :type trim: dict, optional
        """        
        if comp not in self.keys:
            raise KeyError

        # Alignment (assessment) is required
        if isinstance(align, type(None)):
            align = {}
        if not isinstance(align, dict):
            raise TypeError('align must be type dict')
        else:
            align.update({'starttime': self.stats.target_starttime,
                          'sampling_rate': self.stats.target_sampling_rate})

        # Filtering is optional, but suggested
        if not isinstance(filter, (dict, type(None))):
            raise TypeError('filter must be type dict or NoneType')
        elif isinstance(filter, dict):
            if 'type' not in filter.keys():
                raise AttributeError('filter required kwarg "type" not present')
            
        # Detrending is optional, but suggested
        if not isinstance(detrend, (dict, type(None))):
            raise TypeError('detrend must be type dict or NoneType')
        elif isinstance(detrend, dict):
            if 'type' not in detrend.keys():
                raise AttributeError('detrend required kwarg "type" not present')
        
        # Resampling (assessment) is required
        if not isinstance(resample, dict):
            raise TypeError('resample must be type dict')
        elif 'method' not in resample.keys():
            raise AttributeError('resample required kwarg "method" not present')
        
        # Tapering is optional, but suggested
        if not isinstance(taper, (dict, type(None))):
            raise TypeError('taper must be type dict or NoneType')
        
        # Trimming (assessment) is required
        if not isinstance(trim, dict):
            raise TypeError('trim must be type dict')
        elif 'fill_value' not in trim.keys():
            raise AttributeError('trim required kwarg "fill_value" not present')
        elif trim['fill_value'] is None:
            raise ValueError('fill_value must be a float-like value')
        


        # Create copies of contiguous elements of the component
        st = self[comp].split(ascopy=True)
        # Iterate across elements
        for _e, _ft in enumerate(st):

            # Detrend
            if detrend is not None:
                _ft.detrend(**detrend)
            # # Demean
            # _ft.detrend(type='demean')
            # Filter
            if filter is not None:
                _ft.filter(**filter)
            # Align starttime
            _ft.align_starttime(self.stats.target_starttime,
                                self.stats.target_sampling_rate)
            # Resample
            rmethod = resample.pop('method')
            if _ft.stats.sampling_rate != self.stats.target_sampling_rate:
                resample.update({'sampling_rate': self.stats.target_sampling_rate})
                getattr(_ft,rmethod)(**resample)
            # FIXME: In testing multiple calls of this method in a test result decay
            resample.update({'method': rmethod})
            # Taper
            if taper is not None:
                _ft.taper(**taper)
            # Reassemble processed trace elements
            if _e == 0:
                _ft_new = _ft.copy()
            else:
                _ft_new += _ft
        # Assign update fold trace
        self[comp] = _ft_new

        # Trim/Pad/Fill
        if self[comp].stats.starttime != self.stats.target_starttime:
            trim.update({'starttime': self.stats.target_starttime})
        if self[comp].stats.endtime != self.stats.target_endtime:
            trim.update({'endtime': self.stats.target_endtime})
        # Require 'pad' to be True & apply_fill to be True
        trim.update({'pad': True, 'apply_fill': True})
        self[comp].trim(**trim)
        # TODO: Test if inplace modifications will cause chaos with processing
    
    
    ### Window Level Methods ###

    def fill_missing_traces(self, rule='primary', **options):
        """Fill missing/ill-informed traces in this :class:`~.Window`
        using information from an informative trace.

        "informative" here is defined as a trace having a valid fraction of 
        its data (per :meth:`~.FoldTrace.get_valid_fraction`) in the target
        time range specified in **stats** that meets or exceeds the relevant 
        threshold for that trace's componet designation:
         - **pthresh** for the primary component
         - **sthresh** for secondary components

        Fill rules provided by this method are based on a range of strategies
        used in the literature to analyze data that do not meet the input layer
        expectations of machine learning models: uninterrupted, 3-component data.

        Rules
        -----
        In all cases, cloned :class:`~.FoldTrace` **fold** is set to 0 for all
        samples, reflecting that the clones do not add to information density
        for this window.

        In all cases, the component code of clones are updated to match the
        missing/ill-informed trace

        [0] 'zeros' -- Replace missing/ill-informed traces with a clone of the
            primary component trace with its **data** set to a 0-vector. 
            Follows methods in :cite:`Retailleau2022`.

        [1] 'primary' -- Replace missing/ill-informed traces with copies of the
            primary component trace. Follows methods in :cite:`Ni2023`.

        [2] 'secondary' -- If one secondary component traces is sufficiently
            informative, replace missing/ill-informed secondary traces with
            a copy of the informative secondary component trace. If all
            secondary traces are missing/ill-informed, reverts to behavors
            for rule 'primary'. Follows methods in :cite:`Lara2023`.
        
        This method conducts in-place modifications to the contents of this
        :class:`~.Window` object. If you want to preserve the original contents
        use :meth:`~.Window.copy`

        Parameters
        ----------
        :param rule: rule to use, defaults to 'primary'.
            Also supports integer aliases (the [#] values above)
        :type rule: str or int, optional
        :param options: key-word argument collector passed to 
            :meth:`~.Window._check_fvalid` calls within this method
        
        """
        rules = ['zeros','primary','secondary']
        if isinstance(rule, int):
            if 0 <= rule <= 2:
                rule = rules[rule]
            else:
                raise ValueError('Integer aliased rule must be in [0, 2]')
        elif isinstance(rule, str):
            if rule.lower() in rules:
                rule = rule.lower()
            else:
                raise ValueError('rule "{rule}" not supported')
        else:
            raise ValueError(f'rule "{rule}" not supported')
        
        # Validate to check primary
        self._validate()
        # Get primary component
        pcpt = self.primary.stats.component
        # Get shorthand for secondary components
        scpt = self.stats.secondary_components

        # run valid fraction checks on present traces
        fv_checks = {_k: self._check_fvalid(_k, **options) for _k in self.keys}

        # fill in secondary components that are missing
        for _c in scpt:
            # show missing components it as failing
            if _c not in fv_checks.keys():
                fv_checks.update({_c: False})
        # Make sure the primary component has sufficient valid data
        if not fv_checks[pcpt]:
            raise ValueError(f'insufficient data in primary component {pcpt}')
            
        # If all components are passing, do nothing
        if all(_v for _v in fv_checks.values()):
            return
        # If any secondaries pass
        elif any(fv_checks[_c] for _c in scpt):
            # If using secondary
            if rule == 'secondary':
                # flag the cloneable component
                for _c in scpt:
                    if fv_checks[_c]:
                        donor = _c
                        break
            # Otherwise, flag primary for cloning
            else:
                donor = pcpt
        # If all secondaries fail, use primary as the donor 
        else:
            donor = pcpt

        # Create clone of donor component
        ftd = self[donor].copy()
        # Set clone fold to 0 (adds no information density)
        ftd.fold = ftd.data*0
        # If using `zeros` rule, make data a 0-vector on the clone
        if rule == 'zeros':
            ftd.data = ftd.data*0
        # Iterate across components
        for comp, passing in fv_checks.items():
            # If component failed checks
            if not passing:
                # Make a copy of the clone trace -> replacement trace
                ftr = ftd.copy()
                # Update the replacement trace component code
                ftr.stats.channel = ftr.stats.channel[:-1] + comp
                # Replace/add in the Window
                if comp in self.keys:
                    self[comp] = ftr
                else:
                    self.extend(ftr)

    def preprocess(self, components=None, rule=1, threshold_tolerance=1e-3, fold_threshold=0, **options):
        """Preprocess specified components of this :class:`~.Window` object
        using the workflow described in :meth:`~.Window.preprocess_component`
        and subsequently fill any missing/under-informative components using
        :meth:`~.Window.fill_missing_traces`

        :param components: components to preprocess, defaults to None
            - None processes the primary and any existing secondary components
            specified in this Window object's **stats**
            - An iterable that returns characters (e.g., 'Z12') will
            result in processing Z 1 and 2 (if they are present)
        :type components: NoneType or iterable collection of characters, optional
        :param rule: missing/under-informative fill rule, defaults to 1
            see :meth:`~.Window.fill_missing_traces` for more information
        :type rule: int or str, optional
        :param threshold_tolerance: Tolerance for components to meet their
            valid data fraction threshold, defaults to 1e-3
            threshold in :meth:`~.Window._check_fvalid`
        :type threshold_tolerance: float, optional
        :param fold_threshold: Cutoff for fold values to be considered
            valid samples, defaults to 0
            fthresh in :meth:`~.Window._check_fvalid`
        :type fold_threshold: int, optional
        :param options: key-word argument collector passed to
            :meth:`~.Window.preprocess_component`
        """        
        if components is None:
            components = self.keys
        elif hasattr(components, '__iter__'):
            if all(isinstance(_e, str) for _e in components):
                pass
            else:
                raise TypeError('Not all specified components in "components" are type str')
        else:
            raise AttributeError('components must be an iterable comporising component codes or None')

        for comp in components:
            if comp in self.keys:
                self.preprocess_component(comp, **options)
            else:
                pass
            
        self.fill_missing_traces(rule=rule,
                                 tolerance=threshold_tolerance,
                                 fthresh=fold_threshold)


    def to_npy_tensor(self, components=None):
        """Convert the pre-processed contents of this :class:`~.Window` into a
        numpy array that meets the scaling specifications for a SeisBench
        WaveformModel input. This method holds off on converting the tensor
        into a :class:`~torch.Tensor` to allow 

        :param component_axis: axis indexing components, defaults to 0
            This varies with SeisBench model architecture, i.e., 
            EQTransformer uses component_axis=0
            PhaseNet uses component_axis=1
        :type component_axis: int, optional
        :param components: specified component order, defaults to None
            None results in primary component + secondary components
        :type components: None or str, optional
        :param output_type: 
        :return: _description_
        :rtype: _type_
        """
        # If component_order is None, use content from header
        if components is None:
            components = self.order
        # For all candidate components, make sure they are iterables
        if hasattr(components, '__iter__'): 
            # That call only extant traces
            if all(_k in self.keys for _k in components):
                pass
            # Otherwise raise key error
            else:
                raise ValueError('Not all specified components in "components" are present')
        # If components isn't iterable - attribute error
        else:
            raise AttributeError('components must be an iterable comprising component code characters or None')

        # Form tensor with appropriate dimensions
        shp = (len(components), self.stats.target_npts)
        tensor = np.full(shape=shp, fill_value=0.)

        # Fill in Tensor
        for _e, _c in enumerate(components):
            # Make sure component passes checks
            result = self._check_targets(_c)
            if result != set([]):
                raise AttributeError(f'Component "{_c}" falied the following: {result}')
            # Catch case where data attribute is still a masked array with no masking
            if isinstance(self[_c].data, np.ma.MaskedArray):
                tensor[_e, :] = self[_c].data.data
            else:
                tensor[_e, :] = self[_c].data

        return tensor
    
    def collapse_fold(self, components=None):
        """Sum the fold vectors of all (or specified) components
        in this :class:`~.Window`

        :param components: string of component codes to include
            in the fold collapse, defaults to None
            None -> uses primary component + secondary components
        :type components: bool, optional
        :return: **summed_fold** (*numpy.ndarray*) -- summed fold vector
        """ 
        if components is None:
            components = self.order
        elif hasattr(components, "__iter__"):
            if all(_e in self.keys for _e in components):
                pass
            else:
                raise ValueError('Element(s) of "components" do not correspond to components in this Window')
        else:
            raise AttributeError('components must be an iterable comprising component codes in this Window or None')

        # Initialize new summed_fold vector
        summed_fold = np.full(shape=(self.stats.target_npts,), fill_value=0.)
        # Iterate across desired components and sum fold
        for _c in components:
            if _c in self.keys:
                result = self._check_targets(_c)
                if result != set([]):
                    raise AttributeError(f'Component "{_c}" failed these tests: {result}')
                summed_fold += self[_c].fold
            else:
                continue
        return summed_fold