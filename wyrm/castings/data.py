"""
:module: wyrm.core.data
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module houses wyrm classes that define data-containing objects
    for streaming (sequential) data/prediction packets that traverse a wyrm module. 

    Buffers are defined by their relaxed rules on the synchronicity and size of
    data between buffer objects.

    Windows have specific rules on their temporal bounds and sampling.

    Classes:

        InstrumentWindow - child class of obspy.core.stream.Stream providing
            "window" based methods and component tracking attributes to transform
            trace (potentially gappy or misaligned) data into organized, continuous
            tensors expected by ML models with metadata - e.g., a PredictionWindow object

            [Trace, Trace, Trace] -> {processing} -> PredictionWindow
                                      (ProcWyrm)

            Processing can be handled in a number of ways, in Wyrm this is facilitated
            using one or more ProcWyrm objects (wyrm.core.process.ProcWyrm)

        PredictionWindow - abstraction of the obspy.core.stream.Stream and
            obspy.core.trace.Trace classes that houses a 2-D data array
            that conforms to the input or output scale of SeisBench hosted
            PyTorch models. This is the data class associaetd with for MachineWyrm
            (wyrm.core.process.MachineWyrm)

            PredictionWindow -> Tensor  -> MLModel -> Tensor -> PredictionWindow
                             \_ Metadata --------> Metadata _/

        TraceBuffer - "tbuff" for short
            adapted version of the obspy.realtime.rttrace.RtTrace class
            that enables in-filling of data due to non-sequential packets

        PredictionBuffer - 'pbuff' for short
            adaptation of the obspy.core.stream.Stream and obspy.realtime.rttrace.RtTrace
            classes that buffers 2-D arrays along the 1-axis (time-axis).
        
        BufferTree - "tree" for short
            Provides a 3-tiered dictionary structure terminating in one
            of the above buffer class objects

"""
import os
import fnmatch
import inspect
import torch
import numpy as np
import pandas as pd
import seisbench.models as sbm
import wyrm.util.compatability as wuc
from wyrm.util.semblance import shift_trim
from obspy import Trace, Stream, UTCDateTime, read
from pandas import DataFrame
from copy import deepcopy
from collections import deque

class InstrumentWindow(Stream):

    def __init__(
            self,
            Ztr,
            Ntr=None,
            Etr=None,
            target_starttime=None,
            model_name='EQTransformer',
            component_order='ZNE',
            target_npts=6000,
            target_samprate=100,
            target_blinding=500,
            missing_component_rule='zeros',
    ):
        # Inherit from obspy.core.stream.Stream
        super().__init__()
        # Z trace compat check
        if isinstance(Ztr, Trace):
            self.Z = Ztr.id
            if isinstance(Ztr, TraceBuffer):
                self.traces.append(Ztr.to_trace())
            else:
                self.traces.append(Ztr)
        else:
            raise TypeError('Ztr must be type obspy.core.trace.Trace or wyrm.core.buffer.trace.TraceBuffer')
        # N trace compat check
        if isinstance(Ntr, Trace):
            self.N = Ntr.id
            if isinstance(Ntr, TraceBuffer):
                self.traces.append(Ntr.to_trace())
            else:
                self.traces.append(Ntr)
        elif Ntr is None:
            self.N = None
        else:
            raise TypeError('Ntr must be type obspy.core.trace.Trace or wyrm.core.buffer.trace.TraceBuffer')
        # E trace compat check
        if isinstance(Etr, Trace):
            self.E = Etr.id
            if isinstance(Etr, TraceBuffer):
                self.traces.append(Etr.to_trace())
            else:
                self.traces.append(Etr)
        elif Etr is None:
            self.E = None
        else:
            raise TypeError('Etr must be type obspy.core.trace.Trace or wyrm.core.buffer.trace.TraceBuffer')
        # target starttime compat check
        if target_starttime is None:
            self.target_starttime = self.traces[0].stats.starttime
        elif isinstance(target_starttime, UTCDateTime):
            self.target_starttime = target_starttime
        elif isinstance(target_starttime, float):
            self.target_starttime = UTCDateTime(target_starttime)
        else:
            raise TypeError('target_starttime must be type float, UTCDateTime, or NoneType')
        # target sampling rate compat check
        self.target_samprate = wuc.bounded_floatlike(
            target_samprate,
            name='target_samprate',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # model_name compat check
        if not isinstance(model_name, str):
            raise TypeError('model_name must be type str')
        else:
            self.model_name=model_name
        # target npts compat check
        self.target_npts = wuc.bounded_intlike(
            target_npts,
            name='target_npts',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # target order compat check
        if not isinstance(component_order, str):
            raise TypeError('component_order must be type str')
        if not all((_c.upper() in 'ZNE' for _c in component_order)):
            raise ValueError('not all characters in component_order are supported')
        else:
            self.component_order = ''.join([_c.upper() for _c in component_order])
        # target blinding compat check
        self.target_blinding = wuc.bounded_intlike(
            target_blinding,
            name='target_blinding',
            minimum=0,
            maximum=self.target_npts/2,
            inclusive=True
        )

        # missing component rule compat check
        if not isinstance(missing_component_rule, str):
            raise TypeError('missing_component_rule must be type str')
        elif missing_component_rule.lower() not in ['zeros','clonez','clonehz']:
            raise ValueError('missing_component_rule "{missing_component_rule}" not supported.\nSupported: zeros, clonez, clonehz')
        else:
            self.mcr = missing_component_rule.lower()

        # Create processing documentation list
        self.processing = []
        # Apply missing component rule (mcr)
        self.apply_missing_component_rule()

    # ################### #
    # DATA LOOKUP METHODS #
    # ################### #

    def fetch_with_id(self, nslc_id, ascopy=False):
        """
        Fetch the trace(s) that match the specified N.S.L.C formatted
        id string

        :: INPUT ::
        :param nslc_id: [str] N.S.L.C formatted channel ID string
        :param ascopy: [bool] operate on copies of traces?

        :: OUTPUT ::
        :return traces: [None] - if no traces matches nslc_id
                        [obspy.Traces] - if one trace object matches nslc_id
                        [obspy.Stream] - if multiple trace objects match nslc_id
        """
        traces = []
        if not isinstance(nslc_id, (type(None), str)):
            raise TypeError("id must be a N.S.L.C. string or None")
        elif nslc_id is None:
            return None
        else:
            for _tr in self.traces:
                if _tr.id == nslc_id:
                    if ascopy:
                        traces.append(_tr.copy())
                    else:
                        traces.append(_tr)
        if len(traces) == 0:
            traces = None
        elif len(traces) == 1:
            traces = traces[0]
        else:
            traces = Stream(traces)

        return traces
    
    def fetch_with_alias(self, alias, ascopy=False):
        if alias.upper() == 'Z':
            return self.Zdata(ascopy=ascopy)
        if alias.upper() == 'N':
            return self.Ndata(ascopy=ascopy)
        if alias.upper() == 'E':
            return self.Edata(ascopy=ascopy)
    
    def Zdata(self, ascopy=False):
        """
        Convenience method for fetching the trace that corresponds with
        id = self.Z, with an option to present the data as a copy. 
        Default is to provide an linked view of the data (i.e., changes
        made to this data will alter the source data)

        :: INPUT ::
        :param ascopy: [bool] return a deepcopy of the Z trace?
                    default = False

        :: OUTPUT ::
        :return trs: [obspy.core.trace.Trace] requested trace
        """
        trs = self.fetch_with_id(self.Z, ascopy=ascopy)
        return trs

    def Ndata(self, ascopy=False):
        """
        Convenience method for fetching the trace that corresponds with
        id = self.N, with an option to present the data as a copy. 
        Default is to provide an linked view of the data (i.e., changes
        made to this data will alter the source data)

        :: INPUT ::
        :param ascopy: [bool] return a deepcopy of the N trace?
                    default = False

        :: OUTPUT ::
        :return trs: [obspy.core.trace.Trace] requested trace
        """
        trs = self.fetch_with_id(self.N, ascopy=ascopy)
        return trs

    def Edata(self, ascopy=False):
        """
        Convenience method for fetching the trace that corresponds with
        id = self.E, with an option to present the data as a copy. 
        Default is to provide an linked view of the data (i.e., changes
        made to this data will alter the source data)

        :: INPUT ::
        :param ascopy: [bool] return a deepcopy of the E trace?
                    default = False

        :: OUTPUT ::
        :return trs: [obspy.core.trace.Trace] requested trace
        """
        trs = self.fetch_with_id(self.E, ascopy=ascopy)
        return trs

    # ############################## #
    # MISSING COMPONENT RULE METHODS #
    # ############################## #

    def apply_missing_component_rule(self):
        """
        Convenience method - wrapper for all three missing component rule (mcr)
        applying methods: "zeros", "clonez", and "clonehz"

        :: OUTPUT ::
        :return self: [InstrumentWindow] allow cascading
        """
        if self.mcr == 'zeros':
            self._apply_zeros_mcr()
        elif self.mcr == 'clonez':
            self._apply_clonez_mcr()
        elif self.mcr == 'clonehz':
            self._apply_clonehz_mcr()
        else:
            raise ValueError('Assigned self.mcr {self.mcr} is not supported.')
        return self

    def _apply_zeros_mcr(self):
        """
        Apply the zero-padding missing component rule used in Retailleau et al. (2021)
        wherein if one or both horizontal traces are missing, assing 0-vectors to
        both traces
        i.e., 
        Z N E -> Z N E
        y n n -> Z 0 0
        y y n -> Z 0 0
        y n y -> Z 0 0
        y y y -> Z N E

        """
        if self.mcr != 'zeros':
            raise AttributeError('This InstrumentStream has mcr {self.mcr} -- not "zeros"')
        if self.N is None or self.E is None:
            _trs = self.Zdata()
            if isinstance(_trs, Trace):
                self.traces = [_trs]
            elif isinstance(_trs, Stream):
                self.traces = _trs.traces
            self.N = None
            self.E = None
            self.processing.append({'apply_missing_component_rule': ('zeros', True)})
        else:
            self.processing.append({'apply_missing_component_rule': ('zeros', False)})
        return self
    
    def _apply_clonez_mcr(self):
        """
        Apply the clone vertical data missing component rule used in Ni et al. (2023)
        wherein if one or both horizontal traces are missing, clone data from the vertical
        channel into both horizontal channels.
        i.e., 
        Z N E -> Z N E
        y n n -> Z Z Z
        y y n -> Z Z Z
        y n y -> Z Z Z
        y y y -> Z N E
        """
        if self.mcr != 'clonez':
            raise AttributeError('This InstrumentStream has mcr {self.mcr} -- not "clonez"')
        if self.N is None or self.E is None:
            self.N = self.Z
            self.E = self.Z
            _trs = self.Zdata()
            if isinstance(_trs, Trace):
                self.traces = [_trs]
            elif isinstance(_trs, Stream):
                self.traces = _trs.traces
            self.processing.append({'apply_missing_component_rule': ('clonez', self.Z)})
        else:
            self.processing.append({'apply_missing_component_rule': ('clonez', False)})
        return self
    
    def _apply_clonehz_mcr(self):
        """
        Apply the clone horizontal or vertical data missing component rule adapted from
        Lara et al., (2023) and Ni et al. (2023) wherein if one horizontal channel trace
        is missing, the other horizontal trace is cloned. However, if both are missing, the
        vertical trace data are cloned into the horizontals.


        I.e.,
        Z N E -> Z N E
        y n n -> Z Z Z
        y y n -> Z N N
        y n y -> Z E E
        y Y Y -> Z N E
        """
        if self.mcr != 'clonehz':
            raise AttributeError('This InstrumentStream has mcr {self.mcr} -- not "clonehz')
        if self.N is None and self.E is None:
            self.mcr = 'clonez'
            self._apply_clonez_mcr()
            self.mcr = 'clonehz'
        elif self.N is None:
            self.N = self.E
            self.processing.append({'apply_missing_component_rule': ('clonehz', self.E)})
        elif self.E is None:
            self.E = self.N
            self.processing.append({'apply_missing_component_rule': ('clonehz', self.N)})
        else:
            self.processing.append({'apply_missing_component_rule': ('clonehz', False)})
        return self

    # ########################### #
    # COMPATABILITY CHECK METHODS #
    # ########################### #

    def are_traces_masked(self):
        """
        For each trace in self.trace, determine the masked status
        of its data and return a list of bools

        :: OUTPUT ::
        :return status: [list] sequential outputs from np.ma.is_masked(tr.data)
        """
        status = [np.ma.is_masked(_tr.data) for _tr in self.traces]
        return status
    
    def is_window_split(self):
        """
        Determine if the message data looks like it has been split
        i.e., duplicate trace IDs

        :: OUTPUT ::
        :return status: [bool] are there more traces than trace ids?
        """
        id_list = []
        for _tr in self.traces:
            _id = _tr.id
            if _id not in id_list:
                id_list.append(_id)
        if len(id_list) < len(self):
            status = True
        elif len(id_list) == len(self):
            status = False
        else:
            raise IndexError("more trace ids than traces...?")

        return status

    # ############################# #
    # TIMESERIES PROCESSING METHODS #
    # ############################# #

    def split_window(self):
        """
        Check if any traces in this InstrumentWindow are masked and
        split them into non-masked trace segments using the obspy.core.trace.Trace.split()
        method.

        If any traces are split, processing flags this process result as True
        """
        if any (self.are_traces_masked()):
            self.traces = self.split()
            self.processing.append({'split_window': True})
        else:
            self.processing.append({'split_window': False})
        return self
    
    def merge_window(self, **options):
        """
        Merge any split traces using the obspy.core.stream.Stream.merge() method
        if the InstrumentWindow appears to have split data.

        If data are not split, nothing is done to the data and self.processing is
        updated with a "merge_window": False

        otherwise, merge is applied and self.processing is updated with a True

        :: INPUTS ::
        :param **options: [kwargs] key-word arguments to pass to self.merge() to
                        change from default settings in ObsPy
                        see obspy.core.stream.Stream.merge()
        :: OUTPUT ::
        :return self: [InstrumentStream] enable cascading
        """
        if self.is_window_split():
            self.merge(**options)
            self.processing.append({'merge_window': True})
        else:
            self.processing.append({'merge_window': False})
        return self
        
    def sync_window_timing(self):
        """
        Synchronize data sampling between traces and with the target
        starttime and sampling_rate, interpolating trace data to align
        the first valid sample of a given trace at the nearest sample in the set
            [target_starttime + n/target_sampling_rate]
        
        Trace sampling rate is not altered by this process
        """
        # Initialize processing log
        proc = {'sync_window_timing': []}
        # Check that data in this iwind are not split
        if not self.is_window_split():
            # Check that data in this iwind are not masked
            if not any(self.are_traces_masked()):
                for _tr in self.traces:
                    # If starttimes are not on target - check if sampling is aligned with target
                    if _tr.stats.starttime != self.target_starttime:
                        # Get trace initial starttime
                        trt0 = _tr.stats.starttime
                        # Get delta time between target starttime and current starttime
                        dt = self.target_starttime - _tr.stats.starttime
                        # Convert to delta samles
                        dn = dt/self.target_samprate
                        # If samples are misaligned with target sampling - apply sync
                        if dn != int(dn):
                            # Front pad with the leading sample value
                            _tr.trim(
                                starttime=self.target_starttime - _tr.stats.delta,
                                pad=True,
                                fill_value=_tr.data[0]
                            )
                            # Sync time using interpolate
                            _tr.interpolate(
                                _tr.stats.sampling_rate,
                                starttime = self.target_starttime
                            )
                            # Remove any padding values
                            _tr.trim(starttime=trt0,
                                     nearest_sample=True,
                                     pad=False,
                                     fill_value=self.fill_value)
                            # If sampling is misaligned
                            proc['sync_window_timing'].append({_tr.id: 'sync required'})
                        # If trace sampling is aligned with target, but starttime does not match
                        else:
                            proc['sync_window_timing'].append({_tr.id: 'pad required'})
                    # If trace starttime matches target
                    else:
                        proc['sync_window_timing'].append({_tr.id: "already syncd"})
            # If any data are masked, raise error
            else:
                raise UserWarning('InstrumentWindow contains masked traces - cannot conduct sync')
        # If any data are split into parts, raise error
        else:
            raise UserWarning('InstrumentWindow contains split traces - cannot conduct sync')
        # Update processing
        self.processing.append(proc)
        return self
                            
    def trim_window(self, pad=True, **options):
        """
        Trim traces to the target starttime and endtime 
            target_starttime + target_npts/target_samprate
        using the obspy.core.stream.Stream.trim() method with default values here that
        differ from obspy defaults

        :: INPUTS ::
        :param pad: [bool] - change default to True - generate masked traces
        :param **options: [kwargs] additional kwarg options passed to Stream.trim()

        :: OUTPUT ::
        :return self: [InstrumentWindow] enable cascading
        """
        # Calculate window start and endtimes
        ts = self.target_starttime
        te = ts + (self.target_npts - 1) / self.target_samprate
        # Check if any of the traces require padding/trimming
        if any(_tr.stats.starttime != ts for _tr in self.traces):
            run_trim = True
        elif any(_tr.stats.endtime != te for _tr in self.traces):
            run_trim = True
        else:
            run_trim = False
        if run_trim:
            self.trim(
                starttime=ts,
                endtime=te,
                pad=pad,
                **options
            )
            if any(_tr.stats.npts != self.target_npts for _tr in self.traces):
                raise UserWarning(f'Not all trimmed traces in this InstrumentWindow meet target_npts {self.target_npts}')
            self.processing.append({'trim_window': True})
        else:
            self.processing.append({'trim_window': False})
        return self

    def fill_window_gaps(self, fill_value=0.):
        """
        Wrapper for numpy.ma.filled to apply a fill_value in-place to masked
        trace data arrays

        :: INPUT ::
        :param fill_value: [int], [float], or [None]

        :: OUTPUT ::
        :return self: [InstrumentWindow] enable cascading
        """
        any_padding = False
        for _tr in self.traces:
            if np.ma.is_masked(_tr.data):
                _tr.data = np.ma.filled(_tr.data, fill_value=fill_value)
                _tr.stats.processing.append(
                    f'NumPy {np.__version__} numpy.ma.filled(fill_value={fill_value})')
                any_padding = True
        self.processing.append({'fill_window_gaps': any_padding})
        return self        
        
    def normalize_window(self, method='peak'):
        if not isinstance(method, str):
            raise TypeError('method must be type str')
        if method.lower() in ['peak','max','minmax','maximum']:
            self.normalize()
            self.processing.append({'normalize_window': 'max'})
        elif method.lower() in ['std','sigma','standard','standardscalar']:
            for _tr in self.traces:
                # Get the std of the data
                if np.ma.is_masked(_tr.data):
                    _tnf = np.nanstd(_tr.data.data[~_tr.data.mask])
                else:
                    _tnf = np.nanstd(_tr.data)
                if np.isfinite(_tnf):
                    # Apply normalization
                    _tr.normalize(_tnf)
                else:
                    raise ValueError(f'{_tr.id} | standardscalar (std) normalization prduces a non-fininite scalar {_tnf}')
            self.processing.append({'normalize_window': 'std'})
        else:
            raise ValueError(f'method "{method}" not supported. See documentation')
        return self

    # ################### #
    # TRANSLATION METHODS #
    # ################### #
    def copy(self):
        """
        return a deepcopy of this InstrumentWindow object
        """
        return deepcopy(self)
    
    def on_target(self, mode='any'):
        """
        Provide an assessment if the traces in this InstrumentWindow meet the
        target windowing requirements with options on the level of detail output

        :: INPUT ::
        :param mode: [str] style of output. Supported values & output types
                'all' - return [bool] - do all traces meet all targets? (default)
                'any' - return [bool] - do any traces meet any targets? (less useful)
                'call' - return [dict] - keyed by trace.id - does a given trace meet all targets?
                'cany' - return [dict] - keyed by trace.id - does a given trace meet any targets?
                'debug' - return [pandas.DataFrame] - [bool] values with columns labeled by trace.id
                                                    and indices by target attributes
        
        :: OUTPUT ::
        :return output: [bool], [dict], or [pandas.DataFrame] see above.
        
        """
        index_labels = ['starttime', 'endtime', 'sampling_rate', 'npts', 'masked']
        bool_dict = {} 
        # Run Checks
        for _tr in self.traces:
            bool_list = []
            # All starttimes must be target starttimes
            bool_list.append(_tr.stats.starttime == self.target_starttime)
            endtime = self.target_starttime + (self.target_npts - 1)/self.target_samprate
            bool_list.append(_tr.stats.endtime == endtime)
            bool_list.append(_tr.stats.sampling_rate == self.target_samprate)
            bool_list.append(_tr.stats.npts == self.target_npts)
            bool_list.append(~np.ma.is_masked(_tr.data))
            bool_dict.update({_tr.id: bool_list})    
        if mode in ['any', 'all']:
            slist = []
            for _v in bool_dict.values():
                slist += _v
            if mode == 'any':
                output = any(slist)
            elif mode == 'all':
                output = all(slist)
        elif mode in ['cany','call']:
            output = {}
            for _k, _v in bool_dict.items():
                if mode == 'cany':
                    output.update({_k, any(_v)})
                elif mode == 'call':
                    output.update({_k, all(_v)})
        # Debug handling
        elif mode == 'debug':
            output = DataFrame(bool_dict, index=index_labels)
        else:
            raise ValueError(f'mode {mode} not supported. See documentation')
        # Send output
        return output
   
    def to_stream(self, ascopy=False):
        """
        Create a stream representation of this InstrumentStream with an option
        to make the view a deepcopy of the source data (default is False)
        """
        st = Stream()
        for _c in self.component_order:
            st.append(self.fetch_with_alias(_c, ascopy=ascopy))
        return st
    
    def to_numpy(self, ascopy=False, other_order=None):
        """
        Run a series of checks to make sure that trace data have the target
        starttimes, endtimes, and sampling_rates and then produce a numpy.ndarray
        with dimensions (3, target_npts)

        :: INPUTS ::
        :param ascopy: [bool] - provide view of data as a deepcopy?
        :param other_order: [str] - alternative order to place data in
                        must be a combination of "Z" "N" and "E" as
                        a string: e.g., 'NEZ'
        
        :: OUTPUT ::
        :return data: [numpy.ndarray] data array
        """
        if not self.on_target(mode='all'):
            raise AttributeError('Not all attributes of data in traces match target values - use self.on_target(mode="debug") to diagnose')
        # Compose holder array if 
        data = np.full(shape=(3, self.target_npts), fill_value=np.nan, dtype=np.float32)
        if isinstance(other_order, str):
            if all(_c.upper() in 'ZNE' for _c in other_order):
                order = other_order.upper()
        else:
            order = self.component_order

        for _i, _c in enumerate(order):
            data[_i, :] = self.fetch_with_alias(_c, ascopy=ascopy).data
        return data
            
    def to_pwind(self, ascopy=False):
        """
        Return processed data and metadata as a
        wyrm.core.data.PredictionWindow object
        with the option to pass data as a copy or alias of 
        data held within this InstrumentWindow.

        :: INPUT ::
        :param ascopy: [bool] Should data be transferred as a copy?
                        default False
        
        :: OUTPUT ::
        :return pwind: [wyrm.core.data.PredictionWindow]
        """
        kwargs = {'data': self.to_numpy(ascopy=ascopy),
                  'id': self.Z[:-1],
                  't0': self.Zdata().stats.starttime.timestamp,
                  'samprate': self.Zdata().stats.sampling_rate,
                  'blinding': self.target_blinding,
                  'labels': self.component_order,
                  'model_name': self.model_name}
        pwind = PredictionWindow(**kwargs)
        return pwind

    # ################# #
    # EXAMPLE WORKFLOW  #
    # ################# #
    
    def default_processing(self, taper_sec = 0.06, filter_kwargs={'type': 'bandpass', 'freqmin': 1, 'freqmax': 45}, ascopy=False):
        """
        Suggested pre-processing steps for EQTransformer based on Ni et al. (2023) and references therein

        :: INPUTS ::
        :param taper_sec: [float] length of maximum taper length passed to Stream.taper(max_length=taper_sec)
        :param filter_kwargs: [None] or [dict] input args (formatted as kwargs) and kwargs for Stream.filter(**filter_kwargs)
        :param ascopy: [bool] should resulting pwind be a deepcopy of the data in this InstrumentWindow?

        :: OUTPUT ::
        :return pwind: [wyrm.core.data.PredictionWindow]

        Processing Steps:
            Handle masked data
                split_window()
                filter()
                detrend('demean')
                detrend('lienar')
            Hit target sampling
                resample(target_samprate)
            Clean up edges
                taper(None, max_length=taper_sec, side='both')
            Clean up timing
                merge_window()
                sync_window_timing()
                trim_window()
            Clean up gaps
                fill_window_gaps()
            Prepare for export
                normalize_window()
            Export
                to_pwind()
        """
        # Split, if needed
        self.split_window()
        # Filter if specified
        if filter_kwargs:
            self.filter(**filter_kwargs)
        # Detrend and demean
        self.detrend('demean')
        self.detrend('linear')
        # Resample data
        self.resample(self.target_samprate)
        self.processing.append(f'obspy resample({self._tsr})')
        # Taper data
        self.taper(None, max_length=taper_sec, side="both")
        self.processing.append(f'obspy taper(None, max_length={taper_sec}, side="both")')
        # Merge data
        self.merge_window()
        # Sync timings
        self.sync_window_timing()
        # Trim and pad window
        self.trim_window()
        # Fill masked values
        self.fill_window_gaps(fill_value=0)
        # Normalize data
        self.normalize_window()
        # Convert to PredictionWindow object
        pwind = self.to_pwind(ascopy=ascopy)
        return pwind

    def __repr__(self):
        """
        User friendly string representation of this InstrumentWindow
        """
        keys = {"Z": self.Z, "N": self.N, "E": self.E}
        site = '.'.join(self.Z.split(".")[:3])
        inst = self.Z.split('.')[-1][:-1]
        # WindowMsg parameter summary
        rstr = f"{len(self)} trace(s) in InstrumentStream | "
        rstr += f'Site: {site} | Instrument: {inst}\n'
        rstr += "Target > "
        rstr += f"model: {self.model_name} ð "
        rstr += f"dims: ({len(self.component_order)}, {self.target_npts}) ð "
        rstr += f"S/R: {self.target_samprate} Hz | "
        rstr += f"missing comp. rule: {self.mcr}\n"
        # Unique Trace List
        for _i, _tr in enumerate(self.traces):
            # Get alias list
            aliases = [_o for _o in self.component_order if keys[_o] == _tr.id]
            rstr += f"{_tr.__str__()}"
            # Add alias information to trace line
            if len(aliases) > 0:
                rstr += " | ("
                for _o in aliases:
                    rstr += _o
                rstr += ") alias\n"
            else:
                rstr += " | [UNUSED]\n"
        rstr += f" Trim Target ð {self.target_starttime} - {self.target_starttime + self.target_npts/self.target_samprate} ð"
        rstr += f'\n Window Processing: {self.processing}'
        return rstr  


    def __str__(self):
        """
        String representation of the arguments used to create this InstrumentWindow
        """


###################################################################################
# PREDICTION WINDOW CLASS DEFINITION ##############################################
###################################################################################

class PredictionWindow(object):
    """
    This object houses a data array and metadata attributes for documenting a pre-processed
    data window prior to ML prediction and predicted values by an ML operation. It provides
    options for converting contained (meta)data into various formats.
    """
    def __init__(self,
        data=None,
        labels=[],
        npts=None,
        id='..--.',
        t0=0.,
        samprate=1.,
        blinding=0,
        model_name=None,
        weight_name=None
        ):
        """
        Initialize a PredictionWindow (pwind) object

        :: INPUTS ::
        :param data: [numpy.ndarray] or None
                    Data array to house in this PredictionWindow. Must be a
                    2-dimensional array with one axis that has the same number
                    of entries (rows or columns) as there are iterable items
                    in `labels`. This axis will be assigned as the 0-axis.
        :param labels: [list-like] of [str] labels to associate with row vectors in data.
                    Must match the 0-axis of self.data.
        :param npts: [int] number of data samples in the 1-axis of self.data
        :param id: [str] instrument ID code for this window. Must conform to the
                        N.S.L.bi notation (minimally "..."), where:
                            N = network code (2 characters max)
                            S = station code (5 characters max)
                            L = location code (2 characters)
                            bi = band and instrument characters for SEED naming conventions
        :param t0: [float] timestamp for first sample (i.e., seconds since 1970-01-01T00:00:00)
        :param samprate: [float] sampling rate in samples per second
        :param blinding: [int] number of samples to blind on the left and right ends of this window
                            when stacking sequential prediction windows

        :param model_name: [str] name of the ML model this window is associated with this pwind
        :param weight_name: [str] name of the pretrained model weights associated with this pwind
        :
        """
        # data compat. check
        if not isinstance(data, (type(None), np.ndarray)):
            raise TypeError('data must be type numpy.ndarray or NoneType')
        elif data is None:
            self.data = data
            self._blank = True
        elif data.ndim != 2:
            raise IndexError(f'Expected a 2-dimensional array. Got {data.ndim}-d')
        else:
            self.data = data
            self._blank = False

        # labels initial compat check
        if not isinstance(labels, (list, tuple, str)):
            raise TypeError('labels must be type list, tuple, or str')
        elif isinstance(labels, (tuple, str)):
            labels = [_l for _l in labels]

        # npts initial compat check
        if npts is None:
            self.npts = None
        else:
            self.npts = wuc.bounded_intlike(
                npts,
                name='npts',
                minimum=1,
                maximum=None,
                inclusive=True
            )


        # labels/npts/data crosschecks
        if self.data is not None:
            if len(labels) == self.data.shape[0]:
                self.labels = labels
                if self.npts != self.data.shape[1]:
                    self.npts = self.data.shape[1]
            elif len(labels) == self.data.shape[1]:
                self.labels = labels
                if self.npts != self.data.shape[0]:
                    self.npts = self.data.shape[0]
                self.data = self.data.T
            else:
                raise IndexError(f'Number of labels ({len(labels)}) not in self.data.shape ({self.data.shape})')
        else:
            self.labels = labels

        # id compat. check
        if not isinstance(id, (str, type(None))):
            raise TypeError('id must be type str or NoneType')
        elif id is None:
            self.id = None
        elif len(id.split('.')) == 4:
            self.id = id
        else:
            raise SyntaxError('str-type id must consit of 4 "." delimited string elements to match N.S.L.C notation')
        # t0 compat check
        if not isinstance(t0, (int, float)):
            raise TypeError('t0 must be type int, float, or obspy.core.utcdatetime.UTCDateTime')
        elif isinstance(t0, UTCDateTime):
            self.t0 = t0.timestamp
        else:
            self.t0 = float(t0)
        # samprate compat checks
        self.samprate = wuc.bounded_floatlike(
            samprate,
            name='samprate',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # blinding compat checks
        self.blinding = wuc.bounded_intlike(
            blinding,
            name='blinding',
            minimum=0,
            maximum=None,
            inclusive=True
        )
        
        # model_name compat check
        if not isinstance(model_name, (str, type(None))):
            raise TypeError('model_name must be type str or NoneType')
        else:
            self.model_name = model_name
        # weight_name compat check
        if not isinstance(weight_name, (str, type(None))):
            raise TypeError('weight_name must be type str or NoneType')
        else:
            self.weight_name = weight_name


    def get_metadata(self):
        """
        Return a dictionary containing metadata attributes from this PredictionWindow

        :: OUTPUT ::
        :return meta: [dict] metadata dictionary containing the 
            following attributes as key:value pairs (note: any of these may also have a None value)
            'id': [str] - station/instrument ID
            'labels': [list] of [str] - string names of model/data labels
            'npts': [int] - number of time-indexed samples in this window
            't0': [float] - starttime of windowed data (epoch seconds / timestamp)
            'samprate': [float] - sampling rate of windowed data (samples per second)
            'model_name': [str] or None - name of ML model this window corresponds to
            'weight_name': [str] or None - name of pretrained ML model weights assocaited with this window
        """
        meta = {'id': self.id,
                'labels': self.labels,
                'npts': self.npts,
                't0': self.t0,
                'samprate': self.samprate,
                'model_name': self.model_name,
                'weight_name': self.weight_name,
                'blinding': self.blinding}
        return meta
    
    def to_stream(self, apply_blinding=False):
        """
        Return an obspy.core.stream.Stream representation of the data and labels in this PredictionWindow
        The first letter of each label is appended to the end of the channel name for each trace in the
        output Stream object


        :: INPUT ::
        :param apply_blinding: [bool] should data vectors be converted into masked arrays with
                                     masked values in blinded locations?

        :: OUTPUT ::
        :return st: [obspy.core.stream.Stream]
        """
        st = Stream()
        header = dict(zip(['network','station','location','channel'], self.id.split('.')))
        header.update({'starttime': UTCDateTime(self.t0), 'sampling_rate': self.samprate})
        for _i, _l in enumerate(self.labels):
            _data = self.data[_i, :].copy()
            if apply_blinding:
                _data = np.ma.masked(data=_data, mask=np.full(shape=_data.shape, fill_value=False))
                _data.mask[:self.blinding] = True
                _data.mask[-self.blinding:] = True
            tr = Trace(data=_data, header=header)
            tr.stats.channel += _l[0].upper()
            st += tr
        return st

    def to_npy(self, dir, fstr='{site}_{inst}_{mod}_{t0:016.6f}_{sr:05.2f}Hz.npy'):
        site = '.'.join(self.id.split('.')[:2])
        inst = '.'.join(self.id.split('.')[2:])
        mod = '.'.join([self.model_name, self.weight_name])
        t0 = self.t0
        sr = self.samprate
        fname = fstr.format(site=site, inst=inst, mod=mod, t0=t0, sr=sr)
        pfout = os.path.join(dir,fname)
        np.save(pfout, self.stack)

    def split_for_ml(self):
        """
        Convenience method for splitting the data and metadata in this PredictionWindow into
        a pytorch Tensor and a metadata dictionary 

        :: OUTPUTS ::
        :return tensor: [torch.Tensor] Tensor formatted version of self.data
        :return meta: [dict] metadata dictionary - see self.get_metadata()
        """
        meta = self.get_metadata()
        if np.ndim(self.data) == 3:
            tensor = torch.Tensor(self.data.copy())
        elif np.ndim(self.data) == 2:
            tensor = torch.Tensor(self.data.copy()[np.newaxis, :, :])
        return tensor, meta

    def copy(self):
        """
        Return a deepcopy of this PredictionWindow
        """
        return deepcopy(self)

    def __repr__(self):
        rstr =  'Prediction Window\n'
        rstr += f'{self.id} | t0: {UTCDateTime(self.t0)} | S/R: {self.samprate: .3f} sps | Dims: {self.data.shape}\n'
        rstr += f'Model: {self.model_name} | Weight: {self.weight_name} | Labels: {self.labels} | Blind: {self.blinding} \n'
        if self._blank:
            rstr += f'np.zeros(shape={self.shape})'
        else:
            rstr += f'{self.data.__repr__()}'
        return rstr
    
    def __str__(self):
        rstr = 'wyrm.core.window.prediction.PredictionWindow('
        rstr += f'model_name={self.model_name}, weight_name={self.weight_name}, '
        if self._blank:
            rstr += f'data=None, '
        else:
            rstr += f'data={self.data}, '
        rstr += f'id={self.id}, samprate={self.samprate}, t0={self.t0}, labels={self.labels})'
        return rstr

    def __eq__(self, other, include_flags=False):
        """
        Rich representation of self == other. Includes an option to include
        status flags in the comparison.

        :: INPUT ::
        :param other: [object]
        :param include_flags: [bool] - include self._blank == other._blank in comparison?

        :: OUTPUT ::
        :return status: [bool]
        """
        bool_list = []
        if not isinstance(other, PredictionWindow):
            status = False
        else:
            bool_list = [(self.data == other.data).all()]
            for _attr in ['t0','model_name','weight_name','samprate','labels','shape']:
                bool_list.append(eval(f'self.{_attr} == other.{_attr}'))
            if include_flags:
                for _attr in ['_blank']:
                    bool_list.append(eval(f'self.{_attr} == other.{_attr}'))

            if all(bool_list):
                status = True
            else:
                status = False 
        return status

    def update_from_seisbench(self, model, weight_name=None, is_prediction=True):
        """
        Update attributes with a seisbench.model.WaveformModel child-class object
        if they pass compatability checks with the current self.data array in
        this PredictionWindow

        :: INPUTS ::
        :param model: [seisbench.models.WaveformModel] child-class object of the 
                        WaveformModel baseclass (e.g., EQTransformer)
        :param weight_name: [str] or [None] weight name to update self.weight_name
                        with (None input indicates no change)
        :param is_prediction: [bool] use the labels for predictions (True) or inputs (False)
                            True - data axis 0 scaled by len(model.labels)
                            False - data axis 0 scaled by len(model.component_order)
        :: OUTPUT ::
        :return self: [wyrm.core.window.prediction.PredictionWindow] enable cascading
        """
        # model compat check
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError('model must be type seisbench.models.WaveformModel')
        # weight_name compat check
        if not isinstance(weight_name, (str, type(None))):
            raise TypeError('weight_name must be type str or NoneType')
        # dim0 compat check
        if not isinstance(is_prediction, bool):
            raise TypeError('is_prediction must be type bool')
        
        if not is_prediction:
            dim0 = model.component_order
            d0name = 'component_order'
        else:
            dim0 = model.labels
            d0name = 'labels'
        # Fetch dim1
        dim1 = model.in_samples
        # Check that the input model has viable data/prediction array dimensions
        if dim0 is None or dim1 is None:
            raise ValueError(f'model.{d0name} and/or model.in_samples are None - canceling update - suspect model is a base class WaveformModel, rather than a child class thereof')
        else:
            # If data is None, populate
            if self.data is None:
                self.data = np.zeros(shape=(len(dim0), dim1), dtype=np.float32)
                self._blank = True
            else:
                if self.data.shape != (len(dim0), dim1):
                    raise IndexError(f'proposed data array dimensions from model ({len(dim0)}, {dim1}) do not match the current data array ({self.data.shape})')
                self._blank = False
            # Update weight_name if type str
            if isinstance(weight_name, str):
                self.weight_name = weight_name
            # Update labels
            self.labels = [_l for _l in dim0]
            # Update blinding
            bt = model._annotate_args['blinding'][1]
            if all(np.mean(bt) == _b for _b in bt):
                self.blinding = int(np.mean(bt))
            else:
                raise ValueError('model._annotate_args["blinding"] with mixed blinding values not supported.')
            # Update samprate
            self.samprate = model.sampling_rate
        return self


###################################################################################
# TRACE BUFFER CLASS DEFINITION ###################################################
###################################################################################

class TraceBuffer(Trace):
    """
    Adapted version of the obspy.realtime.rttrace.RtTrace class
    that provides handling for gappy data that is not developed
    in the current release of obspy (1.4.0).

    A key difference in this class compared to the RtTrace class
    is that all trace.data attributes are handled as numpy.ma.masked_array
    objects. The data.mask sub-attribute is used to validate
    window completeness in processing steps leading into ML input-data
    window production.

    """
    def __init__(
        self,
        max_length=1.0,
        fill_value=None,
        method=1,
        interpolation_samples=-1
    ):
        """
        Initialize an empty TraceBuffer object

        :: INPUTS ::
        :param max_length: [int], [float], or [None]
                            maximum length of the buffer in seconds
                            with the oldest data being trimmed from
                            the trace (left trim) in the case where
                            max_length is not None
        :param fill_value: [None], [float], or [int]
                            fill_value to assign to all masked
                            arrays associated with this TraceBuffer
        :param method: method kwarg to pass to obspy.core.stream.Stream.merge()
                        internal to the TraceBuffer.append() method
                        NOTE: Default here (method=1) is different from the
                        default for Stream.merge() (method=0). This default
                        was chosen such that overlapping data are handled
                        with interpolation (method=1), rather than gap generation
                        (method=0).
        :param interpolation_samples: interpolation_samples kwarg to pass to
                        obspy.core.stream.Stream.merge() internal to
                        the TraceBuffer.append() method.
                        NOTE: Default here (-1) is different form Stream.merge()
                        (0). This was chosen such that all overlapping samples
                        are included in an interpolation in an attempt to suppress
                        abrupt steps in traces
        """
        # Inherit from Trace
        super().__init__()
        # Compatability check for max_length
        self.max_length = wuc.bounded_floatlike(
            max_length, name="max_length", minimum=0.0, maximum=None, inclusive=False
        )

        # Compatability check for fill_value
        if not isinstance(fill_value, (int, float, type(None))):
            raise TypeError("fill_value must be type int, float, or None")
        else:
            self.fill_value = fill_value

        # Compatability check for method
        if method not in [0, 1, -1]:
            raise ValueError(
                "stream.merge method must be -1, 0, or 1 - see obspy.core.stream.Stream.merge"
            )
        else:
            self.method = method

        # Compatability check for interpolation_samples:
        self.interpolation_samples = wuc.bounded_intlike(
            interpolation_samples,
            minimum=-1,
            maximum=self.max_length * 1e6,
            inclusive=True,
        )
        if not isinstance(interpolation_samples, int):
            raise TypeError(
                "interpolation_samples must be type int - see obspy.core.stream.Stream.merge"
            )
        else:
            self.interpolation_samples = interpolation_samples

        # Set initial state of have_appended trace
        self._have_appended_trace = False
        # Initialize buffer statistics parameters
        self.filled_fraction = 0
        self.valid_fraction = 1

    def copy(self):
        """
        Create a deepcopy of this TraceBuffer
        """
        return deepcopy(self)

    def __eq__(self, other):
        """
        Implement rich comparison of TraceBuffer objects for "==" operator.

        Traces are the same if their data, stats, and mask arrays are the same
        """
        if not isinstance(other, TraceBuffer):
            return False
        else:
            return super(TraceBuffer, self).__eq__(other)

    def to_trace(self):
        """
        Return a deepcopy obspy.core.trace.Trace representation
        sof this TraceBuffer object
        """
        tr = Trace(data=self.copy().data, header=self.copy().stats)
        return tr

    def enforce_max_length(self):
        """
        Enforce max_length on data in this TraceBuffer using
        the endtime of the TraceBuffer as the reference datum
        and trimming using _ltrim (left-trim) if the data length
        exceeds self.max_length in seconds

        note: trim is conducted in-place, so trimmed-off data will
        be lost
        """
        te = self.stats.endtime
        sr = self.stats.sampling_rate
        max_samp = int(self.max_length * sr + 0.5)
        ts = te - max_samp / sr
        if ts > self.stats.starttime:
            self._ltrim(
                ts,
                pad=True,
                nearest_sample=True,
                fill_value=self.fill_value,
            )
        else:
            pass
        return self

    def append(self, trace):
        """
        Append a candidate trace to this RtTraceBufferer for a suite of scenarios:
            1) first ever append - self._first_append()

        IF id, stats.calib, stats.sampling_rate, data.dtype all match
            2) appending data with gaps - uses self._gappy_append()
            3) appending data without gaps - uses self._contiguous_append()
            takes away extra processing steps used to handle gaps, saving
            processing time relative to self._gappy_append().

        IF id matches
            4) appending data with gaps larger than self.max_length - uses self._first_append()
                This allows for restart of buffers in the event of data outages or
                changes in acquisition parameters without crashing the module.

        :: INPUT ::
        :param trace: [obspy.core.trace.Trace] candidate trace to append
        """
        # Compatability check that input trace is an obspy trace
        if not isinstance(trace, Trace):
            raise TypeError('input "trace" must be type obspy.core.trace.Trace')
        # If this is truly the first append to buffer, proceed directly to append
        if not self._have_appended_trace:
            self._first_append(trace)
        # Otherwise conduct checks
        else:
            # Get information on relative timing
            status, gap_sec = self._assess_relative_timing(trace)
            # If trace is fully before or after current contents reset
            if status in ["leading", "trailing"]:
                # If the gap is larger than max_length, run as _first_append
                if gap_sec > self.max_length:
                    # Do cursory check that NSLC matches
                    if self.id == trace.id:
                        self._have_appended_trace = False
                        self._first_append(trace)
                    else:
                        emsg = f"id of self ({self.id}) and trace ({trace.id}) mismatch"
                        raise ValueError(emsg)
                # If the gap is smaller than max_length, run as _gappy_append
                else:
                    self.check_trace_compatability(trace)
                    self._gappy_append(trace, method=1)
            elif status == "overlapping":
                self.check_trace_compatability(trace)
                # If either self or trace data are masked, treat as gappy
                if np.ma.is_masked(self.data) or np.ma.is_masked(trace.data):
                    self._gappy_append(trace)
                # Otherwise, conduct as a contiguous append without gap safety checks
                # - saves compute time, but exposes
                else:
                    self._contiguous_append(trace)
            else:
                raise TypeError(
                    f'_check_overlap_status retured status of {status}: must be "leading", "lagging", or "overlapping"'
                )
        return self
    


    def __iadd__(self, other):
        """
        Magic method - inplace add (+=)

        self += other

        This iadd method wraps the TraceBuffer.append() method and accepts "other" inputs
        obspy.core.trace.Trace and wyrm.core.dadta.TraceBuffer (a Trace child-class), treating
        the right-hand-side as an obspy.core.trace.Trace object being appended to the left-hand-side
        
        """
        if not isinstance(other, Trace):
            raise TypeError('other must be type TraceBuffer or type obspy.core.trace.Trace')
        elif isinstance(other, TraceBuffer):
            self.append(other.to_trace())
        elif isinstance(other, Trace):
            self.append(other)
        return self

    # ################## #
    # Append Subroutines #
    # ################## #
    def _first_append(self, trace):
        """
        Special case append where data appended can exceed the self.max_length
        limit placed on all subsequent appends to this buffer. Data and Stats
        are copied from trace and data.

        :: INPUT ::
        :param trace: [obspy.core.trace.Trace] trace object to append

        :: ATTRIBUTES USED ::
        :attrib data: updated with trace.data
        :attrib stats: updated with trace.stats
        :attrib _have_appended_trace: update to True
        """
        if self._have_appended_trace:
            raise AttributeError(
                "Cannot execute _first_append() with self._have_appended_trace == True"
            )
        if not isinstance(trace, Trace):
            raise TypeError("trace must be type obspy.Trace")
        self.data = trace.copy().data
        self.stats = trace.copy().stats
        self._have_appended_trace = True
        self._update_buffer_stats()
        return self

    def _contiguous_append(self, trace):
        """
        Append data method for when contiguous data are anticipated
        """
        if not isinstance(trace, Trace):
            raise TypeError("trace must be type obspy.Trace")
        tr_self = self.to_trace()
        st = Stream([tr_self, trace.copy()])
        st.merge(
            fill_value=self.fill_value,
            method=self.method,
            interpolation_samples=self.interpolation_samples,
        )
        if len(st) == 1:
            tr = st[0]
            self.data = tr.data
            self.stats = tr.stats
            self.enforce_max_length()._update_buffer_stats()
        else:
            emsg = f"TraceBuffer({self.id}).append "
            emsg += f"produced {len(st)} distinct traces "
            emsg += "rather than 1 expected. Canceling append."
            print(emsg)
        return self

    def _gappy_append(self, trace):
        """
        Append data method for when gappy (masked) data are anticipated
        to be present due to self or trace having gaps or the existance of
        a gap between self and trace data. This private method includes a
        split() method internally to create a stream of contiguous data
        segments and then merge to re-merge the data into (semi) contiguous
        data. This also allows for nonsequential packet loading (i.e., gap
        back-filling) that could occur in some marginal cases.

        This added split() requires additional compute resources, but is safe
        for both gappy and contiguous data.

        :: INPUT ::
        :param trace: [obspy.core.trace.Trace] candidate trace to append

        :: ATTRIBUTES USED ::
        :attrib data: TraceBuffer data array
        :attrib stats: TraceBuffer metadata
        :attrib method: kwarg value passed to obspy.Stream.merge()
        :attrib interpolation_samples: kwarg value passed to obspy.Stream.merge
        """
        if not isinstance(trace, Trace):
            raise TypeError("trace must be type obspy.Trace")
        # Create a trace representation
        tr_self = self.to_trace()
        # Put trace and tr_self into a stream
        st = Stream([tr_self, trace.copy()])
        # Conduct split to convert gappy data into contiguous, unmasked chunks
        st = st.split()
        # Use merge to stitch data back together
        st.merge(
            fill_value=self.fill_value,
            method=self.method,
            interpolation_samples=self.interpolation_samples,
        )
        # Check that one trace resulted rom merge
        if len(st) == 1:
            tr = st[0]
            self.data = tr.data
            self.stats = tr.stats
            self.enforce_max_length()._update_buffer_stats()
        else:
            emsg = f"TraceBuffer({self.id}).append "
            emsg += f"produced {len(st)} distinct traces "
            emsg += "rather than 1 expected. Canceling append."
            print(emsg)
        return self

    def _assess_relative_timing(self, trace):
        """
        Assesss the relative start/endtimes of TraceBuffer and candidate trace
        :: INPUT ::
        :param trace: [obspy.core.trace.Trace] candidate trace

        :: OUTPUT ::
        :return status: [str] 'leading' - trace is entirely before this TraceBuffer
                              'trailing' - trace is entirely after this TraceBuffer
                              'overlapping' - trace overlaps with this TraceBuffer
                              None - in the case where something goes wrong with
                                    types of timing information
        :return gap_sec: [float] - in the case of 'leading' and 'trailing' this is
                                    the length of the gap in seconds
                                   in the case of 'overlapping', defaults to None
        """
        starts = None
        ends = None
        if not isinstance(trace, Trace):
            raise TypeError('input "trace" must be type obspy.core.trace.Trace')
        sss = self.stats.starttime
        sse = self.stats.endtime
        tss = trace.stats.starttime
        tse = trace.stats.endtime
        # if trace starts before self starts
        if tss < sss:
            starts = "before"
        # if trace starts during self
        elif sss <= tss <= sse:
            starts = "during"
        # if trace starts after self ends
        elif sse < tss:
            starts = "after"

        # If trace ends before self ends
        if tse < sse:
            ends = "before"
        elif sss <= tse <= sse:
            ends = "during"
        elif sse < tse:
            ends = "after"

        if starts == ends == "before":
            status = "leading"
            gap_sec = sss - tse
        elif starts == ends == "after":
            status = "trailing"
            gap_sec = tss - sse
        else:
            if starts is not None and ends is not None:
                status = "overlapping"
                gap_sec = None
            else:
                status = None
                gap_sec = None
        return status, gap_sec

    # ########################################## #
    # Trace Metadata / Dtype Compatability Check #
    # ########################################## #

    def check_trace_compatability(self, trace):
        """
        Check compatability of key attriubters of self and trace:
        .id
        .stats.calib
        .stats.sampling_rate
        .data.dtype

        if any mismatch, return False
        else, return True
        """
        if not isinstance(trace, Trace):
            raise TypeError("trace must be type obspy.core.trace.Trace")
        xc_attr = ["id", "stats.calib", "stats.sampling_rate", "data.dtype"]
        bool_list = []
        for _attr in xc_attr:
            try:
                bool_list.append(
                    self._check_attribute_compatability(trace, attr_str=_attr)
                )
            except ValueError:
                bool_list.append(False)
        return all(bool_list)

    def _check_attribute_compatability(self, trace, attr_str="id"):
        """
        Check compatability of an attribute using eval() statements for
        self and trace:

        i.e., eval(f'self.{attr_str}') != eval(f'trace.{attr_str}'):
        Raises ValueError if mismatch
        returns True if match
        """
        self_val = eval(f"self.{attr_str}")
        trace_val = eval(f"trace.{attr_str}")
        if self_val != trace_val:
            emsg = f'attribute ".{attr_str}" mismatches'
            emsg += f"for buffer ({self_val}) and candidate trace ({trace_val})"
            raise ValueError(emsg)
        else:
            return True

    #####################################
    # Data completeness check & display #
    #####################################
    def _update_buffer_stats(self):
        """
        Update the self.filled_fraction and self.valid_fraction
        attributes of this TraceBuffer with its current contents
        """
        self.filled_fraction = self.get_filled_fraction()
        self.valid_fraction = self.get_unmasked_fraction()
        return self

    def get_filled_fraction(self):
        """
        Get the fractional amount of data (masked and unmasked)
        present in this TraceBuffer relative to self.max_length
        """
        ffnum = self.stats.endtime - self.stats.starttime
        ffden = self.max_length
        return ffnum / ffden

    def get_trimmed_valid_fraction(
        self, starttime=None, endtime=None, wgt_taper_len=0.0, wgt_taper_type="cosine"
    ):
        """
        Get the valid (unmasked) fraction of data contained within a specified
        time window, with the option of introducing downweighting functions at
        the edges of the candidate window.

        :: INPUTS ::
        :param starttime: [None] or [obspy.UTCDateTime] starttime to pass
                            to obspy.Trace.trim. None input uses starttime
                            of this TraceBuffer
        :param endtime: [None] or [obspy.UTCDateTime] endtime to pass
                            to obspy.Trace.trim. None input uses endtime
                            of this TraceBuffer
                            NOTE: trim operations are conducted on a temporary
                            copy of this TraceBuffer and the trim uses pad=True
                            and fill_value=None to generate masked samples
                            if the specified start/endtimes lie outside the
                            bounds of data in this TraceBuffer
        :param wgt_taper_len: [float] non-negative, finite number of seconds
                            that a specified taper function should last on
                            each end of the candidate window. Value of 0
                            results in no taper applied.
        :param wgt_taper_type: [str] name of taper type to apply
                        Supported values:
                        "cosine" - apply a cosine taper to the data weighting
                                   function
                            aliases: 'cos', 'tukey'
                        "step" - set all data weights in the tapered 
                                 region to 0
                            aliases: 'h', 'heaviside'
        :: OUTPUT ::
        :return vf: [float] (weighted) fraction of non-masked data in the
                    candidate window. vf \in [0, 1]
        """

        tmp_copy = self.copy()
        tmp_copy.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=None)
        vf = tmp_copy.get_unmasked_fraction(
            wgt_taper_len=wgt_taper_len, wgt_taper_type=wgt_taper_type
        )
        return vf

    def get_unmasked_fraction(self, wgt_taper_len=0.0, wgt_taper_type="cosine"):
        """
        Get the fractional amount of unmasked data present in
        this TraceBuffer, or a specified time slice of the data.
        Options are provided to introduce a tapered down-weighting
        of data on either end of the buffer or specified slice of
        the data

        :: INPUTS ::
        :param wgt_taper_len: [float] tapered weighting function length
                        for each end of (subset) trace in seconds on either
                        end of the trace.
        :param wgt_taper_type: [str] type of taper to apply for data weighting.
                    Supported Values:
                    'cosine' - cosine taper (Default)
                        aliases: 'cos', 'tukey' (case insensitive)
                    'step' - Step functions centered on the nearest sample to
                            starttime + wgt_taper_len and endtime - wgt_taper_len
                        aliases: 'heaviside', 'h' (case insensitive)
        :: OUTPUT ::
        :return valid_frac: [float] value \in [0,1] indicating the (weighted)
                            fraction of valid (non-masked) samples in the
                            (subset)buffer

        NOTE: The 'step' wgt_taper_type is likely useful when assessing data windows
        that will be subject to blinding (e.g., continuous, striding predictions
        with ML workflows like EQTransformer and PhaseNet). The 'cosine' wgt_taper_type
        is likely more appropriate for general assessments of data when targeted
        features are assumed to be window-centered.
        """
        # Handle empty buffer case
        if self.stats.npts == 0:
            valid_frac = 1.0
        else:
            # Compatability check for wgt_taper_len
            wgt_taper_len = wuc.bounded_floatlike(
                wgt_taper_len,
                name="wgt_taper_len",
                minimum=0,
                maximum=self.stats.endtime - self.stats.starttime,
            )
            # Compatability check for wgt_taper_type
            if not isinstance(wgt_taper_type, str):
                raise TypeError("wgt_taper_type must be type str")
            elif wgt_taper_type.lower() in [
                "cosine",
                "cos",
                "tukey",
                "step",
                "h",
                "heaviside",
            ]:
                pass
            else:
                raise ValueError(
                    'wgt_taper_types supported: "cosine", "step", and select aliases'
                )

            # Taper generation section
            # Zero-length taper
            if wgt_taper_len == 0:
                tsamp = 0
                tap = 1
            # Non-zero-length taper
            else:
                tsamp = int(wgt_taper_len * self.stats.sampling_rate)
                # Cosine taper
                if wgt_taper_type.lower() in ["cosine", "cos", "tukey"]:
                    tap = 0.5 * (1.0 + np.cos(np.linspace(np.pi, 2.0 * np.pi, tsamp)))
                # Step taper
                elif wgt_taper_type.lower() in ["step", "h", "heaviside"]:
                    tap = np.zeros(tsamp)
            # Assess percent valid (valid_frac)
            # If unmasked data --> 100% completeness
            if not np.ma.is_masked(self.data):
                valid_frac = 1.0
            else:
                # Unmasked values = 1
                bin_array = np.ones(self.stats.npts)
                # Masked values = 0
                bin_array[self.data.mask] = 0
                # If no tapered weighting function
                if tsamp == 0:
                    valid_frac = np.mean(bin_array)
                # If tapered weighting function
                elif tsamp > 0:
                    # Compose weight array
                    wt_array = np.ones(self.stats.npts)
                    wt_array[:tsamp] *= tap
                    wt_array[-tsamp:] *= tap[::-1]
                    # Calculate weighted average
                    num = np.sum(wt_array * bin_array)
                    den = np.sum(wt_array)
                    valid_frac = num / den
        return valid_frac

    # ##################### #
    # Display class methods #
    # ##################### #

    def get_sncl(self):
        """
        Fetch string for
        Sta.Net.Chan.Loc
        """
        sta = self.stats.station
        net = self.stats.network
        cha = self.stats.channel
        loc = self.stats.location
        sncl = f"{sta}.{net}.{cha}.{loc}"
        return sncl

    def get_snil_c(self):
        """
        Fetch strings for tuple
        ('Sta.Net.Inst.Loc', 'Comp')
        """
        sta = self.stats.station
        net = self.stats.network
        cha = self.stats.channel
        inst = cha[:-1]
        comp = cha[-1]
        loc = self.stats.location
        snil = f"{sta}.{net}.{inst}.{loc}"
        return (snil, comp)

    def get_nsli_c(self):
        """
        Fetch strings for tuple
        ('Net.Sta.Loc.Inst', 'Comp')
        """
        sta = self.stats.station
        net = self.stats.network
        cha = self.stats.channel
        inst = cha[:-1]
        comp = cha[-1]
        loc = self.stats.location
        nsli = f"{net}.{sta}.{loc}.{inst}"
        return (nsli, comp)

    def __repr__(self, compact=False):
        """
        Return short summary string of the current TraceBuffer
        with options for displaying buffer status graphically

        :: INPUT ::
        :param compact: [bool] show a compact representation of buffer fill fraction
                        and masked fraction?
        :: OUTPUT ::
        :return rstr: [str] representative string
        """
        # If non-compact representation 
        if not compact:
            # Start Trace-inherited __str__ 
            rstr = f"{super().__str__()}"
            # Add buffer, mask, and max length metrics in pretty formatting
            rstr += f" | buffer {100.*(self.filled_fraction):.1f}%"
            rstr += f" | masked {100.*(1. - self.valid_fraction):.1f}%"
            rstr += f" | max {self.max_length:.1f} sec"
        # If compact representation (used with BufferTree)
        else:
            rstr = f"B:{self.filled_fraction:.1f}|M:{(1. - self.valid_fraction):.1f}"
        return rstr

    def __str__(self):
        """
        Return a repr string for this TraceBuffer
        """
        rstr = 'wyrm.buffer.trace.TraceBuffer('
        rstr += f'max_length={self.max_length}, fill_value={self.fill_value}, '
        rstr += f'method={self.method}, interpolation_samples={self.interpolation_samples})'
        return rstr

###################################################################################
# PREDICITON BUFFER CLASS DEFINITION ##############################################
###################################################################################

class PredictionBuffer(object):
    """
    Class definition for a PredictionBuffer object. 
    
    This class is structured loosely after an obspy.realtime.RtTrace, hosting
    a 2-D numpy.ndarray in the self.stack attribute, rather than a data vector. 
    The purpose of this buffer is to house l-labeled, t-time-sampled, windowed
    predictions from machine learning models and provide methods for stacking
    methods (i.e. overlap handling) used for these applications (i.e., max value and
    average value) that differ from the overlap handing provided in obspy (interpolation
    or gap generation - see obspy.core.trace.Trace.__add__).

    Like the obspy.realtime.RtTrace object, all data, and most metadata are populated
    in a PredictionBuffer object (pbuff, for short) by the first append of a
    pwind object (wyrm.core.window.prediction.PredicitonWindow) to the buffer.

    
    In this initial version:
     1) metadata are housed in distinct attributes, rather than an Attribute Dictionary 
        as is the case with obspy Trace-like objects (this may change in a future version)
     2) non-future appends (i.e., appends that back-fill data) have conditions relative to
        buffer size and data timing for if an append is viable. 
        APPEND RULES:
        a) If pwind is later than the contents of pbuff (i.e. future append) - unconditional append
        b) If pwind is earlier than the contents of pbuff - pbuff must have enough space to fit
            the data in pwind without truncating data at the future (right) end of the buffer
        c) If pwind is within the contents of pbuff - pbuff cannot have a data fold greater than
            1 for all samples wher pwind overlaps pbuff.

    """
    
    def __init__(self, max_samples=15000, stacking_method='max'):
        """
        Initialize a PredictionBuffer object that has predicted value
        arrays added to it throught the append method like TraceBuffer

        Unlike TraceBuffer, this class leverages the uniform sampling
        and timing of incoming values from assorted labels (channels)
        to accelerate stacking/buffer update operations via numpy's ufunc's

        :: INPUTS ::
        :param max_samples: [int] maximum number of samples the buffer can contain
                        Once data are appended, this corresponds to the buffered
                        data "stack" 1-axis (or j-axis)
        :param stacking_method: [str] stacking method to apply to overlapping data
                        'max': maximum value at the jth overlapping sample
                        'avg': mean value at the jth overlapping sample
        
        :: POPULATED ATTRIBUTES ::
        :attr max_samples: positive integer, see param max_samples
        :attr stacking_method: 'max', or 'avg', see stacking_method
        :attr fold: (max_samples, ) numpy.ndarray - keeps track of the number
                        of non-blinded samplesn stacked a particular time-point
                        in this buffer. dtype = numpy.float32
        :attr _has_data: PRIVATE - [bool] - flag indicating if data have been
                        appended to this pbuff via the pbuff.append() method
                        Default = False
        """
        # max_samples compat. check
        self.max_samples = wuc.bounded_intlike(
            max_samples,
            name='max_samples',
            minimum=1,
            maximum=None,
            inclusive=True
        )
        # stacking_method compat. check
        if stacking_method not in ['max', 'avg']:
            raise ValueError('stacking method must be either "max" or "avg"')
        else:
            self.stacking_method = stacking_method
        # Create fold vector
        self.fold = np.zeros(self.max_samples, dtype=np.float32)
        # Set _has_data flag
        self._has_data = False
    
    def validate_pwind(self, pwind):
        """
        Validate that metadata attributes in this PredictionBuffer and a candidate
        PredictionWindow. In the case where (meta) data have not been appended
        to this pbuff, attributes are assigned from the input pwind object
        (see :: ATTRIBUTES ::)

        :: INPUT ::
        :param pwind: [wyrm.core.window.prediction.PredictionWindow]
                    Candidate prediction window object
        :: OUTPUT ::
        :return status: [bool] is pwind compatable with this PredictionBuffer?

        :: ATTRIBUTES :: - only updated if self._has_data = False
        :attr id: [str] - instrument ID
        :attr t0: [float] - timestamp of first sample in pwind (seconds)
        :attr samprate: [float] - sampling rate of data in samples per second
        :attr model_name: [str] - machine learning model architecture name associated
                                with this pwind/pbuff
        :attr labels: [list] of [str] - names of data/prediction labels (axis-0 in self.stack)
        :attr stack: [numpy.ndarray] - (#labels, in_samples) array housing appended/stacked
                    data
        :attr blinding: [2-tuple] of [int] - number of blinding samples at either end
                     of a pwind
        """
        attr_list = ['id','samprate','model_name','weight_name','labels','blinding', 'npts']
        if self._has_data:
            bool_list = []
            for _attr in attr_list:
                bool_list.append(eval(f'self.{_attr} == pwind.{_attr}'))
            if all(bool_list):
                status = True
            else:
                status = False
        else:
            # Scrape metadata
            self.id = pwind.id
            self.t0 = pwind.t0
            self.samprate = pwind.samprate
            self.npts = pwind.npts
            self.model_name = pwind.model_name
            self.weight_name = pwind.weight_name
            self.labels = pwind.labels
            self.blinding = pwind.blinding
            # Populate stack
            self.stack = np.zeros(shape=(len(pwind.labels), self.max_samples), dtype=np.float32)
            status = True
        return status

    def append(self, pwind, include_blinding=False):
        """
        Append a PredictionWindow object to this PredictionBuffer if it passes compatability
        checks and shift scenario checks that:
            1) unconditionally allow appends of data that add to the right end of the buffer
                i.e., appending future data
            2) allows internal appends if there are some samples that have fold > 1 in the
                merge space
                i.e., appending to fill gaps
            3) allows appends to left end of buffer if the corresponding sample shifts
                would not truncate right-end samples
                i.e. appending past data

        :: INPUTS ::
        :param pwind: [wyrm.core.window.prediction.PredicitonWindow] pwind object with
                    compatable data/metadata with this pbuff
        :param include_blinding: [bool] should blinded samples be included when calculating
                    the candidate shift/stack operation?
        
        :: OUTPUT ::
        :return self: [wyrm.core.data.PredictionBuffer] enable cascading
        """
        # pwind compat. check
        if not isinstance(pwind, PredictionWindow):
            raise TypeError('pwind must be type wyrm.core.window.prediction.PredictionWindow')
        # pwind validation/metadata scrape if first append
        elif not self.validate_pwind(pwind):
            raise BufferError('pwind and this PredictionBuffer are incompatable')
        
        # include_blinding compat. check
        if not isinstance(include_blinding, bool):
            raise TypeError('include_blinding must be type bool')
        
        # Get stacking instructions
        indices = self.get_stacking_indices(pwind, include_blinding=include_blinding)
        # Initial append - unconditional
        if not self._has_data:
            self._shift_and_stack(pwind, indices)
            self._has_data = True
        # Future append - unconditional
        elif indices['npts_right'] < 0:
            self._shift_and_stack(pwind, include_blinding=include_blinding)
        # Past append - conditional
        elif indices['npts_right'] > 0:
            # Raise error if proposed append would clip the most current predictions
            if any(self.fold[-indices['npts_right']:] > 0):
                raise BufferError('Proposed append would trim off most current predictions in this buffer - canceling append')
            else:
                self._shift_and_stack(pwind, indices)
        # Internal append - conditional
        else: #if indices['npts_right'] == 0:
            # If all sample points have had more than one append, cancel
            if all(self.fold[indices['i0_s']:indices['i1_s']] > 1):
                raise BufferError('Proposed append would strictly stack on samples that already have 2+ predictions - canceling append')
            else:
                self._shift_and_stack(pwind, indices)
        return self
    
    def get_stacking_indices(self, pwind, include_blinding=True):
        """
        Fetch the time-sample shift necessary to fit pwind into this
        pbuff object (npts_right), the indices of data to include
        from this pwind object [i0_p:i1_p], and the indices that the
        data will be inserted into after the time-sample shift has
        been applied to self.stack/.fold [i0_s:i1_s].

        :: INPUTS ::
        :param pwind: [wyrm.core.window.prediction.PredictionWindow] pwind object
                        to append to this pbuff.
        :param include_blinding: [bool] should the blinded samples be included
                        when calculating the shift and indices for this proposed append?

        :: OUTPUT ::
        :return indices: [dict] dictionary containing calculated values for
                            npts_right, i0_p, i1_p, i0_s, and i1_s as desecribed
                            above.
        """
        # Set prediction window sampling indices
        if include_blinding:
            indices = {'i0_p': 0, 'i1_p': None}
        else:
            indices = {'i0_p': self.blinding,
                       'i1_p': -self.blinding}
        # If this is an initial append
        if not self._has_data:
            indices.update({'npts_right': 0, 'i0_s': None})
            if include_blinding:
                indices.update({'i1_s': self.npts})
            else:
                indices.update({'i1_s': self.npts - self.blinding - self.blinding})
        # If this is for a subsequen tappend
        else:
            dt = pwind.t0 - self.t0
            i0_init = dt*self.samprate
            # Sanity check that location is integer-valued
            if int(i0_init) != i0_init:
                raise ValueError('proposed new data samples are misaligned with integer sample time-indexing in this PredBuff')
            # Otherwise, ensure i0 is type int
            else:
                i0_init = int(i0_init)
            # Get index of last sample in candidate prediction window
            i1_init = i0_init + self.npts
            # If blinding samples are removed, adjust the indices
            if not include_blinding:
                i0_init += self.blinding
                i1_init -= self.blinding
                di = self.npts - self.blinding - self.blinding
            else:
                di = self.npts

            # Handle data being appended occurs before the current buffered data
            if i0_init < 0:
                # Instruct shift to place the start of pred at the start of the buffer
                indices.update({'npts_right': -i0_init,
                                     'i0_s': None,
                                     'i1_s': di})

            # If the end of pred would be after the end of the current buffer timing
            elif i1_init > self.max_samples:
                # Instruct shift to place the end of pred at the end of the buffer
                indices.update({'npts_right': self.max_samples - i1_init,
                                     'i0_s': -di,
                                     'i1_s': None})
            # If pred entirely fits into the current bounds of buffer timing
            else:
                # Instruct no shift and provide in-place indices for stacking pred into buffer
                indices.update({'npts_right': 0,
                                     'i0_s': i0_init,
                                     'i1_s': i1_init})

        return indices


    def _shift_and_stack(self, pwind, indices):
        """
        Apply specified npts_right shift to self.stack and self.fold and
        then stack in pwind.data at specified indices with specified
        self.stacking_method, and update self.fold. Internal shifting routine
        is wyrm.util.stacking.shift_trim()
        
        :: INPUTS ::
        :param pwind: [wyrm.core.window.prediction.PredictionWindow] 
                        validated prediction window object to append to this prediction buffer
        :param indices: [dict] - stacking index instructions from self.get_stacking_indices()
        
        :: OUTPUT ::
        :return self: [wyrm.core.buffer.prediction.PredictionBuffer] to enable cascading
        """

        # Shift stack along 1-axis
        self.stack = shift_trim(
            self.stack,
            indices['npts_right'],
            axis=1,
            fill_value=0.,
            dtype=self.stack.dtype)
        
        # Shift fold along 0-axis
        self.fold = shift_trim(
            self.fold,
            indices['npts_right'],
            axis=0,
            fill_value=0.,
            dtype=self.fold.dtype)
        
        # # ufunc-facilitated stacking # #
        # Construct in-place prediction slice array
        pred = np.zeros(self.stack.shape, dtype=self.stack.dtype)
        pred[:, indices['i0_s']:indices['i1_s']] = pwind.data[:, indices['i0_p']:indices['i1_p']]
        # Construct in-place fold update array
        nfold = np.zeros(shape=self.fold.shape, dtype=self.stack.dtype)
        nfold[indices['i0_s']:indices['i1_s']] += 1
        # Use fmax to update
        if self.stacking_method == 'max':
            # Get max value for each overlapping sample
            np.fmax(self.stack, pred, out=self.stack); #<- Run quiet
            # Update fold
            np.add(self.fold, nfold, out=self.fold); #<- Run quiet
        elif self.stacking_method == 'avg':
            # Add fold-scaled stack/prediction arrays
            np.add(self.stack*self.fold, pred*nfold, out=self.stack); #<- Run quiet
            # Update fold
            np.add(self.fold, nfold, out=self.fold); #<- Run quiet
            # Normalize by new fold to remove initial fold-rescaling
            np.divide(self.stack, self.fold, out=self.stack, where=self.fold > 0); #<- Run quiet
        
        # If a shift was applied, update t0 <- NOTE: this was missing in version 1!
        if indices['npts_right'] != 0:
            self.t0 -= indices['npts_right']/self.samprate


    def merge(self, other):
        if not isinstance(other, PredictionBuffer):
            raise TypeError('other must be type wyrm.core.data.PredictionBuffer')

        if self.samprate != other.samprate:
            raise AttributeError('sampling rates mismatch between self and other')
        
        if self.id != other.id:
            raise AttributeError('id mismatch between self and other')

        if self.model_name != other.model_name:
            raise AttributeError('model_name mismatch')
        
        if self.weight_name != other.weight_name:
            raise AttributeError('weight_name mismatch')
        
        if self.labels != other.labels:
            raise AttributeError('labels mismatch')
        
        dt = other.t0 - self.t0
        dn = dt/self.samprate
        if int(dn) != dn:
            raise ValueError('time sampling is misaligned')
        

        # if self.max_length != other.max_length:
        #     raise AttributeError('max_length mismatch between self and others')

        # if self.t0 - other.t0// self.samprate
        # if 


    def __iadd__(self, other):
        if not isinstance(other, (PredictionBuffer, PredictionWindow)):
            raise TypeError('other must be type wyrm.core.data.PredictionBuffer')
        elif isinstance(other, PredictionWindow):
            self.append(other)
        elif isinstance(other, PredictionBuffer):
            self.merge(other)

    # I/O AND DUPLICATION METHODS #


        
        return self

    def copy(self):
        """
        Return a deepcopy of this PredictionBuffer
        """
        return deepcopy(self)
    


    def to_stream(self, min_fold=1, fill_value=None):
        """
        Create a representation of self.stack as a set of labeled
        obspy.core.trace.Trace objects contained in an obspy.core.stream.Stream
        object. Traces are the full length of self.stack[_i, :] and masked using
        a threshold value for self.fold.

        Data labels are formatted as follows
        for _l in self.labels
            NLSC = self.id + _l[0].upper()
        
        i.e., the last character in the NSLC station ID code is the first, capitalized
              letter of the data label. 
        e.g., for UW.GNW.--.BHZ, 
            self.id is generally trucated by the RingWyrm or DiskWyrm data ingestion
            to UW.GNW.--.BH
            UW.GNW.--.BH with P-onset prediction becomes -> UW.GNW.--.BHP

        The rationale of this transform is to allow user-friendly interface with the 
        ObsPy API for tasks such as thresholded detection and I/O in standard seismic formats.

        :: INPUTS ::
        :param min_fold: [int-like] minimum fold required for a sample in stack to be unmasked
        :param fill_value: compatable value for numpy.ma.masked_array()'s fill_value kwarg

        :: OUTPUT ::
        :return st: [obspy.core.stream.Stream]
        """
        # Create stream
        st = Stream()
        # Compose boolean mask
        mask = self.fold < min_fold
        # Use default fill value
        if fill_value is None:
            fill_value = self.fill_value
        # Compose generic header
        n,s,l,bi = self.id.split('.')
        # If the ID happens to be using the full 3-character SEED code, truncate
        if len(bi) == 3:
            bi = bi[:2]
        header = {'network': n,
                  'station': s,
                  'location': l,
                  'starttime': self.t0,
                  'sampling_rate': self.samprate}
        # Construct specific traces with label names
        for _i, _l in enumerate(self.labels):
            header.update({'channel':f'{bi}{_l[0].upper()}'})
            _tr = Trace(data=np.ma.masked_array(data=self.stack[_i,:],
                                                mask=mask, fill_value=fill_value),
                        header=header)
            st.append(_tr)
        
        # Construct fold trace
        header.update({'channel': f'{bi}f'})
        _tr = Trace(data=self.fold, header=header)
        st.append(_tr)
        return st
    

    def to_mseed(self, save_dir='.', min_fold=1, fill_value=None):
        """
        EXAMPLE WORKFLOW

        This serves as an example wrapper around to_stream() for 
        saving outputs to miniSEED format as individual traces

        :: INPUTS ::
        :param save_dir: [str] - path to directory in which to save
                        the output miniSEED files (exclude trailing \ or /)
        :param min_fold: [int] - see self.to_stream()
        :param fill_value: - see self.to_stream()
        """

        st = self.to_stream(min_fold=min_fold, fill_value=fill_value)
        labels = ''
        for _l in self.labels:
            labels += f'{_l}_'
        labels = labels[:-1]
        of_str = '{model}_{weight}_{id}_{t0:.6f}_{labels}.mseed'.format(
            model=self.model_name,
            weight=self.weight_name,
            id=self.id,
            t0=self.t0,
            labels=labels
        )
        out_fp = os.path.join(save_dir, of_str)
        st.write(out_fp, fmt='MSEED')

    def from_stream(self, st, model_name=None, weight_name=None, tol=1e-5):
        """
        Populate a PredictionBuffer object from a stream object 
        """
        # Only run "append" from stream if no data are present
        if self._has_data:
            raise BufferError('This PredictionBuffer already has data - cannot populate from Stream')

        # Convert labels into channel names
        channels = [_tr.stats.channel for _tr in st]
        labels = [_tr.stats.channel[-1] for _tr in st]
        
        # Run cross-checks on data traces and fold trace
        for _i, tr_i in enumerate(st):
            for _j, tr_j in enumerate(st):
                if _i < _j:
                    if abs(tr_i.stats.starttime - tr_j.stats.starttime) > tol:
                        raise AttributeError(f'starttime mismatch: {tr_i.id} vs {tr_j.id}')
                    if tr_i.stats.npts != tr_j.stats.npts:
                        raise AttributeError(f'npts mismatch: {tr_i.id} vs {tr_j.id}')
                    if tr_i.stats.sampling_rate != tr_j.stats.sampling_rate:
                        raise AttributeError(f'sampling_rate mismatch: {tr_i.id} vs {tr_j.id}')
                    if tr_i.id[-1] != tr_j.id[:-1]:
                        raise AttributeError(f'instrument code (id[:-1]) mismatch: {tr_i.id} vs {tr_j.id}')
                    
        # populate attributes
        self.t0 = st[0].stats.starttime.timestamp
        self.labels = labels
        self.id = st[0].id[:-1]
        self.weight_name = weight_name
        self.model_name = model_name
        self._has_data = True

        # populate stack
        self.stack = np.zeros(shape=(len(labels), st[0].stats.npts), dtype=np.float32)
        # get data
        _i = 0
        for _c in (channels): 
            _tr = st.select(channel=_c)[0]
            if _c[-1] != 'f':
                self.stack[_i, :] = _tr.data
                _i += 1
            else:
                self.fold = _tr.data
        
        return self

    def from_mseed(self, infile ,tol=1e-5):
        """
        Populate this PredictionBuffer from a version saved as a MSEED file using
        the to_mseed() method above. This can only be executed on a pbuff object that
        has not had data appended to it.

        Wraps the PredictionBuffer.from_stream() method.

        :: INPUTS ::
        :param infile: [str] path and file name string - formatting must conform to the 
                     format set by PredictionBuffer.to_mseed()
        :param tol: [float] mismatch tolerance between trace starttimes in seconds

        :: OUTPUT ::
        :return self: [wyrm.core.buffer.prediction.PredictionBuffer] enable cascading
        """
        # Only run "append" from stream if no data are present
        if self._has_data:
            raise BufferError('This PredictionBuffer already has data - cannot populate from mSEED')

        # Parse input file name for metadata
        # Strip filename from infile
        _, file = os.path.split(infile)
        # Remove extension
        file = os.path.splitext(file)[0]
        # Split by _ delimiter
        fparts = file.split('_')
        if len(fparts) != 4:
            raise SyntaxError(f'infile filename {file} does not conform with the format "model_weight_id_t0_l1-l2-l3"')
        # Alias metadata from file name
        model = fparts[0]
        weight = fparts[1]
        # id = fparts[2]
        # ch = id.split('.')[-1]
        # t0 = float(fparts[3])
        # labels = fparts[4].split('-')
        
        # Load data
        st = read(infile).merge()
        # Run data ingestion
        self.from_stream(st, model_name=model, weight_name=weight, tol=tol)
        
        return self

    def __repr__(self, compact=False):
        """
        Return a short summary string of the current PredictionBuffer
        status with options for one-line compact representation in
        BufferTree.__repr__()
        
        :: INPUT ::
        :param compact: [bool] should the PredictionBuffer be
        """
        # If running as compact - show fraction of data fold
        if compact:
            try: 
                # Get Condensed Fold Brackets
                ffold0 = sum(self.fold == 0)/self.max_samples
                ffold1 = sum(self.fold == 1)/self.max_samples
                rstr = f'PBfold|0: {ffold0:.2f}|1: {ffold1:.2f}|'
                # Get greater than 1 fold fraction if max fold exceeds 1
                if self.fold.max() > 1:
                    ffoldn = 1 - ffold0 - ffold1
                    rstr += f'2+: {ffoldn:.2f}|'
            except AttributeError:
                rstr = f'PB | init | Max: {self.max_samples} | Stack: {self.stacking_method}'
        else:
            try:
                rstr = f'PredictionBuffer | Model: {self.model_name} | Weight: {self.weight_name} | Stacking: {self.stacking_method}\n'
                rstr += f'{self.id:16} | {UTCDateTime(self.t0)} - {UTCDateTime(self.t0 + self.max_samples/self.samprate)}\n'
                rstr += f'Labels: {", ".join(self.labels)} | Window: {self.npts} | Buffer: {self.max_samples} | S/R: {self.samprate} Hz\n'
                rstr += 'Stack Fold'
                for _f in range(int(self.fold.max())+1):
                    pfold = 100*sum(self.fold == _f)/self.max_samples
                    rstr += f' -> {int(_f)}: {pfold:.2f}%'
                rstr += f' | dtype: {self.stack.dtype}'
            except AttributeError:
                rstr = f'Prediction Buffer | initialized | Buffer: {self.max_samples} | Stacking: {self.stacking_method}'
        return rstr
    
    def __str__(self):
        rstr = f'wyrm.core.data.PredictionBuffer(max_samples={self.max_samples}, stacking_method="{self.stacking_method}")'
        try:
            xstr = f'.append(<PredictionWindow({self.id}, {self.model_name}, {self.weight_name})>)'
        except AttributeError:
            xstr = ''
        rstr += xstr
        return rstr


###################################################################################
# BUFFER TREE CLASS DEFINITION ####################################################
###################################################################################
class BufferTree(dict):
    """
    Standardized structure and search methods for an 3-tiered dictionary with 
    the structure
        "limb" tier0 - a primary key-set, generally site code string
            "branch" tier1 - a secondary key-set, generally band and instrument code string
                "twig" tier2 - a thrid key-set, component character for Traces, model weight for Predictions
                    "bud" buff - a data holding object with an obj.append() method

    The rationale of this structure is to accelerate searching by limiting the number
    of keys that are searched in a given list of keys, and leveraging the query speed
    of hash-tables underlying the Python `dict` class. Earlier tests showed that this
    structured system was significantly faster at retrieving data from large sets of
    objects compared to an unstructured list-like holder.

    """
    def __init__(self, buff_class=TraceBuffer, **buff_init_kwargs):
        # Inherit dict properties
        super().__init__()
        # Compatability check for buff_class input
        if not isinstance(buff_class, type):
            raise TypeError('buff_class must be a class-defining type')
        # Ensure that buff_class has an append method
        elif 'append' not in dir(buff_class):
            raise TypeError('buff_class does not have an "append" method - incompatable')
        else:
            self.buff_class = buff_class

        # Compatability check for buff_init_kwargs input
        bcargs = inspect.getfullargspec(self.buff_class).args[1:]
        bcdefs = inspect.getfullargspec(self.buff_class).defaults
        bdefaults = dict(zip(bcargs, bcdefs))
        emsg = ''
        for _k in buff_init_kwargs.keys():
            if _k not in bcargs:
                if emsg == '':
                    emsg += f'The following kwargs are not compabable with {buff_class}:'
                emsg += f'\n{_k}'
        # If emsg has any information populated in it, kick error with emsg report
        if emsg != '':
            raise TypeError(emsg)
        # Otherwise clear init_kwargs
        else:
            for _k in bcargs:
                if _k not in buff_init_kwargs and _k != 'self':
                    buff_init_kwargs.update({_k: bdefaults[_k]})
            self.bkwargs = buff_init_kwargs
        # Create a template buffer (some methods have larger overhead, so making a template saves time)
        self._template_buff = self.buff_class(**self.bkwargs)
        self.nt0b = 0
        self.nt1b = 0
        self.nt2b = 0

    def update_counts(self):
        self.nt0b = len(self.keys())
        nt1b = 0
        nt2b = 0
        for k0 in self.keys():
            nt1b += len(self[k0].keys())
            for k1 in self[k0].keys():
                nt2b += len(self[k0][k1].keys())
        self.nt1b = nt1b
        self.nt2b = nt2b

    def copy(self):
        """
        Return a deepcopy of this wyrm.buffer.structures.BufferTree
        """
        return deepcopy(self)
    
    def is_in_tree(self, tk0, tk1=None, tk2=None):
        if tk0 in self.keys():
            if tk1 is not None:
                if tk1 in self[tk0].keys():
                    if tk2 is not None:
                        if tk2 in self[tk0][tk1].keys():
                            return True
                        else:
                            return False
                    else:
                        return True
                else:
                    return False
            else:
                return True
        else:
            return False

    def grow_tree(self, TK0, TK1=None, TK2=None):
        """
        Add a new limb/branch/twig/bud structure to this tiered buffer 
        with specified tier-0, tier-1, and tier-2 keys and the specified
        "bud" buff_class

        :: INPUTS ::
        :param TK0: [str] tier-0 key for candidate branch
        :param TK1: [str] or None
                    tier-1 key for candidate branch
                    If None, an empty branch keyed to TK0 is
                    added if tier-0 key TK0 does not already exist
        :param TK2: [str] or None
                    tier-2 key for candidate branch
                    If None, an empty branch/limb keyed to TK0/TK1 is
                    added if it does not already exist
        
        :: OUTPUT ::
        :return self: [BufferTree] representation of self to enable cascading
        """
        # If limb doesn't exist
        if TK0 not in self.keys():
            # If tier-1 key is provided
            if TK1 is not None:
                # If tier-2 key is provided
                if TK2 is not None:
                    self.update({TK0: {TK1: {TK2: self._template_buff.copy()}}})
                    self.nt0b += 1
                    self.nt1b += 1
                    self.nt2b += 1
                # If tier-2 key is None
                else:
                    self.update({TK0: {TK1: {}}})
                    self.nt0b += 1
                    self.nt1b += 1
            # If tier-1 key is None
            else:
                self.update({TK0: {}})
                self.nt0b += 1
        # If limb exists
        else:
            # If branch does not exist
            if TK1 is not None:
                # If tier-1 key is provided and the branch doesn't exist
                if TK1 not in self[TK0].keys():
                    # If tier-2 key is provided 
                    if TK2 is not None:
                        self[TK0].update({TK1: {TK2: self._template_buff.copy()}})
                        self.nt1b += 1
                        self.nt2b += 1
                    # If tier-2 key is None
                    else:
                        self[TK0].update({TK1: {}})
                        self.nt1b += 1
                # If this limb/branch already exists
                else:
                    # If tier-2 key is provided
                    if TK2 is not None:
                        # If twig doesn't exist yet
                        if TK2 not in self[TK0][TK1].keys():
                            self[TK0][TK1].update({TK2: self._template_buff.copy()})
                            self.nt2b += 1
        self.update_counts()
        return self

    def apply_buffer_method(self, TK0='', TK1='', TK2='', method='__repr__', **inputs):
        """
        Apply specified buff_class class method to entries that match (regex) strings
        for tier-0 and tier-1 keys as self[TK0][TK1].method(**inputs)

        NOTE: Operations are applied to data contained within each buff_class object as
                specified by the method and are expected to act on data in-place. 

                e.g., For buff_class = TraceBuff
                self.apply_buffer_method(method='filter', TK0='UW.*.EH', TK1='Z', type='bandpass', freqmin=1, freqmax=45)
                    would apply a 1-45 Hz Butterworth filter to analog, 1-component seismometers in the UW network


        :: INPUTS ::
        :param method: [str] string representation of a valid classmethod
                        for self.buff_class objects
                        default: "__repr__"
        :param TK0: [str] (regex-compatable) string to search for among tier-0 keys
        :param TK1: [str] (regex-compatable) string to search for among tier-1 keys
                            belonging to matched tier-0 key branches
        :param **inputs: [kwargs] inputs for `method`. Any arg-type inputs should be
                            presented as kwargs to ensure correct assignment
        
        :: OUTPUT ::
        :return self: [BufferTree] representation of self to enable cascading
        """
        # Check that method is compatable with buff_class
        if method not in dir(self.buff_class):
            raise SyntaxError(f'specified "method" {method} is not compatable with buffer type {self.buff_class}')
        # Conduct search on tier-0 keys
        for _k0 in fnmatch.filter(self.keys(), TK0):
            # Conduct search on tier-1 keys
            for _k1 in fnmatch.filter(self[_k0].keys(), TK1):
                for _k2 in fnmatch.filter(self[_k0][_k1].keys(), TK2):
                    # Compose eval str to enable inserting method
                    eval_str = f'self[_k0][_k1][_k2].{method}(**inputs)'
                    eval(eval_str); # <- run silent
        return self

    def append(self, obj, TK0, TK1, TK2, **options):
        """
        Append a single object to a single buffer object in this BufferTree using
        the buffer_object.append() method

        :: INPUTS ::
        :param obj: [object] object to append to a self.buff_class object
            Must be compatable with the self.buff_class.append() method
        :param TK0: [str] key string for tier-0 associated with `obj`
        :param TK1: [str] key string for tier-1 associated with `obj`

        :param **options: [kwargs] key-word arguments to pass to 
                    self.buff_class.append()
        
        :: OUTPUT ::
        :return self: [wyrm.buffer.structures.BufferTree] return
                    alias of self to enable cascading.
        """
        # Add new structural elements a needed
        self.grow_tree(TK0, TK1=TK1, TK2=TK2)
        # Append obj to buffer [TK0][TK1]
        self[TK0][TK1][TK2].append(obj, **options)
        # Update indices
        self.update_counts()
        # Return self
        return self
    
    def __str__(self):
        """
        Provide a representative string documenting the initialization of
        this BufferTree
        """
        rstr = f'wyrm.buffer.structures.BufferTree(buff_class={self.buff_class}'
        for _k, _v in self.bkwargs.items():
            rstr += f', {_k}={_v}'
        rstr += ')'
        return rstr
    
    def _repr_line(self, _k0, _k1):
        """
        Compose a single line string that represents contnts of 
        branch [_k0]
        """
        bistr =  f'{_k0:13}| {_k1:2} '
        rstr = f'{bistr}'
        for _k2 in self[_k0][_k1].keys():
            rstr += f'| [{_k2}] {self[_k0][_k1][_k2].__repr__(compact=True)} '
        return rstr 
    
    def __repr__(self, extended=False):
        """
        String representation of the contents of this BufferTree
        similar to the format of obspy.core.stream.Stream objects

        :: INPUT ::
        :param extended: [bool] Show all tier-0 entries and their
                        compact representation of tier-1 labels and
                        buffer contents?

        :: OUTPUT ::
        :return rstr: [str] representative string
        """
        _cl = 3
        # Create header line
        rstr = f'BufferTree with {self.nt0b} limbs | {self.nt1b} branches\n'
        rstr += f'holding {self.nt2b} {self.buff_class} buffers\n'
        # If extended, iterate across all tier-0 keys
        if extended:
            for _k0 in self.keys():
                for _k1 in self[_k0].keys():
                    # And print tier-1 keys plus compact buffer representations
                    rstr += self._repr_line(_k0, _k1)
                    rstr += '\n'
        # If not extended produce a similar trucated format as 
        # obspy.core.stream.Stream.__str__(extended=False)
        else:
            # Iterate across tier-0 keys
            _i = 0
            for _k0 in self.keys():
                for _k1 in self[_k0].keys():
                    # If the first 2 or last 2 lines
                    if _i < _cl or _i > self.nt1b - _cl - 1:
                        rstr += self._repr_line(_k0, _k1)
                        rstr += '\n'
                    elif _i in [_cl, len(self.keys()) - _cl - 1]:
                        rstr == '    ...    \n'
                    
                    elif _i == _cl + 1:
                        rstr += f' ({self.nt2b - _cl*2} other branches)\n'
                    
                    _i += 1
            if not extended and len(self) > _cl*2:
                rstr += 'To see full contents, use print(BufferTree.__repr__(extended=True))'
        return rstr



    def __iadd__(self, other):
        if not isinstance(other, BufferTree):
            raise TypeError('other must be type wyrm.core.data.BufferTree')
        elif self.buff_class != other.buff_class:
            raise AttributeError(f'other.buff_class must be the same as this BufferTree: type {self.buff_class}')
        else:
            # Iterate across other's tier-0 keys
            for _k0 in other.keys():
                # If limb in other DNE in self, append whole limb
                if _k0 not in self.keys():
                    self.update({_k0: other[_k0]})
                    continue
                # Otherwise cross-check branches
                else:
                    pass
            
                for _k1 in other[_k0].keys():
                    # If branch in other DNE in self, append branch
                    if _k1 not in self[_k0].keys():
                        self[_k0].update({_k1: other[_k0][_k1]})
                        continue
                    else:
                        pass
                    for _k2 in other[_k0][_k1].keys():
                        # If twig in other DNE in self, append twig
                        if _k2 not in self[_k0][_k1].keys():
                            self[_k0][_k1].update({_k2: other[_k0][_k1][_k2]})
                        else:
                            # Run iadd on buff_class to try append
                            try:
                                self[_k0][_k1][_k2] += other[_k0][_k1][_k2]
                            except TypeError:
                                pass
            self.update_counts()
                        # END TWIG LOOP
                    # END BRANCH LOOP
            # END LIMB LOOP
                            
            return self
                                
    def append_stream(self, stream):
        """
        Convenience method for appending the contents of obspy.core.stream.Stream to
        a BufferTree with buff_class=wyrm.buffer.trace.TraceBuff, using the 
        following key designations

        TK0 = trace.id[:-1]
        TK1 = trace.id[-1]

        :: INPUT ::
        :param stream: [obspy.core.stream.Stream] or [obspy.core.trace.Trace]
                Trace or Stream to append to this BufferTree
        :: OUTPUT ::
        :return self: [wyrm.buffer.structures.BufferTree] allows cascading
        """
        # Ensure the buff_class is TraceBuffer
        if self.buff_class != TraceBuffer:
            raise AttributeError('Can only use this method when buff_class is wyrm.buffer.trace.TraceBuff')
        # Ensure the input is an obspy Trace or Stream
        if not isinstance(stream, (Trace, Stream)):
            raise TypeError
        # If this is a trace, turn it into a 1-element list
        if isinstance(stream, Trace):
            stream = [stream]
        # Iterate across traces in stream
        for _tr in stream:
            n,s,l,c = _tr.id.split('.')
            # Generate keys
            tk0 = '.'.join([n,s,l])
            tk1 = c[:-1]
            tk2 = c[-1]
            # Append to TraceBuff, generating branches as needed
            self.append(_tr, TK0=tk0, TK1=tk1, TK2=tk2)
        return self

    def append_pwind(self, pwind):
        """
        Convenience method for appending a wyrm.data.PredictionWindow (pwind) 
        to this BufferTree using pre-defined key formulations
            tier-0: Station Code
            tier-1: Instrument Code (Band & Instrument from SEED channel naming conventions)
            tier-2: Model weight
        """
        if self.buff_class != PredictionBuffer:
            raise AttributeError('Can only use this method when buff_class is wyrm.data.PredictionBuffer')
        if not isinstance(pwind, PredictionWindow):
            raise TypeError('pwind must be type wyrm.data.PredictionWindow')
        n,s,l,inst = pwind.id.split('.')
        tk0 = '.'.join([n,s,l])
        tk1 = inst
        tk2 = pwind.weight_name
        self.append(pwind, TK0=tk0, TK1=tk1, TK2=tk2)
        return self
    


