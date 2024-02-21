"""
:module: wyrm.core.window.instrument
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains the class definition for an InstrumentWindow, which
    serves as a data-handling class that bridges between continuous trace data
    and windowed, pre-processed waveform data.

    Location in the data chain
    TraceBuffer -> *InstrumentWindow* -> PredictionWindow
"""

from obspy import Trace, Stream, UTCDateTime
from wyrm.core.buffer.trace import TraceBuffer
from wyrm.core.window.prediction import PredictionWindow
import wyrm.util.compatability as wcc
import seisbench.models as sbm
import numpy as np
import torch
from copy import deepcopy

class InstrumentWindow(Stream):

    def __init__(
            self,
            Ztr,
            Ntr=None,
            Etr=None,
            target_starttime=None,
            target_sr=100,
            model_name='EQTransformer',
            target_npts=6000,
            target_order='ZNE',
            target_blinding=(500,500),
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
            self.t_starttime = self.traces[0].stats.starttime
        elif isinstance(target_starttime, UTCDateTime):
            self.t_starttime = target_starttime
        elif isinstance(target_starttime, float):
            self.t_starttime = UTCDateTime(target_starttime)
        else:
            raise TypeError('target_starttime must be type float, UTCDateTime, or NoneType')
        # target sampling rate compat check
        self.t_sr = wcc.bounded_floatlike(
            target_sr,
            name='target_sr',
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
        self.t_npts = wcc.bounded_intlike(
            target_npts,
            name='target_npts',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # target order compat check
        if not isinstance(target_order, str):
            raise TypeError('target_order must be type str')
        if not all((_c.upper() in 'ZNE' for _c in target_order)):
            raise ValueError('not all characters in target_order are supported')
        else:
            self.t_order = ''.join([_c.upper() for _c in target_order])
        # target blinding compat check
        if not isinstance(target_blinding, (list, tuple)):
            raise TypeError('target_blinding must be a list-like object')
        elif len(target_blinding) != 2:
            raise ValueError('target_blinding must be a 2-element list-like object')
        else:
            lblnd = wcc.bounded_intlike(target_blinding[0],
                                        name='target_blinding[0]',
                                        minimum=0,
                                        maximum=self.t_npts/2,
                                        inclusive=True)
            rblnd = wcc.bounded_intlike(target_blinding[1],
                                        name='target_blinding[1]',
                                        minimum=0,
                                        maximum=self.t_npts/2)
            self.t_blinding = (lblnd, rblnd)
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
            self.processing.append({'apply_missing_component_rule': ('clone', self.Z)})
        else:
            self.processing.append({'apply_missing_component_rule': ('clone', False)})
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
            self._apply_clonez_mcr()
        elif self.N is None:
            self.N = self.E
            self.processing.append({'apply_missing_component_rule': ('clone', self.E)})
        elif self.E is None:
            self.E = self.N
            self.processing.append({'apply_missing_component_rule': ('clone', self.N)})
        else:
            self.processing.append({'apply_missing_component_rule': ('clone', False)})
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
                    if _tr.stats.starttime != self.t_starttime:
                        # Get trace initial starttime
                        trt0 = _tr.stats.starttime
                        # Get delta time between target starttime and current starttime
                        dt = self.t_starttime - _tr.stats.starttime
                        # Convert to delta samles
                        dn = dt/self.t_sr
                        # If samples are misaligned with target sampling - apply sync
                        if dn != int(dn):
                            # Front pad with the leading sample value
                            _tr.trim(
                                starttime=self.t_starttime - _tr.stats.delta,
                                pad=True,
                                fill_value=_tr.data[0]
                            )
                            # Sync time using interpolate
                            _tr.interpolate(
                                _tr.stats.sampling_rate,
                                starttime = self.t_starttime
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
        ts = self.t_starttime
        te = ts + (self.t_npts - 1) / self.t_sr
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
            if any(_tr.stats.npts != self.t_npts for _tr in self.traces):
                raise UserWarning(f'Not all trimmed traces in this InstrumentWindow meet target_npts {self.t_npts}')
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
    def to_stream(self, ascopy=False):
        """
        Create a stream representation of this InstrumentStream with an option
        to make the view a deepcopy of the source data (default is False)
        """
        st = Stream()
        for _c in self.t_order:
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
        # Run Checks
        for _tr in self.traces:
            # All starttimes must be target starttimes
            if _tr.stats.starttime != self.t_starttime:
                raise AttributeError('not all traces have target starttimes')
            # All endtimes must be target endtimes
            if _tr.stats.endtime != self.t_starttime + self.t_npts/self.t_sr:
                raise AttributeError('not all traces have target endtimes')
            # All sampling_rates must be targe sampling rates
            if _tr.stats.sampling_rate != self.t_sr:
                raise AttributeError('not all traces have target sampling_rate')
            # All npts match target
            if _tr.stats.npts != self.t_npts:
                raise AttributeError('not all traces have target npts')
            # All traces have gaps filled
            if np.ma.is_masked(_tr.data):
                raise AttributeError('not all traces have had their gaps filled')
        # Compose holder array if 
        data = np.fill(shape=(3, self.t_npts), fill_value=np.nan, dtype=np.float32)
        if all(_c.upper() in 'ZNE' for _c in other_order):
            order = other_order.upper()
        else:
            order = self.t_order

        for _i, _c in enumerate(order):
            data[_i, :] = self.fetch_with_alias(_c, ascopy=ascopy).data
        return data
            
    def to_pwind(self, ascopy=False):
        """
        Return processed data and metadata as a
        wyrm.core.window.prediction.PredictionWindow object
        with the option to pass data as a copy or alias of 
        data held within this InstrumentWindow.

        :: INPUT ::
        :param ascopy: [bool] Should data be transferred as a copy?
                        default False
        
        :: OUTPUT ::
        :return pwind: [wyrm.core.window.prediction.PredictionWindow]
        """
        kwargs = {'model_name': self.model_name,
                  'data': self.to_numpy(ascopy=ascopy),
                  'id': self.Z[:-1],
                  'samprate': self.Zdata.stats.samprate,
                  't0': self.Zdata.stats.starttime.timestamp,
                  'blinding': self.t_blinding,
                  'labels': self.t_order,
                  'in_samples': self.t_npts}
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
        :return pwind: [wyrm.core.window.prediction.PredictionWindow]

        Processing Steps:
            Handle masked data
                split_window()
                filter()
                detrend('demean')
                detrend('lienar')
            Hit target sampling
                resample(target_sr)
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
        self.detrend('linaer')
        # Resample data
        self.resample(self.t_sr)
        self.processing.append(f'obspy resample({self._tsr})')
        # Taper data
        self.taper(None, max_length=taper_sec, side="both")
        self.processing.append(f'obspy taper(None, max_length={taper_sec}, side="both")')
        self.merge_window()
        self.sync_window_timing()
        self.trim_window()
        self.fill_window_gaps(fill_value=0)
        self.normalize_window()
        pwind = self.to_pwind(ascopy=ascopy)
        return pwind
        








