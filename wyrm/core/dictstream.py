import fnmatch, inspect
import numpy as np
import pandas as pd
import wyrm.util.compatability as wuc
from decorator import decorator
from copy import deepcopy
from obspy import Stream, UTCDateTime
from obspy.core.trace import Trace, Stats
from obspy.core.compatibility import round_away
from obspy.core.util.attribdict import AttribDict
from tqdm import tqdm
from wyrm.core.trace import MLTrace

@decorator
def _add_processing_info(func, *args, **kwargs):
    """
    This is a decorator that attaches information about a processing call as a string
    to the DictStream.stats.processing lists

    Attribution: Directly adapted from the obspy.core.trace function of the same name.
    """
    callargs = inspect.getcallargs(func, *args, **kwargs)
    callargs.pop("self")
    kwargs_ = callargs.pop("kwargs", {})
    info = "Wyrm 0.0.0: {function}(%s)".format(function=func.__name__)
    arguments = []
    arguments += \
        ["%s=%s" % (k, repr(v)) if not isinstance(v, str) else
         "%s='%s'" % (k, v) for k, v in callargs.items()]
    arguments += \
        ["%s=%s" % (k, repr(v)) if not isinstance(v, str) else
         "%s='%s'" % (k, v) for k, v in kwargs_.items()]
    arguments.sort()
    info = info % "::".join(arguments)
    self = args[0]
    result = func(*args, **kwargs)
    # Attach after executing the function to avoid having it attached
    # while the operation failed.
    self._internal_add_processing_info(info)
    return result

class DictStreamHeader(AttribDict):
    defaults = {
        'common_id': '',
        'ref_starttime': None,
        'ref_sampling_rate': None,
        'ref_npts': None,
        'ref_model': None,
        'ref_weight': None,
        'min_starttime': None,
        'max_starttime': None,
        'min_endtime': None,
        'max_endtime': None,
        'sync_status': {'starttime': False, 'sampling_rate': False, 'npts': False},
        'processing': []
    }

    def __init__(self, header={}):
        super(DictStreamHeader, self).__init__(header)
    
    def __repr__(self):
        rstr = '----Stats----'
        for _k, _v in self.items():
            if _v is not None:
                if _k == 'sync_status':
                    rstr += f'\n{_k:>18}: '
                    for _k2, _v2 in _v.items():
                        rstr += f'\n{_k2:>26}: {_v2}'
                elif 'starttime' in _k:
                    if _k == 'min_starttime':
                        rstr += f'\n{"starttime range":>18}: {self[_k]} - {self.max_starttime}'
                    
                elif 'endtime' in _k:
                    if _k == 'min_endtime':
                        rstr += f'\n{"endtime range":>18}: {self[_k]} - {self.max_endtime}'
                else:
                    rstr += f'\n{_k:>18}: {_v}'
        return rstr
    
    def _internal_add_processing_info(self, info):
        proc = self.stats.set
    
    def update_time_range(self, trace):
        if self.min_starttime is None or self.min_starttime > trace.stats.starttime:
            self.min_starttime = trace.stats.starttime
        if self.max_starttime is None or self.max_starttime < trace.stats.starttime:
            self.max_starttime = trace.stats.starttime
        if self.min_endtime is None or self.min_endtime > trace.stats.endtime:
            self.min_endtime = trace.stats.endtime
        if self.max_endtime is None or self.max_endtime < trace.stats.endtime:
            self.max_endtime = trace.stats.endtime




class DictStream(Stream):

    def __init__(
            self,
            traces=None,
            header={},
            **options
    ):
        """
        Initialize a DictStream object

        :: INPUTS ::
        :param traces: [obspy.core.trace.Trace] or [list-like] thereof
                        that are added to the self.traces attribute via
                        the __add__ method.
        :param header: [dict] dict stream header information
                        see wyrm.core.dictstream.DictStreamHeader
        :param options: [kwargs] collector for kwargs to pass to DictStream.__add__
        """

        super().__init__()

        self.stats = DictStreamHeader(header=header)
        self.traces = {}
        self.siteinst_index = {}
        self.nsite = 0
        self.ninst = 0
        self.options = options

        if traces is not None:
            self.__add__(traces, **options)    
            self.is_syncd(run_assessment=True)

        # if self.is_syncd(run_assessment=True):
        #     print('Traces ready for tensor conversion')
        # else:
        #     print('Steps required to sync traces prior to tensor conversion')

    def __iter__(self):
        """
        Return a robust iterator for dictstream.traces to iterate
        across keyed values (i.e., list(self.traces.values()))
        """
        return list(self.traces.values()).__iter__()


    def __getitem__(self, index):
        """
        Fusion between the __getitem__ method for lists and dictionaries
        This accepts integer and slice indexing to access items in
        DictStream.traces, as well as str-type key values. 

        __getitem__ calls that retrieve a single trace return a trace-type object
        whereas calls that retrieve multiple traces return a DictStream object

        Because the DictStream class defaults to using trace.id values for
        keys (which are str-type), this remove the ambiguity in the expected
        type for self.traces' keys.

        :: INPUTS ::
        :param index: [int] - returns the ith trace in list(self.traces.values())
                      [slice] - returns a DictStream with the specified slice from 
                            list(self.traces.values())
                      [str] - returns the trace corresponding to self.traces[index]
                      [list] of [str] - return a DictStream containing the traces as 
                            specified by a list of trace keys
        
        :: OUTPUT ::

        """
        # Handle single item fetch
        if isinstance(index, int):
            trace = self.traces[list(self.traces.keys())[index]]
            out = trace
        # Handle slice fetch
        elif isinstance(index, slice):
            keyslice = list(self.traces.keys()).__getitem__(index)
            traces = [self.traces[_k] for _k in keyslice]
            out = self.__class__(traces=traces)
        # Preserve dict.__getitem__ behavior for string arguments
        elif isinstance(index, str):
            if index in self.traces.keys():
                out = self.traces[index]
            else:
                raise KeyError(f'index {index} is not a key in this DictStream\'s traces attribute')
        elif isinstance(index, list):
            if all(isinstance(_e, str) and _e in self.traces.keys() for _e in index):
                traces = [self.traces[_k] for _k in index]
                out = self.__class__(traces=traces)
            else:
                raise KeyError('not all keys in index are str-type and keys in this DictStream\'s traces attribute')
        else:
            raise TypeError('index must be type int, str, list, or slice')
        return out


    def __add__(self, other, **options):
        """
        Add a Trace-type or Stream-type object to this DictStream
        """
        # If adding a list-like
        if isinstance(other, (list, tuple)):
            # Check that each entry is a trace
            for _tr in other:
                # Raise error if not
                if not isinstance(_tr, Trace):
                    raise TypeError('all elements in traces must be type obspy.core.trace.Trace')
                # Use Trace ID as label
                else:
                    iterant = other
        # Handle case where other is Stream-like
        elif isinstance(other, Stream):
                iterant = other
        # Handle case where other is type-Trace (includes TraceBuffer)
        elif isinstance(other, Trace):
            # Make an iterable list of 1 item
            iterant = [other]

        elif isinstance(other, dict):
            if all(isinstance(_tr, Trace) and _k == _tr.id for _k, _tr in dict.items()):
                iterant = other.values()
        
        else:
            raise TypeError(f'input "other" type {type(other)} not supported')


        ## UPDATE/ADD SECTION ##
        # Iterate across list-like of traces "iterant"
        for _tr in iterant:
            # LEAVE THE NEXT 2 LINES OUT - LET THE USER/CODE SET WHAT TYPE OF TRACE
            # if not isinstance(_tr, MLTrace):
            #     _tr = MLTrace().from_trace(_tr)
            # If new trace ID, update traces using dictionary method 'update'
            if _tr.id not in self.traces.keys():
                self.traces.update({_tr.id: _tr})
            # If extant trace ID, add new trace to current holdings using the Trace.__add__ method
            else:
                self.traces[_tr.id].__add__(_tr, **options)
            # Update time range
            self.stats.update_time_range(_tr)
        self._update_siteinst_index()

    def _update_siteinst_index(self):
        """
        Scan across site (Network.Station) and Instrument (Location.Band|Instrument)
        """
        new_index = {}
        nsite = 0
        ninst = 0
        for _tr in self.values():
            comp = _tr.stats.component
            inst = f'{_tr.stats.location}.{_tr.stats.channel[:-1]}'
            site = f'{_tr.stats.network}.{_tr.stats.station}'
            if site not in new_index.keys():
                new_index.update({site: {inst: [comp]}})
                nsite += 1
                ninst += 1
            elif inst not in new_index[site]:
                new_index[site].update({inst: [comp]})
                ninst += 1
            elif comp not in new_index[site][inst]:
                new_index[site][inst].append(comp)
            else:
                pass
        self.siteinst_index = new_index
        self.nsite = nsite
        self.ninst = ninst

    def __repr__(self, extended=False):
        rstr = self.stats.__repr__()
        if len(self.traces) > 0:
            id_length = max(len(_tr.id) for _tr in self.traces.values())
        else:
            id_length=0
        rstr += f'\n{len(self.traces)} MLTrace(s) in DictStream from {self.ninst} instruments(s) across {self.nsite} stations(s):\n'
        if len(self.traces) <= 20 or extended is True:
            for _l, _tr in self.items():
                rstr += f'{_l:} : {_tr.__str__(id_length)}\n'
        else:
            _l0, _tr0 = list(self.items())[0]
            _lf, _trf = list(self.items())[-1]
            rstr += f'{_l0:} : {_tr0.__str__(id_length)}\n'
            rstr += f'...\n({len(self.traces) - 2} other traces)\n...\n'
            rstr += f'{_lf:} : {_trf.__str__(id_length)}\n'
            rstr += f'[Use "print(DictStream.__repr__(extended=True))" to print all labels and MLTraces]'
        return rstr
    
    def __str__(self):
        rstr = 'wyrm.core.data.MLStream()'
        return rstr

    def copy(self):
        """
        Return a deepcopy of this MLStream
        """
        return deepcopy(self)
    
    def copy_dataless(self):
        """
        Return an empty MLStream with a deepcopy of this
        MLStream's self.stats
        """
        hdr = deepcopy(self.stats)
        dst = DictStream()
        dst.stats = hdr
        return dst   
    
    def fnselect(self, fnstr):
        """
        Create a copy of this DictStream with trace ID's (keys) that conform to an input
        Unix wildcard-compliant string. This updates the self.stats.common_id attribute

        :: INPUT ::
        :param fnstr: [str] search string to search with
                        fnmatch.filter(self.traces.keys(), fnstr)

        :: OUTPUT ::
        :return dst: [wyrm.core.data.DictStream] subset copy
        """
        matches = fnmatch.filter(self.list_labels(), fnstr)
        dst = self.copy()
        dst.traces = {}

        for _m in matches:
            dst.traces.update({_m: self.traces[_m].copy()})
        dst.stats.common_id = fnstr
        dst._update_siteinst_index()
        dst.stats.processing.append(f'Wyrm 0.0.0: fnselect(fnstr="{fnstr}")')
        return dst
    
    def sort_labels(self, reverse=True):
        self.traces = dict(sorted(self.items(), reverse=reverse))
        return self
    
    def values(self):
        return self.traces.values()

    def list_traces(self):
        """
        Return a list-formatted view of the traces (values) in this MLStream.traces
        """
        return list(self.values())

    def labels(self):
        return self.traces.keys()
    
    def list_labels(self):
        """
        Return a list-formattted view of the labels (keys) in this MLStream.traces
        """
        return list(self.labels())

    def rerun_labeling(self):
        for _l, _tr in self.items():
            if _l != _tr.id:
                _ = self.pop(_l)
                self.traces.update({_tr.id: _tr})

    def items(self):
        return self.traces.items()

    def list_items(self):
        """
        Return a list-formatted view of the labels and traces (items) in this MLStream.traces
        """
        return list(self.items())
    
    def meta(self):
        """
        Return a view of this LabelStream.stats
        """
        return self.stats
    
    def pop(self):
        """
        Execute a popitem() on MLStream.traces and return the popped item
        """
        return self.traces.popitem()
    
    def get_common_id(self):
        return self.stats.common_id
    
    id = property(get_common_id)

    # PROCESSING METHODS #
    
    def _internal_add_processing_info(self, info):
        proc = self.stats.setdefault('processing', [])
        proc.append(info)   
    
    @_add_processing_info
    def taper(self, *args, **kwargs):
        super().taper(*args, **kwargs)

    @_add_processing_info
    def filter(self, type, **options):
        super().filter(type, **options)

    @_add_processing_info
    def resample(self, sampling_rate='ref', window='hann',no_filter=True, strict_length=False):

        if sampling_rate == 'ref': 
            if self.stats.ref_sampling_rate is not None:
                sampling_rate = self.stats.sampling_rate
            else:
                raise ValueError('DictStream.stats.ref_sampling_rate is None - must be set to use "ref" sampling_rate')
        super().resample(sampling_rate, window=window, no_filter=no_filter, strict_length=strict_length)

    @_add_processing_info
    def interpolate(self, *args, **kwargs):
        if 'sampling_rate' not in kwargs.keys():
            if self.stats.ref_sampling_rate is not None:
                kwargs.update({'sampling_rate': self.stats.ref_sampling_rate})
        super().interpolate(*args, **kwargs)


    @_add_processing_info
    def trim(self, starttime=None, endtime=None, pad=False,
            keep_empty_traces=False, nearest_sample=True, fill_value=None):
        """
        A minorly altered version of obspy.core.stream.Stream.trim to accommodate
        the dictionary-type self.traces attribute of DictStream

        see obspy.core.stream.Strea.trim() for detailed documentation and use
        """
        if not self:
            return self
        if nearest_sample:
            # One of the only differences between this script and obspy's Stream.trim)()
            tr = self.__getitem__(0)

            try:
                if starttime is not None:
                    delta = round_away(
                        (starttime - tr.stats.starttime) *
                        tr.stats.sampling_rate)
                    starttime = tr.stats.starttime + delta * tr.stats.delta
                if endtime is not None:
                    delta = round_away(
                        (endtime - tr.stats.endtime) * tr.stats.sampling_rate)
                    # delta is negative!
                    endtime = tr.stats.endtime + delta * tr.stats.delta
            except TypeError:
                msg = ('starttime and endtime must be UTCDateTime objects '
                       'or None for this call to Stream.trim()')
                raise TypeError(msg)
        for trace in self:
            trace.trim(starttime, endtime, pad=pad,
                       nearest_sample=nearest_sample, fill_value=fill_value)
        if not keep_empty_traces:
            # remove empty traces after trimming
            self.traces = [_i for _i in self if _i.stats.npts]
        return self

    @_add_processing_info
    def apply_trace_method(self, method_name, *args, **kwargs):
        """
        Apply a valid obspy.core.trace.Trace method that modifies
        individual traces inplace
        """
        if not isinstance(method_name, str):
            raise TypeError("method_name must be type str")
        elif method_name not in dir(Trace):
            if method_name in dir(Stream):
                raise AttributeError('method {method_name} is an obspy.core.stream.Stream method')
            raise AttributeError('method {method_name} is not in the attribute list of obspy.core.trace.Trace')
        for _tr in self.traces.values():
            eval(f'_tr.{method_name}(*args, **kwargs)')
        return self

    ###############################
    # DATA TRANSFORMATION METHODS #
    ###############################
    def to_numpy_array(self, **options):
        if self.is_syncd(run_assessment=True, **options):
            if not any(np.ma.is_masked(_tr.data) for _tr in self):
                ndata = np.c_[[_tr.data for _tr in self]]
            else:
                raise ValueError()
        return ndata
    



    def split_by_site(self, **options):
        """
        Create a dictionary containing a split view of the contents of 
        this DictStream object consisting of site code keys and
        DictStream object entries referencing trace objects related
        to 
        """

        site_dict_streams = {}
        for _tr in self.values():
            site_code = f'{_tr.stats.network}.{_tr.stats.station}.{_tr.stats.location}'
            if site_code not in site_dict_streams.keys():
                site_dict_streams.update({site_code: DictStream(traces=_tr)})
                site_dict_streams[site_code].stats.common_id=f'{site_code}.*'
            else:
                site_dict_streams[site_code].__add__(_tr, **options)
        for _idst in site_dict_streams.values():
            _idst._update_siteinst_index()
        return site_dict_streams
    
    def split_by_instrument(self, heirarchical=False, **options):
        """
        Split data indexing in this DictStream into a (heirarchical) dictionary
        at the instrument level. If heirarchical, splits are at the site and instrument
        levels.

        NOTE: Restructured indexing still points back to the same in-memory objects as the
            source DictStream, so changes actioned on the instrument-split view data affect
            contents of both the input and output DictStream objects

        heirarchical is True:
        dict {site_code: {inst_code: DictStream()}}

        heirarchical is False:
        dict {inst_code: DictStream()}

        :: INPUTS ::
        :param heirarchical: [bool] - create a heirarchical dictionary structure? (see above)
        :param **options: [kwargs] - key word argument collector that is passed to the
                        DictStream.__add__ method

        :: OUTPUT ::
        :return dict_dst: [dict] (heirarchical) dictionary of DictStream objects
        """
        dict_dst = {}
        for _tr in self:
            site_code = _tr.site
            inst_code = _tr.inst
            if heirarchical:
                if site_code not in dict_dst.keys():
                    dict_dst.update({site_code:{inst_code: DictStream(traces=_tr)}})
                    dict_dst[site_code][inst_code].stats.common_id = inst_code
                elif inst_code not in dict_dst[site_code].keys():
                    dict_dst[site_code].update({inst_code: DictStream(traces=_tr)})
                    dict_dst[site_code][inst_code].stats.common_id = inst_code
                else:
                    dict_dst[site_code][inst_code].__add__(_tr, **options)

            else:
                if inst_code not in dict_dst.keys():
                    dict_dst.update({inst_code: DictStream(traces=_tr)})
                    dict_dst[inst_code].stats.common_id = inst_code
                else:
                    dict_dst[inst_code].__add__(_tr, **options)
        
        
        for _k0 in dict_dst.keys():
            if heirarchical:
                for _k1 in dict_dst[_k0].keys():
                    dict_dst[_k0][_k1]._update_siteinst_index()
            else:
                dict_dst[_k0]._update_siteinst_index()
        return dict_dst



    # SYNC CHECK METHODS

    def _assess_starttime_sync(self, new_ref_starttime=None):
        """
        Determine if all traces' starttime values match the 
        ref_starttime specified in the header or as an optional input 

        and update the following attributes:
            stats.sync_status['starttime'] [bool]
            stats.ref_starttime (if new_ref_starttime is specified)

        :: INPUTS ::
        :param new_ref_starttime: (optional) [UTCDateTime] - new reference time for assessment
        """
        if isinstance(new_ref_starttime, UTCDateTime):
            ref_t0 = new_ref_starttime
        else:
            ref_t0 = self.stats.ref_starttime
        
        if all(_tr.stats.starttime == ref_t0 for _tr in self.list_traces()):
            self.stats.sync_status['starttime'] = True
        else:
            self.stats.sync_status['starttime'] = False

    def _assess_sampling_rate_sync(self, new_ref_sampling_rate=False):
        """
        Determine if all traces' sampling_rate values match the 
        ref_sampling_rate specified in the header or as an optional input 

        and update the following attributes:
            stats.sync_status['sampling_rate'] [bool]
            stats.ref_sampling_rate (if new_ref_sampling_rate is specified)

        :: INPUTS ::
        :param new_ref_sampling_rate: (optional) [float] - new reference sampling rate
        """
        if isinstance(new_ref_sampling_rate, (float,int)):

            ref_sampling_rate = float(new_ref_sampling_rate)
        else:
            ref_sampling_rate = self.stats.ref_sampling_rate
        
        if all(_tr.stats.sampling_rate == ref_sampling_rate for _tr in self.list_traces()):
            self.stats.sync_status['sampling_rate'] = True
        else:
            self.stats.sync_status['sampling_rate'] = False

    def _assess_npts_sync(self, new_ref_npts=False):
        """
        Determine if all traces' npts values match the 
        ref_npts specified in the header or as an optional input 

        and update the following attributes:
            stats.sync_status['npts'] [bool]
            stats.ref_npts (if new_ref_npts is specified)

        :: INPUTS ::
        :param new_ref_npts: (optional) [UTCDateTime] - new reference number of samples
        """
        if isinstance(new_ref_npts, int):
            ref_npts = new_ref_npts
        else:
            ref_npts = self.stats.ref_npts
        
        if all(_tr.stats.npts == ref_npts for _tr in self.list_traces()):
            self.stats.sync_status['npts'] = True
        else:
            self.stats.sync_status['npts'] = False

    def is_syncd(self, run_assessment=True, **options):
        """
        Assess if the contents of this DictStream object are synced to the 
        target timing, sampling rate, and number of samples (npts) provided
        as reference values in the header of this DictStream. 

        :: INPUTS ::
        :param run_assessment: [bool] - run checks to update stats.sync_status?
                            False - use current information in stats.sync_status
                            True - run  _assess_starttime_sync
                                        _assess_sampling_rate_sync
                                        _assess_npts_sync
        :param **options: [kwargs] optional kwarg gatherer to pass to the
                            _assess_*_sync methods noted above.
        
        :: OUTPUT ::
        :return status: [bool] True  = all sync metrics are True
                               False = not all sync metrics are True 
        """
        if run_assessment:
            if 'new_ref_starttime' in options.keys():
                self._assess_starttime_sync(new_ref_starttime=options['new_ref_starttime'])
            else:
                self._assess_starttime_sync()
            if 'new_ref_sampling_rate' in options.keys():
                self._assess_sampling_rate_sync(new_ref_sampling_rate=options['new_ref_sampling_rate'])
            else:
                self._assess_sampling_rate_sync()
            if 'new_ref_npts' in options.keys():
                self._assess_npts_sync(new_ref_npts=options['new_ref_npts'])
            else:
                self._assess_npts_sync()
        if all(self.stats.sync_status.values()):
            status = True
        else:
            status = False
        return status

    def diagnose_sync(self):
        """
        Produce a pandas.DataFrame that contains the sync status
        for the relevant sampling reference attriburtes:
            starttime
            sampling_rate
            npts
        where 
            True indicates a trace (id in index) matches the
                 reference attribute (column) value
            False indicates the trace mismatches the reference
                    attribute value
            'NoRefVal' indicates no reference value was available

        :: OUTPUT ::
        :return out: [pandas.dataframe.DataFrame] with sync
                    assessment values.
        """

        index = []; columns=['starttime','sampling_rate','npts'];
        holder = []
        for _l, _tr in self.items():
            index.append(_l)
            line = []
            for _f in columns:
                if self.stats['ref_'+_f] is not None:
                    line.append(_tr.stats[_f] == self.stats['ref_'+_f])
                else:
                    line.append(False)
            holder.append(line)
        out = pd.DataFrame(holder, index=index, columns=columns)
        return out

    def get_trace_completeness(self, starttime=None, endtime=None, pad=True, **options):
        completeness_index = {}
        for _l, _tr in self.items():
            _xtr = _tr.copy().trim(starttime=starttime, endtime=endtime, pad=pad, **options)
            if not np.ma.is_masked(_xtr.data):
                completeness_index.update({_l: 1})
            else:
                completeness = 1 - (sum(_xtr.data.mask)/_xtr.stats.npts)
                completeness_index.update({_l: completeness})
        return completeness_index

    # INSTRUMENT LEVEL METHODS #

    def assess_window_readiness(self, ref_comp='Z', ref_comp_thresh=0.95, comp_map={'Z': 'Z3','N': 'N1', 'E': 'E2'}):
        self._update_siteinst_index()
        # Check if one site only
        if self.nsite == 1:
            pass
        else:
            raise ValueError('Fill rule can only be applied to a single station')
        # Check if one instrument only
        if self.ninst == 1:
            pass
        else:
            raise ValueError('Fill rule can only be applied to a single instrument')
        # Check for reference component
        ref_code = None
        for _l, _tr in self.items():
            if _tr.stats.component in comp_map[ref_comp]:
                ref_code = _l
        if ref_code is not None:
            pass
        else:
            raise ValueError('Fill rule can only be applied if the reference trace/component is present')
        # Evaluate trace completeness
        cidx = self.get_trace_completeness()
        # Check if reference trace is sufficiently complete
        if cidx[ref_code] >= ref_comp_thresh:
            return ref_code
        else:
            return False


    def apply_fill_rule(self, ref_comp='Z', rule='zeros', ref_comp_thresh=0.95, other_comp_thresh=0.95, comp_map={'Z': 'Z3','N': 'N1', 'E': 'E2'}):
        ref_code = self.assess_window_readiness(ref_comp=ref_comp, ref_comp_thresh=ref_comp_thresh, comp_map=comp_map)
        if ref_code:
            cidx = self.get_trace_completeness()
        else:
            raise ValueError('Reference trace has insufficient data to apply a fill rule')
        # If all traces meet the other_comp_thresh threshold and there are 3 traces
        if all(_cv >= other_comp_thresh for _cv in cidx.values) and len(self.traces) == 3:
            # Attach processing note that everything passed
            self.stats.processing.append('Wyrm 0.0.0: apply_fill_rule - 3-C data present')
            # Return self
            return self
        # If some piece was missed
        else:
            
            if rule == 'zeros':
                self._apply_zeros_fill_rule(ref_code)
            elif rule == 'clonez':
                self._apply_clonez_fill_rule(ref_code)
            elif rule == 'clonehz':
                self._apply_clonehz_fill_rule(ref_code, cidx, other_comp_thresh)
        return self
    

    def _apply_zeros_fill_rule(self, ref_code):
        """
        -- PRIVATE METHOD --

        ASSUMING THAT ONE OR MORE TRACES ARE BELOW COMPLETENESS THRESHOLDS
        replace non-reference trace(s) with duplicates of the reference trace
        and with component codes of N and E and 0-valued data
         - After Retailleau et al. (2022)

        WARNING: This is conducted on data in-place. If you want to experiment
        with it's behavior, use dictstream.copy()._apply_clonez

        :: INPUT ::
        :param ref_code: [str] reference trace ID string - assumed to be a
                            Z component (or comparable SEED naming convention mapping)
                            (e.g., component 3)
        :: OUTPUT ::
        :return self:
        """

        if ref_code not in self.labels():
            raise ValueError(f'ref_code {ref_code} is not in the trace label set')
        # Iterate across all elements
        for _k in self.labels():
            if _k != ref_code:
                # Pop off non-reference trace
                _tr = self.pop(_k)
        # Append two 0-traces with N and E component codes
        _tr0 = self.traces[ref_code].copy()
        _tr0.data = np.zeros(shape=_tr0.data.shape, dtype=_tr0.data.dtype)
        _tr0.stats.channel = _tr0.stats.channel[:-1]
        for _comp in 'NE':
            _trx = _tr0.copy()
            _trx.stats.channel += _comp
            self.__add__(_trx)
        self.stats.processing.append(f'Wyrm 0.0.0: _apply_zeros_fill_rule({ref_code})')
        return self
    
    def _apply_clonez_fill_rule(self, ref_code):
        """
        -- PRIVATE METHOD --

        ASSUMING THAT ONE OR MORE TRACES ARE BELOW COMPLETENESS THRESHOLDS
        replace non-reference trace(s) with duplicates of the reference trace
        and with component codes of N and E
         - After Ni et al. (2023)

        WARNING: This is conducted on data in-place. If you want to experiment
        with it's behavior, use dictstream.copy()._apply_clonez

        :: INPUT ::
        :param ref_code: [str] reference trace ID string - assumed to be a
                            Z component (or comparable SEED naming convention mapping)
                            (e.g., component 3)
        :: OUTPUT ::
        :return self:
        """

        if ref_code not in self.labels():
            raise ValueError(f'ref_code {ref_code} is not in the trace label set')
        # Iterate across all elements
        for _k in self.labels():
            if _k != ref_code:
                # Pop off non-reference trace
                _tr = self.pop(_k)
        # Append two Z-traces with N and E component codes
        _trC = self.traces[ref_code].copy()
        _trC.stats.channel = _trC.stats.channel[:-1]
        for _comp in 'NE':
            _trx = _trC.copy()
            _trx.stats.channel += _comp
            self.__add__(_trx)
        self.stats.processing.append(f'Wyrm 0.0.0: _apply_clonez_fill_rule({ref_code})')
        return self
    


    

    # def _apply_clonehz_fill_rule(self, ref_code, completeness_index, completeness_threshold):
    #     """
        
    #     """
    #     if ref_code not in self.labels():
    #         raise ValueError(f'ref_code {ref_code} is not in the trace label set')
    #     # Iterate across all elements
    #     for _k in self.labels():
    #         if _k != ref_code:
    #             # If non-reference element falls below threshold
    #             if completeness_index[_k] < completeness_threshold:
    #                 # Pop off non-reference trace
    #                 _tr = self.pop(_k)
    #     # If all traces pass
    #     if len(self.traces) == 3:
    #         pass
    #     # If only ref_code trace left, apply clonez rule
    #     elif len(self.traces) == 1:
    #         # Append two Z-traces with N and E component codes
    #         _trC = self.traces[ref_code].copy()
    #         _trC.stats.channel = _trC.stats.channel[:-1]
    #         for _comp in 'NE':
    #             _trx = _trC.copy()
    #             _trx.stats.channel += _comp
    #             self.__add__(_trx)
    #         self.stats.processing.append(f'Wyrm 0.0.0: _apply_clonez_fill_rule({ref_code})')
    #         # Append two Z-traces with N and E component codes
    #         _trC = self.traces[ref_code].copy()
    #         _trC.stats.channel = _trC.stats.channel[:-1]
    #         for _comp in 'NE':
    #             _trx = _trC.copy()
    #             _trx.stats.channel += _comp
    #             self.__add__(_trx)
    #     else:

    #     return self

                
    # def relabel_by_component(self, comp_map={"Z":'Z3', 'N': 'N1', 'E': 'E2'}):
    #     self._update_siteinst_index()
    #     if self.nsite == 1:
    #         pass
    #     else:
    #         raise ValueError('can only relabel by component for traces from a single site')
    #     if self.ninst == 1:
    #         pass
    #     else:
    #         raise ValueError('can only relabel by component for traces from a single instrument')
    #     for _l, _tr in self.items():
    #         if _l == self.id
        





    # def make_instrument_stream(self, instrument_code, ref_comp='Z', comp_map={"Z": 'Z3', "N": 'N1', "E": 'E2'} fill_rule='zeros'):

    #     ist = self.fnselect(instrument_code+'*')
    #     if all (instrument_code + _rc not in ist.labels() for _rc in comp_map[ref_comp]):
    #         raise ValueError(f'reference channel not found')
    #     else:
            



# class WindowStream(DictStream):

#     def __init__(self, ref_trace, other_traces=None, header={}):



        