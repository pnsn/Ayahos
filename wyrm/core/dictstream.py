import fnmatch, inspect, time, warnings
import numpy as np
from decorator import decorator
from obspy.core.stream import Stream, read
from obspy.core.trace import Trace, Stats
from obspy.core.util.attribdict import AttribDict
from obspy.core import compatibility
from wyrm.core.trace import MLTrace, MLTraceBuffer
from wyrm.util.pyew import wave2mltrace
###################################################################################
# Dictionary Stream Header Class Definition #######################################
###################################################################################

class DictStreamHeader(AttribDict):
    defaults = {
        'reference_id': '*',
        'min_starttime': None,
        'max_starttime': None,
        'min_endtime': None,
        'max_endtime': None,
        'processing': []
    }

    def __init__(self, header={}):
        super(DictStreamHeader, self).__init__(header)
    
    def __repr__(self):
        rstr = '----Stats----'
        for _k, _v in self.items():
            if _v is not None:
                if self.min_starttime != self.max_starttime or self.min_endtime != self.max_endtime:
                    if 'min_' in _k:
                        if _k == 'min_starttime':
                            rstr += f'\n{"min time range":>18}: {self.min_starttime} - {self.min_endtime}'
                    elif 'max_' in _k:
                        if _k == 'max_starttime':
                            rstr += f'\n{"max time range":>18}: {self.max_starttime} - {self.max_endtime}'
                elif 'time' in _k:
                    if _k == 'min_starttime':
                        rstr += f'\n{"uniform range":>18}: {self.min_starttime} - {self.min_endtime}'
                else:
                    rstr += f'\n{_k:>18}: {_v}'
        return rstr

    
    def update_time_range(self, trace):
        if self.min_starttime is None or self.min_starttime > trace.stats.starttime:
            self.min_starttime = trace.stats.starttime
        if self.max_starttime is None or self.max_starttime < trace.stats.starttime:
            self.max_starttime = trace.stats.starttime
        if self.min_endtime is None or self.min_endtime > trace.stats.endtime:
            self.min_endtime = trace.stats.endtime
        if self.max_endtime is None or self.max_endtime < trace.stats.endtime:
            self.max_endtime = trace.stats.endtime

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
    info = [time.time(), "Wyrm 0.0.0:","{function}".format(function=func.__name__), "(%s)"]
    arguments = []
    arguments += \
        ["%s=%s" % (k, repr(v)) if not isinstance(v, str) else
         "%s='%s'" % (k, v) for k, v in callargs.items()]
    arguments += \
        ["%s=%s" % (k, repr(v)) if not isinstance(v, str) else
         "%s='%s'" % (k, v) for k, v in kwargs_.items()]
    arguments.sort()
    info[-1] = info[-1] % "::".join(arguments)
    self = args[0]
    result = func(*args, **kwargs)
    # Attach after executing the function to avoid having it attached
    # while the operation failed.
    self._internal_add_processing_info(info)
    return result

###################################################################################
# Dictionary Stream Class Definition ##############################################
###################################################################################

class DictStream(Stream):
    """
    An adaptation of the obspy.core.stream.Stream class that provides a dictionary
    housing for multiple wyrm.core.trace.MLTrace objects and a header object
    that tracks timing and processing metadata for its contents. Updated methods
    are provided to allow use of obspy.core.stream.Stream class methods, along
    with new methods oriented towards processing tasks in the Wyrm workflow. 

    NOTE: Not all inherited methods are supported, which will be addressed
        on an as-needed basis.
    """
    _max_processing_info = 100
    def __init__(self, traces=None, header={}, **options):
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
        # initialize as empty stream
        super().__init__()
        # Create stats attribute with DictStreamHeader
        self.stats = DictStreamHeader(header=header)
        # Redefine self.traces as dict
        self.traces = {}
        if traces is not None:
            self.__add__(traces, **options)
            self.stats.reference_id = self.get_reference_id()


    def _internal_add_processing_info(self, info):
        """
        Add the given informational string to the `processing` field in the
        trace's :class:`~obspy.core.trace.Stats` object.
        """
        proc = self.stats.setdefault('processing', [])
        if len(proc) == self._max_processing_info-1:
            msg = ('List of processing information in Trace.stats.processing '
                   'reached maximal length of {} entries.')
            warnings.warn(msg.format(self._max_processing_info))
        if len(proc) < self._max_processing_info:
            proc.append(info)

    ###############################################################################
    # MAGIC METHOD UPDATES ########################################################
    ###############################################################################
            
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
        :return out: see INPUTS
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
    
    def __setitem__(self, index, trace):
        if isinstance(index, int):
            key = list(self.traces.keys())[index]
        elif isinstance(index, str):
            key = index
        else:
            raise TypeError(f'index type {type(index)} not supported. Only int and str')
        self.traces.update({key, trace})

    def __delitem__(self, index):
        if isinstance(index, str):
            key = index
        elif isinstance(index, int):
            key = list(self.traces.keys())[index]
        else:
            raise TypeError(f'index type {type(index)} not supported. Only int and str')   
        self.traces.__delitem__(key)


    def __getslice__(self, i, j, k=1):
        """
        Updated __getslice__ that leverages the DictStream.__getitem__ update
        from comparable magic methods for obspy.core.stream.Stream.
        """
        return self.__class__(traces=self[max(0,i):max(0,j):k])


    def __add__(self, other, key_attr='id', **options):
        if isinstance(other, Trace):
            self._add_trace(other, key_attr=key_attr, **options)
        elif isinstance(other, Stream):
            self._add_stream(other, key_attr=key_attr, **options)
        elif isinstance(other, (list, tuple)):
            if all(isinstance(_tr, Trace) for _tr in other):
                self._add_stream(other, key_attr=key_attr, **options)
            else:
                raise TypeError('other elements are not all Trace-like types')
        else:
            raise TypeError(f'other type "{type(other)}" not supported.')

    def _add_trace(self, other, key_attr='id', **options):
        # If potentially appending a wave
        if isinstance(other, dict):
            try:
                other = wave2mltrace(other)
            except SyntaxError:
                pass
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
        
        if isinstance(other, MLTrace):
            # Get id of MLTrace "other"
            key_opts = other.key_opts
            # If id is not in traces.keys() - use dict.update
            key = key_opts[key_attr]
            # If new key
            if key not in self.traces.keys():
                self.traces.update({key: other})
            # If key is in traces.keys() - use __add__
            else:
                self.traces[key].__add__(other, **options)
            self.stats.update_time_range(other)
            self.stats.reference_id = self.get_reference_id()

    def _add_stream(self, stream, **options):
        for _tr in stream:
            self._add_trace(_tr, **options)

    def __str__(self):
        rstr = 'wyrm.core.data.DictStream()'
        return rstr

    def __repr__(self, extended=False):
        rstr = self.stats.__repr__()
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
    # SEARCH METHODS ##############################################################
    ###############################################################################
    
    def fnselect(self, fnstr, ascopy=False):
        matches = fnmatch.filter(self.traces.keys(), fnstr)
        out = self.__class__(header=self.stats.copy())
        for _m in matches:
            if ascopy:
                _tr = self.traces[_m].copy()
            else:
                _tr = self.traces[_m]
            out.traces.update({_m: _tr})
        out.stats.reference_id = out.get_reference_id()
        return out

    def get_unique_id_elements(self):
        N, S, L, C, M, W = [], [], [], [], [], []
        for _tr in self:
            hdr = _tr.stats
            if hdr.network not in N:
                N.append(hdr.network)
            if hdr.station not in S:
                S.append(hdr.station)
            if hdr.location not in L:
                L.append(hdr.location)
            if hdr.channel not in C:
                C.append(hdr.channel)
            if hdr.model not in M:
                M.append(hdr.model)
            if hdr.weight not in W:
                W.append(hdr.weight)
        out = dict(zip(['network','station','location','channel','model','weight'],
                       [N, S, L, C, M, W]))
        return out
    
    def get_reference_id_elements(self):
        ele = self.get_unique_id_elements()
        out = {}
        for _k, _v in ele.items():
            if len(_v) == 0:
                out.update({_k:'*'})
            elif len(_v) == 1:
                out.update({_k: _v[0]})
            else:
                minlen = 999
                maxlen = 0
                for _ve in _v:
                    if len(_ve) < minlen:
                        minlen = len(_ve)
                    if len(_ve) > maxlen:
                        maxlen = len(_ve)
                _cs = []
                for _i in range(minlen):
                    _cc = _v[0][_i]
                    for _ve in _v:
                        if _ve[_i] == _cc:
                            pass
                        else:
                            _cc = '?'
                            break
                    _cs.append(_cc)
                if all(_c == '?' for _c in _cs):
                    _cs = '*'
                else:
                    if minlen != maxlen:
                        _cs.append('*')
                    _cs = ''.join(_cs)
                out.update({_k: _cs})
        return out

    def get_reference_id(self):
        ele = self.get_reference_id_elements()
        out = '.'.join(ele.values())
        return out                
    
    def split_on_key(self, key='instrument', **options):
        if key not in MLTrace().key_opts.keys():
            raise ValueError(f'key {key} not supported.')
        out = {}
        for _tr in self:
            key_opts = _tr.key_opts
            _k = key_opts[key]
            if _k not in out.keys():
                out.update({_k: self.__class__(traces=_tr)})
            else:
                out[_k].__add__(_tr, **options)
        return out
    
    def to_component_streams(self, component_aliases={'Z': 'Z3', 'N': 'N1', 'E': 'E2'}, ascopy=False, **options):
        # Split by instrument_id
        if ascopy:
            out = self.copy().split_on_key(key='instrument', **options)
        else:
            out = self.split_on_key(key='instrument', **options)
        # Iterate and update
        for _k, _v in out.items():
            out.update({_k: ComponentStream(traces=_v, component_aliases=component_aliases)})
        return out
    

    # def split(self, keys=('site', 'instrument'), flat=True, **options):
    #     if isinstance(keys, str):
    #         keys = (keys)
    #     if not all(_k in MLTrace().key_opts.keys() for _k in keys):
    #         raise ValueError
        
    #     out = {}
    #     for _tr in self:    
    #         _kl = []
    #         for _k in keys:
    #             _kl.append(_k)
    #         if flat:
    #             _out_key = '_'.join(_kl)
    #         else:
    #             for _k in _kl[::-1]:
    #                 _x

    ###############################################################################
    # UPDATED METHODS FROM OBSPY STREAM ###########################################
    ###############################################################################
    @_add_processing_info
    def trim(self, starttime=None, endtime=None, pad=True, keep_empty_traces=True, nearest_sample=True, fill_value=None):
        """
        Slight adaptation of obspy.core.stream.Stream.trim() to facilitate the dict-type self.traces
        attribute syntax.

        see obspy.core.stream.Stream.trim() for full explanation of the arguments and behaviors
        
        :: INPUTS ::
        :param starttime: [obspy.core.utcdatetime.UTCDateTime] or [None]
                        starttime for trim on all traces in DictStream
        :param endtime: [obspy.core.utcdatetime.UTCDateTime] or [None]
                        endtime for trim on all traces in DictStream
        :param pad: [bool]
                        should trim times outside bounds of traces
                        produce masked (and 0-valued fold) samples?
                        NOTE: In this implementation pad=True as default
        :param keep_empty_traces: [bool]
                        should empty traces be kept?
        :param nearest_sample: [bool]
                        should trim be set to the closest sample(s) to 
                        starttime/endtime?
        :param fill_value: [int], [float], or [None]
                        fill_value for gaps - None results in masked
                        data and 0-valued fold samples in gaps
    
        """
        if not self:
            return self
        # select start/end time fitting to a sample point of the first trace
        if nearest_sample:
            tr = self[0]
            try:
                if starttime is not None:
                    delta = compatibility.round_away(
                        (starttime - tr.stats.starttime) *
                        tr.stats.sampling_rate)
                    starttime = tr.stats.starttime + delta * tr.stats.delta
                if endtime is not None:
                    delta = compatibility.round_away(
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
            self.stats.update_time_range(trace)
        if not keep_empty_traces:
            # remove empty traces after trimming
            self.traces = {_k: _v for _k, _v in self.traces.items() if _v.stats.npts}
            self.stats.update_time_range(trace)
        return self
    
    ###############
    # I/O METHODS - TODO - GO THROUGH SAC TO APPEND MOD TO CHANNEL #
    ###############

#     def write(self, dirname, **options):
#         # Create a stream to use ObsPy's I/O routines for saving to a directory name
        
#         stream = Stream()
#         for _tr in self:
#             if _tr.mod != '.':
#                 mod = f'{_tr.stats.model}.{_tr.stats.weight}'
#             else:
#                 mod = ''
#             data = _tr.data
#             tr_fold = _tr.get_fold_trace()
#             # Tack model information into Network metadata
#             tr_fold.stats.network += f'_{mod}'


#             hdr = _tr.stats
#             header = Stats()
#             for _k in header.defaults.keys():
#                 header.update({_k: self.stats[_k]})
#             tr_data = Trace(data=data, header=header)
#             tr_data.stats.network += f'_{mod}'
            
#             stream += tr_fold
#             stream += tr_data

#         out = stream.write(filename, format='MSEED', **options)
#         return out

# def read_from_sac(filenames, **options):
#     st = Stream()

#     st = read(filename, fmt='MSEED', **options)
#     dst = DictStream()
#     holder = {}
#     for tr in st:
#         if '_' in tr.stats.network:
#             net, mod, wgt = tr.stats.network.split('_')
#         else:
#             net, mod, wgt, = tr.stats.network, '', ''
#         hdr = tr.stats
#         hdr.update({'network': net, 'model': mod, 'weight': wgt})
#         instrument_id = f'{hdr.network}.{hdr.station}.{hdr.location}.{hdr.channel[:-1]}?.{hdr.model}.{hdr.weight}'
#         if instrument_id in holder.keys():
#             if component != 'f':

        
#         component = hdr.component
#         if component == 'f':


#         if instrument_id in holder.keys():
#             holder[instrument_id].update({component: bundle})
#         else:
#             holder.update({instrument_id: {component: bundle}})

        
#         bundle = {'data': tr.data, ''}
#         mltr = MLTrace(data=tr.data, header=hdr)


###############################################################################
# Window Stream Header Class Definition
###############################################################################

class ComponentStreamHeader(DictStreamHeader):

    def __init__(self, header={}):
        super(ComponentStreamHeader, self).__init__(header=header)
        self.defaults.update({'aliases': dict({}),
                     'reference_starttime': None,
                     'reference_sampling_rate': None,
                     'reference_npts': None,
                     'min_starttime': None,
                     'max_starttime': None,
                     'min_endtime': None,
                     'max_endtime': None,
                     })
    def __str__(self):
        prioritized_keys = ['reference_id',
                            'reference_starttime',
                            'reference_sampling_rate',
                            'reference_npts',
                            'starttime range',
                            'endtime range',
                            'aliases']
        return self._pretty_str(prioritized_keys)
    

class ComponentStream(DictStream):

    def __init__(self, traces=None, header={}, component_aliases={'Z':'Z3', 'N':'N1', 'E':'E2'}, **options):
        super().__init__()
        self.stats = ComponentStreamHeader(header=header)

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
            # Add traces using the DictStream __add__ method that converts non MLTrace objects into MLTrace objects
            self.__add__(traces, key_attr='component', **options)

    # def __add__(self, other, key_attr='component', **options):
    #     super().__add__(other, key_attr=key_attr, **options)

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
            # If using obspy.core.trace.Trace objects, use id with "?" for component char
            if self.ref_type == Trace:
                ref = tr0.id[:-1]+'?'
            # If using wyrm.core.trace.MLTrace(Buffer) objects, as above with the 'mod' extension
            elif self.ref_type in [MLTrace, MLTraceBuffer]:
                ref = f'{tr0.site}.{tr0.inst}?.{tr0.mod}'
        # Run match on all trace ids
        matches = fnmatch.filter([_tr.id for _tr in traces], ref)
        # If all traces conform to ref
        if all(_tr.id in matches for _tr in traces):
            # If reference_id 
            if self.stats.reference_id == self.stats.defaults['reference_id']:
                self.stats.reference_id = ref
        
        else:
            raise KeyError('Trace id(s) do not conform to reference_id: "{self.stats.reference_id}"')
            
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
        Apply a zero-valued trace in-fill rule for all non-reference components
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
        ref_tr = self[ref_component]
        if ref_tr.fvalid < thresh_dict[ref_component]:
            raise ValueError('insufficient valid data in reference trace')
        else:
            pass
        for _k in thresh_dict.keys():
            trC = ref_tr.copy().to_zero(method='fold').set_comp(_k)
            self.traces.update({_k: trC})

    # @_add_processing_info    
    # def _apply_clone_other(self, ref_component, thresh_dict):
    #     # If it is only the reference component present, run _apply_clone_ref() instaed
    #     if list(self.traces.keys()) == [ref_component]:
    #         self._apply_clone_ref(ref_component, thresh_dict)
    #     else:
    #         pass_dict = {_k: None for _k in self.traces.keys()}
    #         for _k, _tr for self.traces.items():
    #             pass_dict.update{_k: _tr.fvalid >= thresh_dict[_k]}
            
