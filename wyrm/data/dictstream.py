import fnmatch, inspect, time, warnings
from decorator import decorator
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.stream import Stream, read
from obspy.core.trace import Trace
from obspy.core.util.attribdict import AttribDict
from obspy.core import compatibility
from wyrm.data.mltrace import MLTrace
from wyrm.data.componentstream import ComponentStream
from wyrm.util.pyew import wave2mltrace
###################################################################################
# Dictionary Stream Stats Class Definition ########################################
###################################################################################

class DictStreamStats(AttribDict):
    """
    A class to contain metadata for a wyrm.core.dictstream.DictStream object
    of the based on the ObsPy AttribDict (Attribute Dictionary) class. 

    This operates very similarly to obspy.core.trace.Trace objects' Stats object
    (a sibling class)
    """
    defaults = {
        'reference_id': '*',
        'min_starttime': None,
        'max_starttime': None,
        'min_endtime': None,
        'max_endtime': None,
        'processing': []
    }

    _types = {'reference_id': str,
              'min_starttime': (type(None), UTCDateTime),
              'max_starttime': (type(None), UTCDateTime),
              'min_endtime': (type(None), UTCDateTime),
              'max_endtime': (type(None), UTCDateTime)}

    def __init__(self, header={}):
        """
        Initialize a DictStreamStats object

        :: INPUT ::
        :param header: [dict] (optional)
                    Dictionary defining attributes (keys) and 
                    values (values) to assign to the DictStreamStats
                    object
        """
        super(DictStreamStats, self).__init__()
        self.update(header)
    
    def _pretty_str(self, priorized_keys=[], hidden_keys=[], min_label_length=16):
        """
        Return better readable string representation of AttribDict object.

        NOTE: Slight adaptation of the obspy.core.util.attribdict.AttribDict
                _pretty_str method, adding a hidden_keys argument

        :type priorized_keys: list[str], optional
        :param priorized_keys: Keywords of current AttribDict which will be
            shown before all other keywords. Those keywords must exists
            otherwise an exception will be raised. Defaults to empty list.
        :param hidden_keys: [list] of [str]
                        Keywords of current AttribDict that will be hidden
                        NOTE: does not supercede items in prioritized_keys
        :type min_label_length: int, optional
        :param min_label_length: Minimum label length for keywords. Defaults
            to ``16``.
        :return: String representation of current AttribDict object.
        """
        keys = list(self.keys())
        # determine longest key name for alignment of all items
        try:
            i = max(max([len(k) for k in keys]), min_label_length)
        except ValueError:
            # no keys
            return ""
        pattern = "%%%ds: %%s" % (i)
        # check if keys exist
        other_keys = [k for k in keys if k not in priorized_keys and k not in hidden_keys]
        # priorized keys first + all other keys
        keys = priorized_keys + sorted(other_keys)
        head = [pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)


    def __str__(self):
        prioritized_keys = ['reference_id',
                            'min_starttime',
                            'max_starttime',
                            'min_endtime',
                            'max_endtime',
                            'processing']
        return self._pretty_str(prioritized_keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    # def __repr__(self):
    #     """
    #     Provide a user-friendly string representation of the contents of this DictStreamStats object
    #     """
    #     rstr = '----Stats----'
    #     for _k, _v in self.items():
    #         if _v is not None:
    #             if self.min_starttime != self.max_starttime or self.min_endtime != self.max_endtime:
    #                 if 'min_' in _k:
    #                     if _k == 'min_starttime':
    #                         rstr += f'\n{"min time range":>18}: {self.min_starttime} - {self.min_endtime}'
    #                 elif 'max_' in _k:
    #                     if _k == 'max_starttime':
    #                         rstr += f'\n{"max time range":>18}: {self.max_starttime} - {self.max_endtime}'
    #             elif 'time' in _k:
    #                 if _k == 'min_starttime':
    #                     rstr += f'\n{"uniform range":>18}: {self.min_starttime} - {self.min_endtime}'
    #             else:
    #                 rstr += f'\n{_k:>18}: {_v}'
    #     return rstr

    def update_time_range(self, trace):
        """
        Update the minimum and maximum starttime and endtime attributes of this
        DictStreamStats object using timing information from an obspy Trace-like
        object.

        :: INPUT ::
        :param trace: [obspy.core.trace.Trace] or child classes from which to 
                    query starttime and endtime information
        """
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
                        see wyrm.core.dictstream.DictStreamStats
        :param options: [kwargs] collector for kwargs to pass to DictStream.__add__
        """
        # initialize as empty stream
        super().__init__()
        # Create stats attribute with DictStreamStats
        self.stats = DictStreamStats(header=header)
        # Redefine self.traces as dict
        self.traces = {}
        if traces is not None:
            self.__add__(traces, **options)
            self.stats.reference_id = self.get_reference_id()


    def _internal_add_processing_info(self, info):
        """
        Add the given informational string to the `processing` field in the
        DictStream's :class:`wyrm.core.dictstream.DictStreamStats` object.
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
        """
        Wrapper method for the _add_trace() method
        """
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
        """
        Add a trace-like object `other` to this DictStream using elements from
        the trace's id as the dictionary key in the DictStream.traces dictionary

        :: INPUTS ::
        :param other: [obspy.core.trace.Trace] or child-class
                        Trace to append
        :param key_attr: [str] name of the attribute to use as a key.
                        Supported Values:
                            'id' - full N.S.L.C(.M.W) code
                            'site' - Net + Station
                            'inst' - Location + Band & Instrument codes from Channel
                            'instrument'- 'site' + 'inst'
                            'mod' - Model + Weight codes
                            'component' - component code from Channel
        :param **options: [kwargs] key-word argument gatherer to pass to the 
                        MLTrace.__add__() or MLTraceBuffer.__add__() method
        """
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
        """
        Supporting method to iterate across a stream-like object
        and apply the _add_trace() DictStream class method

        :: INPUTS ::
        :param stream: [obspy.core.stream.Stream] or similar
                        an iterable object that returns individual
                        obspy Trace-like objects as iterants
        :param **options: [kwargs] optional key-word argument gatherer
                        to pass kwargs to the DictStream._add_trace method
        """
        for _tr in stream:
            self._add_trace(_tr, **options)

    def __str__(self):
        rstr = 'wyrm.core.data.DictStream()'
        return rstr

    def __repr__(self, extended=False):
        rstr = f'--Stats--\n{self.stats.__str__()}\n-------'
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
        """
        Find DictStream.traces.keys() strings that match the
        input `fnstr` string using the fnmatch.filter() method
        and compose a view (or copy) of the subset DictStream

        :: INPUTS ::
        :param fnstr: [str] Unix wildcard compliant string to 
                        use for searching for matching keys
        :param ascopy: [bool] should the returned DictStream
                        be a view (i.e., accessing the same memory blocks)
                        or a copy of the traces contained within?

        :: OUTPUT ::
        :return out: [wyrm.core.dictstream.DictStream] containing
                        subset traces that match the specified `fnstr`
        """
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
        """
        Compose a dictionary containing lists of
        unique id elements: Network, Station, Location,
        Channel, Model, Weight in this DictStream

        :: OUTPUT ::
        :return out: [dict] output dictionary keyed
                by the above elements and valued
                as lists of strings
        """
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
        """
        Return a dictionary of strings that are 
        UNIX wild-card representations of a common
        id for all traces in this DictStream. I.e.,
            ? = single character wildcard
            * = unbounded character count widlcard

        :: OUTPUT ::
        :return out: [dict] dictionary of elements keyed
                    with the ID element name
        """
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
        """
        Get the UNIX wildcard formatted common reference_id string
        for all traces in this DictStream

        :: OUTPUT ::
        :return out: [str] output stream
        """
        ele = self.get_reference_id_elements()
        out = '.'.join(ele.values())
        return out                
    
    def split_on_key(self, key='instrument', **options):
        """
        Split this DictStream into a dictionary of DictStream
        objects based on a given element or elements of the
        constituient traces' ids.

        :: INPUTS ::
        :param key: [str] name of the attribute to split on
                    Supported:
                        'id', 'site','inst','instrument','mod','component',
                        'network','station','location','channel','model','weight'
        :param **options: [kwargs] key word argument gatherer to pass
                        kwargs to DictStream.__add__()
        :: OUTPUT ::
        :return out: [dict] of [DictStream] objects
        """
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
        """
        Split this DictStream by instrument codes (Net.Sta.Loc.BandInst)
        """
        
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



    





            
