import fnmatch, inspect, time
import numpy as np
from decorator import decorator
from obspy import Trace, Stream, UTCDateTime
from obspy.core.util.attribdict import AttribDict
from wyrm.core.trace import MLTrace, MLTraceBuffer
from wyrm.util.pyew import wave2mltrace
###################################################################################
# Dictionary Stream Header Class Definition #######################################
###################################################################################

class DictStreamHeader(AttribDict):
    defaults = {
        'common_id': '',
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
        # initialize as empty stream
        super().__init__()
        # Create stats attribute with DictStreamHeader
        self.stats = DictStreamHeader(header=header)
        # Redefine self.traces as dict
        self.traces = {}

        if traces is not None:
            self.__add__(traces, **options)

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
    
    def __add__(self, other, **options):
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
        elif isinstance(other, Stream):
            self.add_stream(other, **options)

        # Otherwise
        else:
            raise TypeError(f'other {type(other)} not supported.')
        
        if isinstance(other, MLTrace):
            # Get id of MLTrace "other"
            id = other.id
            site = other.site
            inst = other.inst
            mod = other.mod
            comp = other.comp
            # If id is not in traces.keys() - use dict.update
            if id not in self.traces.keys():
                self.traces.update({id: other})
            # If id is in traces.keys() - use __add__
            else:
                self.traces[id].__add__(other, **options)

    def add_stream(self, stream, **options):
        for _tr in stream:
            self.__add__(_tr, **options)

    def fnselect(self, fnstr, ascopy=False):
        matches = fnmatch.filter(self.traces.keys(), fnstr)
        dst = DictStream(header=self.stats.copy())
        for _m in matches:
            if ascopy:
                _tr = self.traces[_m].copy()
            else:
                _tr = self.traces[_m]
            dst.traces.update({_m: _tr})
        dst.stats.common_id = fnstr
        return dst

    def __str__(self):
        rstr = 'wyrm.core.data.MLStream()'
        return rstr

    def __repr__(self, extended=False):
        rstr = self.stats.__repr__()
        if len(self.traces) > 0:
            id_length = max(len(_tr.id) for _tr in self.traces.values())
        else:
            id_length=0
        rstr += f'\n{len(self.traces)} MLTrace(s) in DictStream\n'
        if len(self.traces) <= 20 or extended is True:
            for _l, _tr in self.traces.items():
                rstr += f'{_l:} : {_tr.__str__(id_length)}\n'
        else:
            _l0, _tr0 = list(self.traces.items())[0]
            _lf, _trf = list(self.traces.items())[-1]
            rstr += f'{_l0:} : {_tr0.__str__(id_length)}\n'
            rstr += f'...\n({len(self.traces) - 2} other traces)\n...\n'
            rstr += f'{_lf:} : {_trf.__str__(id_length)}\n'
            rstr += f'[Use "print(DictStream.__repr__(extended=True))" to print all labels and MLTraces]'
        return rstr


class