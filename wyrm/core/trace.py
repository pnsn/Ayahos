import fnmatch, inspect
import numpy as np
import pandas as pd
import wyrm.util.compatability as wuc
from decorator import decorator
from copy import deepcopy
from obspy import Stream, UTCDateTime
from obspy.core.trace import Trace, Stats


class MLTraceStats(Stats):
    # set of read only attrs
    readonly = ['endtime']
    # default values
    defaults = {
        'sampling_rate': 1.0,
        'delta': 1.0,
        'starttime': UTCDateTime(0),
        'endtime': UTCDateTime(0),
        'npts': 0,
        'calib': 1.0,
        'network': '',
        'station': '',
        'location': '',
        'channel': '',
        'model': '',
        'weight': ''
    }
    # keys which need to refresh derived values
    _refresh_keys = {'delta', 'sampling_rate', 'starttime', 'npts'}
    # dict of required types for certain attrs
    _types = {
        'network': str,
        'station': str,
        'location': str,
        'channel': str,
        'model': str,
        'weight': str
    }

    def __init__(self, header={}):
        """
        """
        super(Stats, self).__init__(header)


    def __str__(self):
        """
        Return better readable string representation of Stats object.
        """
        priorized_keys = ['model','weight','station','channel', 'location', 'network',
                          'starttime', 'endtime', 'sampling_rate', 'delta',
                          'npts', 'calib']
        return self._pretty_str(priorized_keys)

@decorator
def _add_processing_info(func, *args, **kwargs):
    """
    This is a decorator that attaches information about a processing call as a string
    to the MLTrace.stats.processing and MLStream.stats.processing lists

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


class MLTrace(Trace):
    """
    Extend the obspy.core.trace.Trace class with additional stats entries for
            model [str] - name of ML model architecture (method)
            weight [str] - name of ML model weights (parameterization)
    that are incorporated into the trace id attribute as :
        Network.Station.Location.Channel.Model.Weight
    and provides additional options for how overlapping MLTrace objects are merged
    to conduct stacking of prediction time-series on the back-end of 
    """
    def __init__(self, data=None, header=None):
        super().__init__(data=data)
        if header is None:
            header = {}
        header = deepcopy(header)
        header.setdefault('npts', len(data))
        self.stats = MLTraceStats(header)
        super(MLTrace, self).__setattr__('data', data)
        # if self.data is not None:
        #     self.fold = np.ones(shape=self.data.shape, dtype=self.data.dtype)
        # else:
        #     self.fold = None
    
    # def __add__(self, trace, method='merge', **options)
    #     if method == 'merge':
    #         self._merge(self, trace, **options)
    #     elif method == 'stack':
    #         self._stack(trace, **options)
    #     else:
    #         raise ValueError(f'method {method} not supported. Only "merge" and "stack"')

    def _merge(self, trace, method=1, interpolation_sampeles=-1, fill_value=)


    def get_ml_id(self):
        id_str = super().get_id()
        if self.stats.model is not None:
            id_str += f'.{self.stats.model}'
        else:
            id_str += '.'
        if self.stats.weight is not None:
            id_str += f'.{self.stats.weight}'
        else:
            id_str += '.'
        return id_str
    
    def get_id(self):
        return super().get_id()

    def get_instrument_code(self):
        inst_code = self.stats.channel[:-1]
        return inst_code
    
    @_add_processing_info
    def set_component(self, new_label):
        if isinstance(new_label, str):
            if len(new_label) == 1:
                self.stats.channel = self.instrument + new_label
            elif len(new_label) > 1:
                self.stats.channel = self.instrument + new_label[0]
            else:
                raise ValueError('str-type new_label must be a non-empty string')
        elif isinstance(new_label, int):
            if new_label < 10:
                self.stats.channel = f'{self.instrument}{new_label}'
            else:
                raise ValueError('int-type new_label must be a single-digit integer') 
        else:
            raise TypeError('new_label must be type str or int')

    trace_id = property(get_id)
    id = property(get_ml_id)
    instrument= property(get_instrument_code)


    def treat_gaps(
            self,
            merge_kwargs={},
            detrend_kwargs={'type': 'linear'},
            filter_kwargs=False,
            taper_kwargs={'max_percentage': None, 'max_length': 0.06, 'side':'both'}
            ):
        self.stats.processing.append('vvv Wyrm 0.0.0: treat_gaps vvv')
        if np.ma.is_masked(self.data):
            self = self.split()
            if detrend_kwargs:
                self.detrend(**detrend_kwargs)
            if filter_kwargs:
                self.filter(**filter_kwargs)
            if taper_kwargs:
                self.taper(**taper_kwargs)
            self.merge(**merge_kwargs)
            if isinstance(self, Stream):
                self = self[0]
        else:
            if detrend_kwargs:
                self.detrend(**detrend_kwargs)
            if filter_kwargs:
                self.filter(**filter_kwargs)
            if taper_kwargs:
                self.taper(**taper_kwargs)
        self.stats.processing.append('^^^ Wyrm 0.0.0: treat_gaps ^^^')
        return self

    def treat_gappy_trace(
            self,
            label,
            merge_kwargs={},
            detrend_kwargs=False,
            filter_kwargs=False,
            taper_kwargs=False
            ):

        _x = self.traces[label].copy()
        if np.ma.is_masked(_x.data):
            _x = _x.split()
            if detrend_kwargs:
                _x.detrend(**detrend_kwargs)
            if filter_kwargs:
                _x.filter(**filter_kwargs)
            if taper_kwargs:
                _x.taper(**taper_kwargs)
            _x.merge(**merge_kwargs)
            if isinstance(_x, Stream):
                _x = _x[0]
        else:
            if detrend_kwargs:
                _x.detrend(**detrend_kwargs)
            if filter_kwargs:
                _x.filter(**filter_kwargs)
            if taper_kwargs:
                _x.taper(**taper_kwargs)
        self.traces.update({label: _x})
        return self




    # def check_sync(self, level='summary'):
    #     # Use sync t0, if available, as reference
    #     if self.sync_t0:
    #         ref_t0 = self.sync_t0
    #     # Otherwise use specified ref_t0 if initially stated (or updated)
    #     elif self.ref_t0 is not None:
    #         ref_t0 = self.ref_t0
    #     # Otherwise use the starttime of the first trace in the Stream
    #     else:
    #         ref_t0 = list(self.traces.values())[0].stats.starttime
    #     # Use sync sampling_rate, if available, as reference
    #     if self.sync_sampling_rate:
    #         ref_sampling_rate = self.sync_sampling_rate
    #     # Otherwise use specified ref_sampling_rate if initially stated (or updated)
    #     elif self.ref_sampling_rate is not None:
    #         ref_sampling_rate = self.ref_sampling_rate
    #     # Otherwise use the starttime of the first trace in the Stream
    #     else:
    #         ref_sampling_rate = list(self.traces.values())[0].stats.starttime
        
    #     # Use sync npts, if available, as reference
    #     if self.sync_npts:
    #         ref_npts = self.sync_npts
    #     # Otherwise use specified ref_npts if initially stated (or updated)
    #     elif self.ref_npts is not None:
    #         ref_npts = self.ref_npts
    #     # Otherwise use the starttime of the first trace in the Stream
    #     else:
    #         ref_npts = list(self.traces.values())[0].stats.npts

    #     trace_bool_holder = {}
    #     for _l, _tr in self.traces.items():
    #         attr_bool_holder = [
    #             _tr.stats.starttime == ref_t0,
    #             _tr.stats.sampling_rate == ref_sampling_rate,
    #             _tr.stats.npts == ref_npts
    #         ]
    #         trace_bool_holder.update({_l: attr_bool_holder})

    #     df_bool = pd.DataFrame(trace_bool_holder, index=['t0','sampling_rate','npts']).T
    #     if level.lower() == 'summary': 
    #         status = df_bool.all(axis=0).all(axis=0)
    #     elif level.lower() == 'trace':
    #         status = df_bool.all(axis=1)
    #     elif level.lower() == 'attribute':
    #         status = df_bool.all(axis=0)
    #     elif level.lower() == 'debug':
    #         status = df_bool
    #     return status








# class MLStreamStats(AttribDict):

#     # readonly = ['sync_status','common_id']
#     defaults = {
#         'common_id': None,
#         'ref_starttime': None,
#         'ref_sampling_rate': None,
#         'ref_npts': None,
#         'ref_model': None,
#         'ref_weight': None,
#         'sync_status': {'starttime': False, 'sampling_rate': False, 'npts': False},
#         'processing': []
#     }

#     def __init__(self, header={}):
#         super(MLStreamStats, self).__init__(header)

#     def __repr__(self):
#         rstr = '----Stats----'
#         for _k, _v in self.items():
#             rstr += f'\n{_k:>18}: '
#             if _k == 'syncstatus':
#                 for _k2, _v2 in _v.items():
#                     rstr += f'\n{_k2:>26}: {_v2}'
#             else:
#                 rstr += f'{_v}'
#         return rstr















# class MLStream(Stream):
#     """
#     Adapted version of the obspy.core.stream.Stream class that uses a dictionary as the holder
#     for Trace-type objects to leverage hash-table search acceleration for large collections of
#     traces.
#     """
#     def __init__(
#             self,
#             traces=None,
#             header={},
#             **options):
#         """
#         Initialize a MLStream object

#         :: INPUTS ::
#         :param traces: [None] - returns an empty dict as self.traces
#                        [obspy.core.stream.Stream] or [list-like] of [obspy.core.trace.Trace]
#                             populates self.traces using each constituient
#                             trace's trace.id attribute as the trace label 
#                             (dictionary key).

#                             If there are multiple traces with the same trace.id
#                             the self.traces['trace.id'] subsequent occurrences
#                             of traces with the same id are appended to the
#                             initial trace using the initial Trace object's 
#                             __add__ ~magic~ method (i.e,. )

#                             In the case of an obspy.core.trace.Trace object, this
#                             results in an in-place execution of the __add__ method
#                             that calls Trace.merge() that is passed **options

#                             In the case of a wyrm.core.data.TraceBuffer, this 
#                             results in an application of the TraceBuffer.append()
#                             method, which has some additional restrictions on the
#                             timing of packets.

#         :param header: [dict] - dictionary passed to WindowStats.__init__ for
#                             defining reference values for this collection of traces
#                             i.e., target values for data attributes
#                             Supported keys:
#                                 'ref_starttime' [float] or [UTCDateTime] target starttime for each trace
#                                 'ref_sampling_rate' [float] target sampling rate for each trace
#                                 'ref_npts' [int] target number of samples in each trace
#                                 'ref_model' [str] name of model architecture associated with thi MLStream
#                                 'ref_weight': [str] name of model parameterization associated with this MLStream

#         :param **options: [kwargs] key-word arguments passed to Trace-like's __add__ method in the
#                                 event of multiple matching trace id's
        
#         """

#         super().__init__(traces=traces)

#         self.stats = MLStreamStats(header=header)
#         self.traces = {}
#         self.options = options

#         if traces is not None:
#             self.__add__(traces, **options)
        
#         if self.is_syncd(run_assessment=True):
#             print('Traces ready for tensor conversion')
#         else:
#             print('Steps required to sync traces prior to tensor conversion')

#     def __add__(self, other, **options):
#         """
#         Add a Trace-type or Stream-type object to this MLStream
#         """
#         # If adding a list-like
#         if isinstance(other, (list, tuple)):
#             # Check that each entry is a trace
#             for _tr in other:
#                 # Raise error if not
#                 if not isinstance(_tr, Trace):
#                     raise TypeError('all elements in traces must be type obspy.core.trace.Trace')
#                 # Use Trace ID as label
#                 else:
#                     iterant = other
#         # Handle case where other is Stream-like
#         elif isinstance(other, Stream):
#             # Particular case for MLStream
#             if isinstance(other, MLStream):
#                 iterant = other.values()
#             else:
#                 iterant = other
#         # Handle case where other is type-Trace (includes TraceBuffer)
#         elif isinstance(other, Trace):
#             # Make an iterable list of 1 item
#             iterant = [other]
#         # Iterate across list-like of traces "iterant"
#         for _tr in iterant:
#             if _tr.id not in self.traces.keys():
#                 self.traces.update({_tr.id: MLTrace(data=_tr.data, header=_tr.stats)})
#             else:
#                 self.traces[_tr.id].__add__(MLTrace(data=_tr.data, header=_tr.stats), **options)


    
#     @_add_processing_info
#     def fnselect(self, fnstr='*'):
#         """
#         Create a copy of this MLStream with trace ID's (keys) that conform to an input
#         Unix wildcard-compliant string. This updates the self.stats.common_id attribute

#         :: INPUT ::
#         :param fnstr: [str] search string to search with
#                         fnmatch.filter(self.traces.keys(), fnstr)

#         :: OUTPUT ::
#         :return lst: [wyrm.core.data.MLStream] subset copy
#         """
#         matches = fnmatch.filter(self.traces.keys(), fnstr)
#         lst = self.copy_dataless()
#         for _m in matches:
#             lst.traces.update({_m: self.traces[_m].copy()})
#         lst.stats.common_id = fnstr
#         return lst
    
#     def __repr__(self, extended=False):
#         rstr = self.stats.__repr__()
#         if len(self.traces) > 0:
#             id_length = max(len(_tr.id) for _tr in self.traces.values())
#         else:
#             id_length=0
#         rstr += f'\n{len(self.traces)} Trace(s) in LabelStream:\n'
#         if len(self.traces) <= 20 or extended is True:
#             for _l, _tr in self.items():
#                 rstr += f'{_l:} : {_tr.__str__(id_length)}\n'
#         else:
#             _l0, _tr0 = list(self.items())[0]
#             _lf, _trf = list(self.items())[-1]
#             rstr += f'{_l0:} : {_tr0.__str__(id_length)}\n'
#             rstr += f'...\n({len(self.traces) - 2} other traces)\n...'
#             rstr += f'{_lf:} : {_trf.__str__(id_length)}\n'
#             rstr += f'[Use "print(MLStream.__repr__(extended=True))" to print all labels and traces]'
#         return rstr
    
#     def __str__(self):
#         rstr = 'wyrm.core.data.MLStream()'
#         return rstr


#     def _assess_starttime_sync(self, new_ref_starttime=False):
#         if isinstance(new_ref_starttime, UTCDateTime):
#             ref_t0 = new_ref_starttime
#         else:
#             ref_t0 = self.stats.ref_starttime
        
#         if all(_tr.stats.starttime == ref_t0 for _tr in self.list_traces()):
#             self.stats.sync_status['starttime'] = True
#         else:
#             self.stats.sync_status['starttime'] = False

#     def _assess_sampling_rate_sync(self, new_ref_sampling_rate=False):
#         if isinstance(new_ref_sampling_rate, float):

#             ref_sampling_rate = new_ref_sampling_rate
#         else:
#             ref_sampling_rate = self.stats.ref_sampling_rate
        
#         if all(_tr.stats.sampling_rate == ref_sampling_rate for _tr in self.list_traces()):
#             self.stats.sync_status['sampling_rate'] = True
#         else:
#             self.stats.sync_status['sampling_rate'] = False

#     def _assess_npts_sync(self, new_ref_npts=False):
#         if isinstance(new_ref_npts, int):
#             ref_npts = new_ref_npts
#         else:
#             ref_npts = self.stats.ref_npts
        
#         if all(_tr.stats.npts == ref_npts for _tr in self.list_traces()):
#             self.stats.sync_status['npts'] = True
#         else:
#             self.stats.sync_status['npts'] = False

#     def is_syncd(self, run_assessment=True, **options):
#         if run_assessment:
#             if 'new_ref_starttime' in options.keys():
#                 self._assess_starttime_sync(new_ref_starttime=options['new_ref_starttime'])
#             else:
#                 self._assess_starttime_sync()
#             if 'new_ref_sampling_rate' in options.keys():
#                 self._assess_sampling_rate_sync(new_ref_sampling_rate=options['new_ref_sampling_rate'])
#             else:
#                 self._assess_sampling_rate_sync()
#             if 'new_ref_npts' in options.keys():
#                 self._assess_npts_sync(new_ref_npts=options['new_ref_npts'])
#             else:
#                 self._assess_npts_sync()
#         if all(self.stats.sync_status.values()):
#             return True
#         else:
#             return False

#     def diagnose_sync(self):
#         """
#         Produce a pandas.DataFrame that contains the sync status
#         for the relevant sampling reference attriburtes:
#             starttime
#             sampling_rate
#             npts
#         where 
#             True indicates a trace (id in index) matches the
#                  reference attribute (column) value
#             False indicates the trace mismatches the reference
#                     attribute value
#             'NoRefVal' indicates no reference value was available

#         :: OUTPUT ::
#         :return out: [pandas.dataframe.DataFrame] with sync
#                     assessment values.
#         """

#         index = []; columns=['starttime','sampling_rate','npts'];
#         holder = []
#         for _l, _tr in self.items():
#             index.append(_l)
#             line = []
#             for _f in columns:
#                 if self.stats['ref_'+_f] is not None:
#                     line.append(_tr.stats[_f] == self.stats['ref_'+_f])
#                 else:
#                     line.append('NoRefVal')
#             holder.append(line)
#         out = pd.DataFrame(holder, index=index, columns=columns)
#         return out


#     def check_sync(self, level='summary'):
#         # Use sync t0, if available, as reference
#         if self.sync_t0:
#             ref_t0 = self.sync_t0
#         # Otherwise use specified ref_t0 if initially stated (or updated)
#         elif self.ref_t0 is not None:
#             ref_t0 = self.ref_t0
#         # Otherwise use the starttime of the first trace in the Stream
#         else:
#             ref_t0 = list(self.traces.values())[0].stats.starttime
#         # Use sync sampling_rate, if available, as reference
#         if self.sync_sampling_rate:
#             ref_sampling_rate = self.sync_sampling_rate
#         # Otherwise use specified ref_sampling_rate if initially stated (or updated)
#         elif self.ref_sampling_rate is not None:
#             ref_sampling_rate = self.ref_sampling_rate
#         # Otherwise use the starttime of the first trace in the Stream
#         else:
#             ref_sampling_rate = list(self.traces.values())[0].stats.starttime
        
#         # Use sync npts, if available, as reference
#         if self.sync_npts:
#             ref_npts = self.sync_npts
#         # Otherwise use specified ref_npts if initially stated (or updated)
#         elif self.ref_npts is not None:
#             ref_npts = self.ref_npts
#         # Otherwise use the starttime of the first trace in the Stream
#         else:
#             ref_npts = list(self.traces.values())[0].stats.npts

#         trace_bool_holder = {}
#         for _l, _tr in self.traces.items():
#             attr_bool_holder = [
#                 _tr.stats.starttime == ref_t0,
#                 _tr.stats.sampling_rate == ref_sampling_rate,
#                 _tr.stats.npts == ref_npts
#             ]
#             trace_bool_holder.update({_l: attr_bool_holder})

#         df_bool = pd.DataFrame(trace_bool_holder, index=['t0','sampling_rate','npts']).T
#         if level.lower() == 'summary': 
#             status = df_bool.all(axis=0).all(axis=0)
#         elif level.lower() == 'trace':
#             status = df_bool.all(axis=1)
#         elif level.lower() == 'attribute':
#             status = df_bool.all(axis=0)
#         elif level.lower() == 'debug':
#             status = df_bool
#         return status

#     @_add_processing_info

    
#     @_add_processing_info
#     def treat_gappy_traces(
#         self, 
#         detrend_kwargs={'type':'linear'},
#         merge_kwargs={'method': 1, 'fill_value': 0, 'interpolation_samples': -1},
#         filter_kwargs={'type': 'bandpass', 'freqmin': 1, 'freqmax': 45},
#         taper_kwargs={ 'max_percentage': None, 'max_length': 0.06, 'side': 'both'},
#     ):
#         for _tr in self.traces.values():
#             _tr.treat_gaps(merge_kwargs=merge_kwargs,
#                            detrend_kwargs=detrend_kwargs
#                            filter_kwargs=filter_kwargs,
#                            taper_kwargs=taper_kwargs)
#         self.stats.processing.append('Wyrm 0.0.0: treat_gappy_traces')
#         return self

#     @_add_processing_info



    # def append_trace(self, trace, label, merge_kwargs={'method': 1, 'fill_value': None, 'pad': True, 'interpolation_samples': -1}):
    #     if label in self.traces.keys():
    #         _tr = self.traces[label].copy()
    #         if _tr.id == trace.id:
    #             try:
    #                 _st = Stream(traces=[_tr, trace]).merge(**merge_kwargs)
    #                 if len(_st) == 1:
    #                     _tr = _st[0]
    #                     self.traces.update({label: _tr})
    #                 else:
    #                     raise AttributeError('traces failed to merge, despite passing compatability tests')
    #             except:
    #                 raise AttributeError('traces fail merge compatability tests')
    #         else:
    #             raise AttributeError('trace.id and self.trace[label].id do not match')
    #     else:
    #         self.traces.update({label: trace})
    #     return self
    
    # def append_stream(self, stream, labels, merge_kwargs={'method': 1, 'fill_value': None, 'pad': True, 'interpolation_samples': -1})


    # def sync_data(self):

        


    # def treat_gaps(self, )

    # def apply_stream_method(self, method, *args, **kwargs)