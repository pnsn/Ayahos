import fnmatch, re
from obspy import Stream, Trace, UTCDateTime

class LabeledStream(Stream):

    def __init__(
            self,
            traces=None,
            ref_id=None,
            ref_model=None,
            ref_t0=None,
            ref_sampling_rate=None,
            ref_npts=None):
        super().__init__(traces=traces)
        if isinstance(traces, list):
            for _tr in traces:
                if not isinstance(_tr, Trace):
                    raise TypeError('all elements in traces must be type obspy.core.trace.Trace')
                
            self.traces = dict(zip([_tr.id for _tr in self.traces], self.traces))
        elif isinstance(traces, dict):
            self.traces = traces
        elif isinstance(traces, Trace):
            self.traces = {traces.id: traces}
        else:
            self.traces = {}

        self.sync_t0 = False
        self.sync_npts = False
        self.sync_sampling_rate = False
        # reference values
        self.ref_id = ref_id
        self.ref_model = ref_model
        self.ref_t0 = ref_t0
        self.ref_sampling_rate = ref_sampling_rate
        self.ref_npts = ref_npts

    def copy(self):
        return deepcopy(self)
    
    def values(self):
        return self.traces.values()
    
    def keys(self):
        return self.traces.keys()
    
    def items(self):
        return self.traces.items()
    
    def pop(self):
        return self.traces.popitem()
    
    def copy_dataless(self):
        lst = LabeledStream()
        lst.sync_t0 = self.sync_t0
        lst.sync_sampling_rate = self.sync_sampling_rate
        lst.sync_npts = self.sync_npts
        lst.ref_t0 = self.ref_t0
        lst.ref_sampling_rate = self.ref_sampling_rate
        lst.ref_npts = self.ref_npts
        return lst         
        
    def fnselect(self, fnstr='*'):
        matches = fnmatch.filter(self.traces.keys(), fnstr)
        lst = self.copy_dataless()
        breakpoint()
        for _m in matches:
            lst.traces.update({_m, self.traces[_m]})
        lst.ref_id = fnstr
        return lst
    
    def __repr__(self, extended=False):
        if len(self.traces) > 0:
            label_length = max(len(_l) for _l in self.traces.keys())
            id_length = max(len(_tr.id) for _tr in self.traces.values())
        else:
            label_length = 0
            id_length=0
        rstr = f'{len(self.traces)} Trace(s) in LabelStream:\n'
        if len(self.traces) <= 20 or extended is True:
            rstr = rstr + [f'{_l:} : {_tr.__str__(id_length)}' for _l, _tr in self.items()]
        else:
            _l0, _tr0 = list(self.items())[0]
            _lf, _trf = list(self.items())[-1]
            rstr += f'{_l0:} : {_tr0.__str__(id_length)}\n'
            rstr += f'...\n({len(self.traces) - 2} other traces)\n...'
            rstr += f'{_lf:} : {_trf.__str__(id_length)}\n'
            rstr += f'[Use "print(LabeledStream.__repr__(extended=True))" to print all labels and traces]'
        return rstr

    def check_sync(self, level='summary'):
        # Use sync t0, if available, as reference
        if self.sync_t0:
            ref_t0 = self.sync_t0
        # Otherwise use specified ref_t0 if initially stated (or updated)
        elif self.ref_t0 is not None:
            ref_t0 = self.ref_t0
        # Otherwise use the starttime of the first trace in the Stream
        else:
            ref_t0 = list(self.traces.values())[0].stats.starttime
        # Use sync sampling_rate, if available, as reference
        if self.sync_sampling_rate:
            ref_sampling_rate = self.sync_sampling_rate
        # Otherwise use specified ref_sampling_rate if initially stated (or updated)
        elif self.ref_sampling_rate is not None:
            ref_sampling_rate = self.ref_sampling_rate
        # Otherwise use the starttime of the first trace in the Stream
        else:
            ref_sampling_rate = list(self.traces.values())[0].stats.starttime
        
        # Use sync npts, if available, as reference
        if self.sync_npts:
            ref_npts = self.sync_npts
        # Otherwise use specified ref_npts if initially stated (or updated)
        elif self.ref_npts is not None:
            ref_npts = self.ref_npts
        # Otherwise use the starttime of the first trace in the Stream
        else:
            ref_npts = list(self.traces.values())[0].stats.npts

        trace_bool_holder = {}
        for _l, _tr in self.traces.items():
            attr_bool_holder = [
                _tr.stats.starttime == ref_t0,
                _tr.stats.sampling_rate == ref_sampling_rate,
                _tr.stats.npts == ref_npts
            ]
            trace_bool_holder.update({_l: attr_bool_holder})

        df_bool = pd.DataFrame(trace_bool_holder, index=['t0','sampling_rate','npts']).T
        if level.lower() == 'summary': 
            status = df_bool.all(axis=0).all(axis=0)
        elif level.lower() == 'trace':
            status = df_bool.all(axis=1)
        elif level.lower() == 'attribute':
            status = df_bool.all(axis=0)
        elif level.lower() == 'debug':
            status = df_bool
        return status

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
        for _l, _tr in self.traces.items():
            eval(f'_tr.{method_name}(*args, **kwargs)')
        return self

    def treat_gaps(
            self, 
            merge_kwargs={
                'method': 1,
                'fill_value': 0,
                'interpolation_samples': -1
            },
            detrend_kwargs={
                'method':'linear'
                },
            filter_kwargs = {
                'type': 'bandpass',
                'freqmin': 1,
                'freqmax': 45
                },
            taper_kwargs={
                'max_percentage': None,
                'max_length': 0.06,
                'side': 'both'
                }):
        for _l in self.traces.keys():
            self.treat_gappy_trace(
                _l,
                merge_kwargs=merge_kwargs,
                detrend_kwargs=detrend_kwargs,
                filter_kwargs=filter_kwargs,
                taper_kwargs=taper_kwargs
                )
        return self

    def treat_gappy_trace(
            self,
            label,
            merge_kwargs,
            detrend_kwargs,
            filter_kwargs,
            taper_kwargs
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


    def append_trace(self, trace, label, merge_kwargs={'method': 1, 'fill_value': None, 'pad': True, 'interpolation_samples': -1}):
        if label in self.traces.keys():
            _tr = self.traces[label].copy()
            if _tr.id == trace.id:
                try:
                    _st = Stream(traces=[_tr, trace]).merge(**merge_kwargs)
                    if len(_st) == 1:
                        _tr = _st[0]
                        self.traces.update({label: _tr})
                    else:
                        raise AttributeError('traces failed to merge, despite passing compatability tests')
                except:
                    raise AttributeError('traces fail merge compatability tests')
            else:
                raise AttributeError('trace.id and self.trace[label].id do not match')
        else:
            self.traces.update({label: trace})
        return self
    
    # def append_stream(self, stream, labels, merge_kwargs={'method': 1, 'fill_value': None, 'pad': True, 'interpolation_samples': -1})


    # def sync_data(self):

        


    # def treat_gaps(self, )

    # def apply_stream_method(self, method, *args, **kwargs)