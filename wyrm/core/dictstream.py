import fnmatch, inspect
import numpy as np
import pandas as pd
import wyrm.util.compatability as wuc
from decorator import decorator
from copy import deepcopy
from obspy import Stream, UTCDateTime
from obspy.core.trace import Trace, Stats
from obspy.core.util.attribdict import AttribDict

class DictStreamHeader(AttribDict):
    defaults = {
        'common_id': '',
        'ref_starttime': None,
        'ref_sampling_rate': None,
        'ref_npts': None,
        'ref_model': None,
        'ref_weight': None,
        'sync_status': {'starttime': False, 'sampling_rate': False, 'npts': False},
        'processing': []
    }

    def __init__(self, header={}):
        super(DictStreamHeader, self).__init__(header)
    
    def __repr__(self):
        rstr = '----Stats----'
        for _k, _v in self.items():
            if _v is not None:
                rstr += f'\n{_k:>18}: '
                if _k == 'syncstatus':
                    for _k2, _v2 in _v.items():
                        rstr += f'\n{_k2:>26}: {_v2}'
                else:
                    rstr += f'{_v}'
        return rstr

class DictStream(Stream):

    def __init__(
            self,
            traces=None,
            header={},
            **options
    ):

        super().__init__(traces=traces)

        self.stats = DictStreamHeader(header=header)
        self.traces = {}
        self.siteinst_index = {}
        self.nsite = 0
        self.ninst = 0
        self.options = options

        if traces is not None:
            self.__add__(traces, **options)    
            self._update_siteinst_index() 
        
        # if self.is_syncd(run_assessment=True):
        #     print('Traces ready for tensor conversion')
        # else:
        #     print('Steps required to sync traces prior to tensor conversion')


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
            # Particular case for DictStream
            if isinstance(other, DictStream):
                iterant = other.values()
            else:
                iterant = other
        # Handle case where other is type-Trace (includes TraceBuffer)
        elif isinstance(other, Trace):
            # Make an iterable list of 1 item
            iterant = [other]

        ## UPDATE/ADD SECTION ##
        # Iterate across list-like of traces "iterant"
        for _tr in iterant:
            if _tr.id not in self.traces.keys():
                self.traces.update({_tr.id: _tr})
            else:
                self.traces[_tr.id].__add__(_tr, **options)
        self._update_siteinst_index()

    def _update_siteinst_index(self):
        new_index = {}
        nsite = 0
        ninst = 0
        for _tr in self.values():
            inst = _tr.stats.channel[:-1]
            site = f'{_tr.stats.network}.{_tr.stats.station}.{_tr.stats.location}'
            if site not in new_index.keys():
                new_index.update({site: [inst]})
                nsite += 1
                ninst += 1
            elif inst not in new_index[site]:
                new_index[site].append(inst)
                ninst += 1
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
        rstr += f'\n{len(self.traces)} Trace(s) in LabelStream from {self.ninst} instruments(s) across {self.nsite} stations(s):\n'
        if len(self.traces) <= 20 or extended is True:
            for _l, _tr in self.items():
                rstr += f'{_l:} : {_tr.__str__(id_length)}\n'
        else:
            _l0, _tr0 = list(self.items())[0]
            _lf, _trf = list(self.items())[-1]
            rstr += f'{_l0:} : {_tr0.__str__(id_length)}\n'
            rstr += f'...\n({len(self.traces) - 2} other traces)\n...\n'
            rstr += f'{_lf:} : {_trf.__str__(id_length)}\n'
            rstr += f'[Use "print(MLStream.__repr__(extended=True))" to print all labels and traces]'
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

    def _assess_starttime_sync(self, new_ref_starttime=False):
        if isinstance(new_ref_starttime, UTCDateTime):
            ref_t0 = new_ref_starttime
        else:
            ref_t0 = self.stats.ref_starttime
        
        if all(_tr.stats.starttime == ref_t0 for _tr in self.list_traces()):
            self.stats.sync_status['starttime'] = True
        else:
            self.stats.sync_status['starttime'] = False

    def _assess_sampling_rate_sync(self, new_ref_sampling_rate=False):
        if isinstance(new_ref_sampling_rate, float):

            ref_sampling_rate = new_ref_sampling_rate
        else:
            ref_sampling_rate = self.stats.ref_sampling_rate
        
        if all(_tr.stats.sampling_rate == ref_sampling_rate for _tr in self.list_traces()):
            self.stats.sync_status['sampling_rate'] = True
        else:
            self.stats.sync_status['sampling_rate'] = False

    def _assess_npts_sync(self, new_ref_npts=False):
        if isinstance(new_ref_npts, int):
            ref_npts = new_ref_npts
        else:
            ref_npts = self.stats.ref_npts
        
        if all(_tr.stats.npts == ref_npts for _tr in self.list_traces()):
            self.stats.sync_status['npts'] = True
        else:
            self.stats.sync_status['npts'] = False

    def is_syncd(self, run_assessment=True, **options):
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
            return True
        else:
            return False

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
        self.stats.processing.append(f'Wyrm 0.0.0: apply_trace_method("{method_name}",{args}, {kwargs})')
        return self


    def split_by_site(self):
        site_dict_streams = {}
        for _site in self.siteinst_index.keys():
            _dst = self.fnselect(_site+'.*')
            site_dict_streams.update({_site:_dst})
        return site_dict_streams
    
    def split_by_instrument(self, heirarchical=False):
        instrument_dict_streams = {}
        for _site, _insts in self.siteinst_index.items():
            for _inst in _insts:
                instcode = f'{_site}.{_inst}'
                _dst = self.fnselect(f'{_site}.{_inst}*')
                if heirarchical:
                    if _site not in instrument_dict_streams.keys():
                        instrument_dict_streams.update({_site:{_inst: _dst}})
                    elif _inst not in instrument_dict_streams[_site]:
                        instrument_dict_streams[_site].update({_inst: _dst})
                else:
                    if instcode not in instrument_dict_streams.keys():
                        instrument_dict_streams.update({instcode: _dst})
        return instrument_dict_streams

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

    def apply_fill_rule(self, ref_comp='Z', rule='zeros', ref_comp_thresh=0.95, other_comp_thresh=0.95, comp_map={'Z': 'Z3','N': 'N1', 'E': 'E2'}):
        self._update_siteinst_index()
        # If one site only
        if self.nsite == 1:
            pass
        else:
            raise ValueError('Fill rule can only be applied to a single station')
            # If one instrument only
        if self.ninst == 1:
            pass
        else:
            raise ValueError('Fill rule can only be applied to a single instrument')
        
        ref_code = None
        for _l, _tr in self.items():
            if _tr.stats.component in comp_map[ref_comp]:
                ref_code = _l
        if ref_code is not None:
            pass
        else:
            raise ValueError('Fill rule can only be applied if the reference trace/component is present')
        
        cidx = self.get_trace_completeness()
        if cidx[ref_code] >= ref_comp_thresh:
            pass
        else:
            raise ValueError('Insufficient unmasked data to accept the reference trace')
        
        if all(_cv >= other_comp_thresh for _cv in cidx.values) and len(self.traces) == 3:
            self.stats.processing.append('Wyrm 0.0.0: apply_fill_rule - 3-C data present')
            return self
        else:
            if rule == 'zeros':
                




    # def make_instrument_stream(self, instrument_code, ref_comp='Z', comp_map={"Z": 'Z3', "N": 'N1', "E": 'E2'} fill_rule='zeros'):

    #     ist = self.fnselect(instrument_code+'*')
    #     if all (instrument_code + _rc not in ist.labels() for _rc in comp_map[ref_comp]):
    #         raise ValueError(f'reference channel not found')
    #     else:
            



# class WindowStream(DictStream):

#     def __init__(self, ref_trace, other_traces=None, header={}):



        