from wyrm.dev.dictstream import DictStream, DictStreamHeader
from wyrm.data.mltrace import MLTrace

class WindowStreamHeader(DictStreamHeader):
    
    defaults = DictStreamHeader.defaults
    defaults.update({
        'ref_starttime': None,
        'ref_sampling_rate': None,
        'ref_npts': None,
        'ref_model': None,
        'ref_weight': None,
    })
   
    def __init__(self, header={}):
        super(WindowStreamHeader, self)


class WindowStream(DictStream):

    def __init__(self, traces=None, header={}):
        super().__init__(self, traces=traces)
        self.stats = WindowStreamHeader(header=header)
        for _l, _tr in self.items():
            if not isinstance(_tr, MLTrace):
                _tr = MLTrace().from_trace(_tr)



    # # INSTRUMENT LEVEL METHODS #

    # def assess_window_readiness(self, ref_comp='Z', ref_comp_thresh=0.95, comp_map={'Z': 'Z3','N': 'N1', 'E': 'E2'}):
    #     self._update_siteinst_index()
    #     # Check if one site only
    #     if self.nsite == 1:
    #         pass
    #     else:
    #         raise ValueError('Fill rule can only be applied to a single station')
    #     # Check if one instrument only
    #     if self.ninst == 1:
    #         pass
    #     else:
    #         raise ValueError('Fill rule can only be applied to a single instrument')
    #     # Check for reference component
    #     ref_code = None
    #     for _l, _tr in self.items():
    #         if _tr.stats.component in comp_map[ref_comp]:
    #             ref_code = _l
    #     if ref_code is not None:
    #         pass
    #     else:
    #         raise ValueError('Fill rule can only be applied if the reference trace/component is present')
    #     # Evaluate trace completeness
    #     cidx = self.get_trace_completeness()
    #     # Check if reference trace is sufficiently complete
    #     if cidx[ref_code] >= ref_comp_thresh:
    #         return ref_code
    #     else:
    #         return False




    # def apply_fill_rule(self, ref_comp='Z', rule='zeros', ref_comp_thresh=0.95, other_comp_thresh=0.95, comp_map={'Z': 'Z3','N': 'N1', 'E': 'E2'}):
    #     ref_code = self.assess_window_readiness(ref_comp=ref_comp, ref_comp_thresh=ref_comp_thresh, comp_map=comp_map)
    #     if ref_code:
    #         cidx = self.get_trace_completeness()
    #     else:
    #         raise ValueError('Reference trace has insufficient data to apply a fill rule')
    #     # If all traces meet the other_comp_thresh threshold and there are 3 traces
    #     if all(_cv >= other_comp_thresh for _cv in cidx.values) and len(self.traces) == 3:
    #         # Attach processing note that everything passed
    #         self.stats.processing.append('Wyrm 0.0.0: apply_fill_rule - 3-C data present')
    #         # Return self
    #         return self
    #     # If some piece was missed
    #     else:
            
    #         if rule == 'zeros':
    #             self._apply_zeros_fill_rule(ref_code)
    #         elif rule == 'clonez':
    #             self._apply_clonez_fill_rule(ref_code)
    #         elif rule == 'clonehz':
    #             self._apply_clonehz_fill_rule(ref_code, cidx, other_comp_thresh)
    #     return self
    

    # def _apply_zeros_fill_rule(self, ref_code):
    #     """
    #     -- PRIVATE METHOD --

    #     ASSUMING THAT ONE OR MORE TRACES ARE BELOW COMPLETENESS THRESHOLDS
    #     replace non-reference trace(s) with duplicates of the reference trace
    #     and with component codes of N and E and 0-valued data
    #      - After Retailleau et al. (2022)

    #     WARNING: This is conducted on data in-place. If you want to experiment
    #     with it's behavior, use dictstream.copy()._apply_clonez

    #     :: INPUT ::
    #     :param ref_code: [str] reference trace ID string - assumed to be a
    #                         Z component (or comparable SEED naming convention mapping)
    #                         (e.g., component 3)
    #     :: OUTPUT ::
    #     :return self:
    #     """

    #     if ref_code not in self.labels():
    #         raise ValueError(f'ref_code {ref_code} is not in the trace label set')
    #     # Iterate across all elements
    #     for _k in self.labels():
    #         if _k != ref_code:
    #             # Pop off non-reference trace
    #             _tr = self.pop(_k)
    #     # Append two 0-traces with N and E component codes
    #     _tr0 = self.traces[ref_code].copy()
    #     _tr0.data = np.zeros(shape=_tr0.data.shape, dtype=_tr0.data.dtype)
    #     _tr0.stats.channel = _tr0.stats.channel[:-1]
    #     for _comp in 'NE':
    #         _trx = _tr0.copy()
    #         _trx.stats.channel += _comp
    #         self.__add__(_trx)
    #     self.stats.processing.append(f'Wyrm 0.0.0: _apply_zeros_fill_rule({ref_code})')
    #     return self
    
    # def _apply_clonez_fill_rule(self, ref_code):
    #     """
    #     -- PRIVATE METHOD --

    #     ASSUMING THAT ONE OR MORE TRACES ARE BELOW COMPLETENESS THRESHOLDS
    #     replace non-reference trace(s) with duplicates of the reference trace
    #     and with component codes of N and E
    #      - After Ni et al. (2023)

    #     WARNING: This is conducted on data in-place. If you want to experiment
    #     with it's behavior, use dictstream.copy()._apply_clonez

    #     :: INPUT ::
    #     :param ref_code: [str] reference trace ID string - assumed to be a
    #                         Z component (or comparable SEED naming convention mapping)
    #                         (e.g., component 3)
    #     :: OUTPUT ::
    #     :return self:
    #     """

    #     if ref_code not in self.labels():
    #         raise ValueError(f'ref_code {ref_code} is not in the trace label set')
    #     # Iterate across all elements
    #     for _k in self.labels():
    #         if _k != ref_code:
    #             # Pop off non-reference trace
    #             _tr = self.pop(_k)
    #     # Append two Z-traces with N and E component codes
    #     _trC = self.traces[ref_code].copy()
    #     _trC.stats.channel = _trC.stats.channel[:-1]
    #     for _comp in 'NE':
    #         _trx = _trC.copy()
    #         _trx.stats.channel += _comp
    #         self.__add__(_trx)
    #     self.stats.processing.append(f'Wyrm 0.0.0: _apply_clonez_fill_rule({ref_code})')
    #     return self
    


    

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