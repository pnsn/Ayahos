from wyrm.wyrms.wyrm import Wyrm
from wyrm.structures.rtinststream import RtInstStream
from wyrm.structures.rtbufftrace import RtBuffTrace
from wyrm.message.window import WindowMsg
from obspy import UTCDateTime, Trace
from collections import deque
import numpy as np
from copy import deepcopy


class WindWyrm(Wyrm):
    """
    The WindWyrm class contains a dictionary of window-generation
    metadata for coordinating slicing (copying) data from buffers
    within a RtInstStream object to initialize WindowMsg objects.

    WindowMsg objects are staged in a queue for subsequent
    pre-processing and ML prediction.

    Each WindWyrm instance is specific to an ML model type.

    TODO:
     - either obsolite substr or use it to filter rtinststream's
     first layer of keys.
    """
    def __init__(
            self,
            model_code='EQT',
            target_npts=6000,
            target_sr=100.,
            stride_sec=18.,
            blind_sec=5.,
            taper_sec=.06,
            vert_comp_pct=.95,
            vert_codes='Z3',
            hztl_comp_pct=.8,
            hztl_codes='N1E2',
            window_fill_rule='zeros',
            substr='*.*.*.*',
            max_pulse_size=20,
            debug=False
    ):
        Wyrm.__init__(self, max_pulse_size=max_pulse_size, debug=debug)

        # Compatability checks for model_code
        if isinstance(model_code, str):
            if model_code.upper() in ['EQT', 'EQTRANSFORMER']:
                self.model_code = 'EQT'
            elif model_code.upper() in ['PN', 'PNET', 'PHASENET']:
                self.model_code = 'PN'
            else:
                raise ValueError('Current supported model_code values are "EQT" and "PN"')
        else:
            raise TypeError('Current supported model_code values are "EQT" and "PN"')
        ## RUN COMPATABILITY CHECKS WITH INHERITED METHODS FROM WYRM ##
        # Compatability checks for target_npts
        self.target_npts = self._bounded_intlike_check(target_npts, name="target_npts", minimum=1)
        # Compatability checks for target_sr
        self.target_sr = self._bounded_intlike_check(target_sr, name='target_sr', minimum=1)
        # Compatability checks for stride_sec
        self.stride_sec = self._bounded_floatlike_check(stride_sec, name='stride_sec', minimum=0.)
        # Compatability checks for blind_sec
        self.blind_sec = self._bounded_floatlike_check(blind_sec, name='blind_sec', minimum=0.)
        # Compatability checks for taper_sec
        self.taper_sec = self._bounded_floatlike_check(taper_sec, name='taper_sec', minimum=0.)
        # Compatability check for vert_comp_pct
        self.vcp_thresh = self._bounded_floatlike_check(vert_comp_pct, name='vert_comp_pct', minimum=0, maximum=1)
        # Compatability check for hztl_comp_pct
        self.hcp_thresh = self._bounded_floatlike_check(hztl_comp_pct, name='hztl_comp_pct', minimum=0, maximum=1)
        # Compatability check for vert_codes
        self.vert_codes = self._iterable_characters_check(vert_codes, name='vert_codes')
        # Compatability check for hztl_codes
        self.hztl_codes = self._iterable_characters_check(hztl_codes, name='hztl_codes')
        
        # Compatability check for window_fill_rule
        if isinstance(window_fill_rule, str):
            if window_fill_rule in ['zeros', 'cloneZ', 'cloneHZ']:
                self.window_fill_rule = window_fill_rule
            else:
                raise ValueError('window_fill_rule must be in: "zeros", "cloneZ", "cloneHZ"')
        else:
            raise TypeError('window_fill_rule must be type str')

        # Compatability check for substr
        if isinstance(substr, str):
            if len(substr.split('.')) == 4:
                self.substr = substr
            elif '*' in substr:
                self.substr = substr
            else:
                raise ValueError('substr should resemble some form of a 4-element, "." delimited string for SNIL/NSLI codes compatable with fnmatch.filter()')
        else:
            raise TypeError('substr must be type str')

        # Initialize default attributes
        self.index = {}
        self._template = {'next_starttime': None} # NOTE: Must have a non-UTCDateTime default value
        self.queue = deque([])

    def __str__(self):
        rstr = f'{super().__str__(self)}\n'
        rstr += f'Targets    | model: {self.model_code} | npts: {self.target_npts} | S/R: {self.target_sr} Hz\n'
        rstr += f'Timing     | stride: {self.stride_sec} sec | blind: {self.blind_sec} sec | taper: {self.tap_sec\n'
        rstr += f'Thresholds | Z completeness {self.vcp_thresh*100}% | H completeness {self.hcp_thresh}%\n'
        rstr += f'Comp Maps  | Z: {self.vert_codes} | H: {self.hztl_codes}\n'

    def _populate_windowmsg(self, ref_starttime=None, V0=None, H1=None, H2=None):
        windowmsg = WindowMsg(
            V0=V0,
            H1=H1,
            H2=H2,
            window_fill_rule=self.window_fill_rule,
            target_sr=self.target_sr,
            target_npts=self.target_npts,
            ref_starttime=ref_starttime,
            normtype=self.norm_type,
            tolsec=self.tolsec,
            tapsec=self.tapsec
            )
        return windowmsg

    def _branch2windowmsg(self, branch, window_ts, pad=True):
        windowmsg_inputs = {'V0': None,
                            'H1': None,
                            'H2': None,
                            'ref_starttime': window_ts}
        for _k2 in branch.keys():
            window_te = window_ts + (self.target_npts - 1)/self.target_sr
            i_buff_stat = branch[_k2].get_window_stats(
                starttime=window_ts,
                endtime=window_te,
                taper_sec=self.blind_sec,
                vert_codes=self.vert_codes,
                hztl_codes=self.hztl_codes,
                pad=pad)
            if i_buff_stat['comp_type'] == 'Vertical':
                if i_buff_stat['percent_valid'] >= self.vcp_thresh:
                    _tr = branch[_k2].to_trace().trim(
                        starttime=window_ts,
                        endtime=window_te,
                        pad=pad,
                        fill_value=None)
                    windowmsg_inputs.update({'V0': _tr})
            elif i_buff_stat['comp_type'] == 'Horizontal':
                if i_buff_stat['percent_valid'] >= self.hcp_thresh:
                    _tr = branch[_k2].to_trace().trim(
                        starttime=window_ts,
                        endtime=window_te,
                        pad=pad,
                        fill_value=None)
                    for _ik in ['H1', 'H2']:
                        if windowmsg_inputs[_ik] is None:
                            windowmsg_inputs.update({_ik: _tr})
        if isinstance(windowmsg_inputs['V0'], Trace):
            windowmsg = self._populate_windowmsg(**windowmsg_inputs)
        else:
            windowmsg = False
        return windowmsg
                    
    def _process_windows(self, rtinststream, pad=True):
        nnew = 0
        for _k1 in rtinststream.keys():
            _branch = rtinststream[_k1]
            # If this branch does not exist in the WindWyrm.index
            if _k1 not in self.index.keys():
                self.index.update({_k1: {deepcopy(self._template)}})
                _idx = self.index[_k1]
            # otherwise, alias matching index entry
            else:
                _idx = self.index[_k1]

            # If this branch has data for the first time
            if _idx['next_starttime'] == self._template['next_starttime']:
                # Iterate across approved vertical component codes
                for _c in self.vert_codes:
                    # If there is a match
                    if _c in _branch.keys():
                        # and the matching RtBuffTrace in the branch has data
                        if len(_branch[_c]) > 0:
                            # use the RtBuffTrace starttime to initialize the windowing index
                            _first_ts = _branch[_c].stats.starttime
                            _idx.update({'next_starttime': _first_ts})
                            # and break
                            break
            # Otherwise, if the index has a UTCDateTime starttime
            elif isinstance(_idx['next_starttime'], UTCDateTime):
                # Do the match to the vertical trace buffer again
                # Set initial None-Type values for window edges
                _data_ts = None
                for _c in self.vert_codes:
                    if _c in _branch.keys():
                        if len(_branch[_c]) > 0:
                            # Grab start and end times
                            _data_ts = _branch[_c].stats.startime
                            break
                # If vertical channel doesn't show up, warning and continue
                if _data_ts is None:
                    print(f'Error retrieving starttime from {_k1} vertical: {_branch.keys()}')
                    continue
                # If data buffer starttime is before or at next_starttime
                elif _data_ts <= _idx['next_starttime']:
                    pass
                # If update next_starttime if data buffer starttime is later
                # Simple treatment for a large gap of data.
                elif _data_ts > _idx['next_starttime']:
                    _idx.update({'next_starttime': _data_ts})
                # Attempt to generate window message from this branch
                ts = _idx['next_starttime']
                windowmsg = self._branch2windowmsg(_branch, ts, pad=pad)
                # If window is generated
                if windowmsg:
                    # Add WindowMsg to queue
                    self.queue.appendleft(windowmsg)
                    # update nnew index for iteration reporting
                    nnew += 1
                    # advance next_starttime for this index by the stride
                    _idx['next_starttime'] += self.stride_sec
                # If window is not generated, go to next instrument
                else:
                    continue
    
        return nnew

    def pulse(self, x):
        """
        Conduct up to the specified number of iterations of
        self._process_windows on an input RtInstStream object
        and return access to this WindWyrm's queue attribute

        :: INPUT ::
        :param x: [wyrm.structure.stream.RtInstStream]

        :: OUTPUT ::
        :return y: [deque] deque of WindowMsg objects
                    loaded with the appendleft() method, so
                    the oldest messages can be removed with
                    the pop() method in a subsequent step
        """
        for _ in range(self.max_pulse_size):
            nnew = self._process_windows(x)
            if nnew == 0:
                break
        # Return y as access to WindowWyrm.queue attribute
        y = self.queue
        return y
        







    # def _assess_risbranch_windowing(self, risbranch, **kwargs):
    #     """
    #     Conduct assessment of window readiness and processing
    #     style for a given RtInstStream branch based on thresholds
    #     and 1C rules set for this WindWyrm

    #     :: INPUTS ::
    #     :param risbranch: [dict] of [RtBuffTrace] objects
    #             Realtime Instrument Stream branch
    #     :param **kwargs: key-word argment collector to pass
    #             to RtBuffTrace.get_window_stats()
    #                 starttime
    #                 endtime
    #                 pad
    #                 taper_sec
    #                 taper_type
    #                 vert_codes
    #                 hztl_codes

    #     :: OUTPUT ::
    #     :return pstats: [dict] dictionary with the following keyed fields
    #             '1C'     [bool]: Process as 1C data?
    #             'pcomp'  [list]: list of component codes to process starting
    #                              with the vertical component that have passed
    #                              validation.

    #         NOTE: {'1C': False, 'pcomp': False} indicates no valid window due to
    #                 absence of vertical component buffer
    #               {'1C': True, 'pcomp': False} indicates no valid window due to
    #                 insufficient data on the vertical
    #     """
    #     # Create holder for branch member stats
    #     bstats = {'vcode': False, 'nbuff': len(risbranch)}
    #     # Get individual buffer stats
    #     for k2 in risbranch.keys():
    #         # Get RtBuffTrace.get_window_stats()
    #         stats = risbranch[k2].get_window_stats(**kwargs)
    #         bstats.update({k2:stats})
    #         # Snag code 
    #         if stats['comp_type'] == 'Vertical':
    #             bstats.update({'vcode': k2})
    #         elif stats['comp_type'] == 'Horizontal':
    #             if 'hcode' not in bstats.keys():
    #                 bstats.update({'hcodes':[k2]})
    #             else:
    #                 bstats['hcodes'].append(k2)
        
    #     # ### SENSE VERTICAL DATA PRESENT
    #     # TODO: a lot of this can get contracted out to WindowMsg!
    #     pstats = {}
    #     # if no vertical present, return bstats as-is
    #     if not bstats['vcode']:
    #         pstats.update({'1C': False, 'pcomp': False})
    #     # If vertical is present, proceed
    #     elif bstats['vcode']:
    #         # If there is not sufficient vertical data in the assessed window
    #         if bstats[bstats['vcode']]['percent_valid'] < self.vcp_thresh:
    #             pstats.update({'1C': True, 'pcomp': False})
    #         # If there is sufficient vertical data
    #         else:
    #             # If there is only a vertical, flag as ready, 1C
    #             if bstats['nbuff'] == 1:
    #                 pstats.update({'1C': True, 'pcomp': [bstats['vcode']]})
    #             # If there is at least one horizontal buffer
    #             elif bstats['nbuff'] == 2:
    #                 # If zero-pad or clone vertical 1c rule, flag as ready, 1C
    #                 if self.ch_fill_rule in ['zeros','cloneZ']:
    #                     pstats.update({'1C': True, 'pcomp': [bstats['vcode']]})
    #                 # If horizontal cloning
    #                 elif self.ch_fill_rule == 'cloneZH':
    #                     # If 
    #                     if bstats[bstats['hcodes'][0]]['percent_valid'] < self.h1c_thresh:
    #                         pstats.update({'1C': True, 'pcomp': [bstats['vcode']]})
    #                     else:
    #                         pstats.update({'1C': False, 'pcomp': [bstats['vcode']] + bstats['hcodes']})
    #                 else:
    #                     raise ValueError(f'ch_fill_rule {self.ch_fill_rule} incompatable')
    #             # If there are three buffers
    #             elif bstats['nbuff'] == 3:
    #                 # If both horizontals have sufficient data flag as ready, 3C
    #                 if all(bstats[_c]['percent_valid'] >= self.h1c_thresh for _c in bstats['hcodes']):
    #                     pstats.update({'1C': False, 'pcomp': [bstats['vcode']] + bstats['hcodes']})
    #                 # If one horizontal has sufficient data
    #                 elif any(bstats[_c]['percent_valid'] >= self.h1c_thresh for _c in bstats['hcodes']):
    #                     pstats.update({'1C': False, 'pcomp': [bstats['vcode']]})
    #                     # If clone horizontal ch_fill_rule
    #                     if self.ch_fill_rule == 'cloneZH':
    #                         for _c in bstats['hcodes']:
    #                             if bstats[_c]['percent_valid'] >= self.h1c_thresh:
    #                                 pstats['pcomp'].append(_c)
    #                     else:
    #                         pstats.update({'1C': True})
    #                 # If no horizontal has sufficient data
    #                 else:
    #                     pstats.update({'1C': True, 'pcomp': [bstats['vcode']]})
    #     return pstats
    
    # def window_rtinststream(self, rtinststream, **kwargs):
    #     nsubmissions = 0
    #     for k1 in rtinststream.keys():
    #         # Alias risbranch
    #         _risbranch = rtinststream[k1]
    #         # Create new template in index if k1 is not present
    #         if k1 not in self.index.keys():
    #             self.index.update({k1:{deepcopy(self._template)}})
    #             _idxbranch = self.index[k1]
    #         else:
    #             _idxbranch = self.index[k1]
    #         # # # ASSESS IF NEW WINDOW CAN BE GENERATED # # #
    #         # If next_starttime is still template value
    #         if isinstance(_idxbranch['next_starttime'], type(None)):
    #             # Search for vertical component
    #             for _c in self.vert_codes:
    #                 if _c in _risbranch.keys():
    #                     # If vertical component has any data
    #                     if len(_risbranch[_c]) > 0:
    #                         # Assign `next_starttime` using buffer starttime
    #                         _buff_ts = _risbranch[_c].stats.starttime
    #                         _idxbranch['next_starttime'] = _buff_ts
    #                         # break iteration loop
    #                         break

    #         # Handle case if next_starttime already assigned
    #         if isinstance(_idxbranch['next_starttime'], UTCDateTime):
    #             ts = _idxbranch['next_starttime']
    #             te = ts + self.wsec

    #             pstats = self._assess_risbranch_windowing(
    #                 _risbranch,
    #                 starttime=ts,
    #                 endtime=te,
    #                 vert_codes=self.vert_codes,
    #                 hztl_codes=self.hztl_codes,
    #                 taper_type=self.ttype,
    #                 taper_sec=self.blindsec)
                
    #             # Update with pstats
    #             _idxbranch.update(pstats)

    #         # If flagged for windowing
    #         if isinstance(_idxbranch['pcomp'], list):
    #             _pcomp = _idxbranch['pcomp']
    #             # If not 3-C
    #             if len(_pcomp) <= 2:
    #                 # Get vertical buffer
    #                 _buff = _risbranch[_pcomp[0]]
    #                 _trZ = _buff.as_trace().trim(
    #                     starttime=ts,
    #                     endtime=te,
    #                     pad=True,
    #                     nearest_sample=True)
    #                 # If zero-padding rule
    #                 if self.ch_fill_rule == 'zeros':
    #                     _trH = _trZ.copy()
    #                     _trH.data *= 0
    #                 # If cloning rule
    #                 elif self.ch_fill_rule in ['cloneZ', 'cloneHZ']:
    #                     # If only vertical available, clone vertical
    #                     if len(_pcomp) == 1:
    #                         _trH = _trZ.copy()
    #                     # If horziontal available
    #                     elif len(_pcomp) == 2:
    #                         # For cloneZ, still clone vertical
    #                         if self.ch_fill_rule == 'cloneZ':
    #                             _trH = _trZ.copy()                              
    #                         # Otherwise, clone horizontal
    #                         else:
    #                             _buff = _risbranch[_pcomp[1]]
    #                             _trH = _buff.as_trace().trim(
    #                                 starttime=ts,
    #                                 endtime=te,
    #                                 pad=True,
    #                                 nearest_sample=True)
    #                 # Compose window dictionary
    #                 window ={'Z': _trZ,
    #                          'N': _trH,
    #                          'E': _trH}
    #             # If 3C
    #             elif len(_pcomp)==3:
    #                 window = {}
    #                 for _c,_a in zip(_pcomp, ['Z','N','E']):
    #                     _buff = _risbranch[_c]
    #                     _tr = _buff.as_trace().trim(
    #                         starttime=ts,
    #                         endtime=te,
    #                         pad=True,
    #                         nearest_sample=True)
    #                     window.update({_a:_tr})
    #             # Append processing metadata to window
    #             window.update(deepcopy(_idxbranch))

    #             # Append window to output queue
    #             self.queue.appendleft(window)

    #             # Advance next_window by stride seconds
    #             _idxbranch['next_starttime'] += self.ssec

    #             # Increase index for submitted
    #             nsubmissions += 1
    #     return nsubmissions
    
    # def pulse(self, x):
    #     """
    #     :: INPUT ::
    #     :param x: [RtInstStream] populated realtime instrument
    #                 stream object

    #     :: OUTPUT ::
    #     :return y: [deque]
    #     """
    #     for _ in self.maxiter:
    #         ns = self.window_rtinststream(x)
    #         if ns == 0:
    #             break
    #     y = self.queue
    #     return y

                    






