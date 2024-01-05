from wyrm.wyrms.wyrm import Wyrm
from wyrm.structures.rtinststream import RtInstStream
from wyrm.structures.rtbufftrace import RtBuffTrace
from wyrm.message.window import WindowMsg
from obspy import UTCDateTime
from collections import deque

class WindWyrm(Wyrm):
    
    def __init__(
            self,
            model_code='EQT',
            target_sr=100.,
            stride_pct=.3,
            blind_pct=1/12,
            taper_type='cosine',
            vert_comp_pct=.95,
            vert_codes='Z3',
            hztl_1c_pct=.8,
            hztl_codes='N1E2',
            ch_fill_rule='zeros',
            substr='*.*.*.*'
    ):
        Wyrm.__init__(self)
        
        # Compatability checks for model_code
        if isinstance(model_code, str):
            if model_code.upper() in ['EQT','EQTRANSFORMER']:
                self.model_code = 'EQT'
                self.wnpts = 6001
                self.comporder = 'Z3N1E2'
            elif model_code.upper() in ['PN','PNET','PHASENET']:
                self.model_code = 'PN'
                self.wnpts = 3001
                self.comporder = 'Z3N1E2'
            else:
                raise ValueError('Current supported model_code values are "EQT" and "PN"')
        else:
            raise TypeError('Current supported model_code values are "EQT" and "PN"')
        
        # Compatability checks for target_sr
        if isinstance(target_sr, int):
            self.target_sr = target_sr
        elif isinstance(target_sr, float):
            self.target_sr = int(target_sr)
        else:
            raise TypeError('target_sr must be int-like')
        if self.target_sr < 0:
            raise ValueError('target_sr must be positive')
        else:
            self.wsec = self.nwpts/self.target_sr
    
        # Compatability checks for stride_pct
        if isinstance(stride_pct, float):
            if 0 < stride_pct < 1:
                self.ssec = self.wsec*stride_pct
            else:
                raise ValueError('stride_pct must be \in (0, 1)')
        else:
            raise TypeError('stride_pct must be type float')
        
        # Compatability check for vert_comp_pct
        if isinstance(vert_comp_pct, float):
            if 0 < vert_comp_pct < 1:
                self.vcp_thresh = vert_comp_pct
            else:
                raise ValueError('vert_comp_pct must be \in (0, 1)')
        else:
            raise TypeError('vert_comp_pct must be type float')
        
        # Compatability check for hztl_1c_pct
        if isinstance(hztl_1c_pct, float):
            if 0 < hztl_1c_pct < 1:
                self.h1c_thresh = hztl_1c_pct
            else:
                raise ValueError('hztl_1c_pct must be \in (0, 1)')
        else:
            raise TypeError('vert_comp_pct must be type float')
        
        # Compatability check for ch_fill_rule
        if isinstance(ch_fill_rule, str):
            if ch_fill_rule in ['zeros','cloneZ','cloneHZ']:
                self.ch_fill_rule = ch_fill_rule
            else:
                raise ValueError('ch_fill_rule must be in: "zeros", "cloneZ", "cloneHZ"')
        else:
            raise TypeError('ch_fill_rule must be type str')

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

        self.index = {}
        self._template = {'next_starttime': None, '1C': False, 'pcomp': False}
        self.queue = deque([])

    def _assess_risbranch_windowing(self, risbranch, **kwargs):
        """
        Conduct assessment of window readiness and processing
        style for a given RtInstStream branch based on thresholds
        and 1C rules set for this WindWyrm

        :: INPUTS ::
        :param risbranch: [dict] of [RtBuffTrace] objects
                Realtime Instrument Stream branch
        :param **kwargs: key-word argment collector to pass
                to RtBuffTrace.get_window_stats()
                    starttime
                    endtime
                    pad
                    taper_sec
                    taper_type
                    vert_codes
                    hztl_codes

        
        :: OUTPUT ::
        :return pstats: [dict] dictionary with the following keyed fields
                '1C'     [bool]: Process as 1C data?
                'pcomp'  [list]: list of component codes to process starting
                                 with the vertical component that have passed
                                 validation.

            NOTE: {'1C': False, 'pcomp': False} indicates no valid window due to
                    absence of vertical component buffer
                  {'1C': True, 'pcomp': False} indicates no valid window due to
                    insufficient data on the vertical
        """
        # Create holder for branch member stats
        bstats = {'vcode': False, 'nbuff': len(risbranch)}
        # Get individual buffer stats
        for k2 in risbranch.keys():
            # Get RtBuffTrace.get_window_stats()
            stats = risbranch[k2].get_window_stats(**kwargs)
            bstats.update({k2:stats})
            # Snag code 
            if stats['comp_type'] == 'Vertical':
                bstats.update({'vcode': k2})
            elif stats['comp_type'] == 'Horizontal':
                if 'hcode' not in bstats.keys():
                    bstats.update({'hcodes':[k2]})
                else:
                    bstats['hcodes'].append(k2)
        
        # ### SENSE VERTICAL DATA PRESENT
        # TODO: a lot of this can get contracted out to WindowMsg!
        pstats = {}
        # if no vertical present, return bstats as-is
        if not bstats['vcode']:
            pstats.update({'1C': False, 'pcomp': False})
        # If vertical is present, proceed
        elif bstats['vcode']:
            # If there is not sufficient vertical data in the assessed window
            if bstats[bstats['vcode']]['percent_valid'] < self.vcp_thresh:
                pstats.update({'1C': True, 'pcomp': False})
            # If there is sufficient vertical data
            else:
                # If there is only a vertical, flag as ready, 1C
                if bstats['nbuff'] == 1:
                    pstats.update({'1C': True, 'pcomp': [bstats['vcode']]})
                # If there is at least one horizontal buffer
                elif bstats['nbuff'] == 2:
                    # If zero-pad or clone vertical 1c rule, flag as ready, 1C
                    if self.ch_fill_rule in ['zeros','cloneZ']:
                        pstats.update({'1C': True, 'pcomp': [bstats['vcode']]})
                    # If horizontal cloning
                    elif self.ch_fill_rule == 'cloneZH':
                        # If 
                        if bstats[bstats['hcodes'][0]]['percent_valid'] < self.h1c_thresh:
                            pstats.update({'1C': True, 'pcomp': [bstats['vcode']]})
                        else:
                            pstats.update({'1C': False, 'pcomp': [bstats['vcode']] + bstats['hcodes']})
                    else:
                        raise ValueError(f'ch_fill_rule {self.ch_fill_rule} incompatable')
                # If there are three buffers
                elif bstats['nbuff'] == 3:
                    # If both horizontals have sufficient data flag as ready, 3C
                    if all(bstats[_c]['percent_valid'] >= self.h1c_thresh for _c in bstats['hcodes']):
                        pstats.update({'1C': False, 'pcomp': [bstats['vcode']] + bstats['hcodes']})
                    # If one horizontal has sufficient data
                    elif any(bstats[_c]['percent_valid'] >= self.h1c_thresh for _c in bstats['hcodes']):
                        pstats.update({'1C': False, 'pcomp': [bstats['vcode']]})
                        # If clone horizontal ch_fill_rule
                        if self.ch_fill_rule == 'cloneZH':
                            for _c in bstats['hcodes']:
                                if bstats[_c]['percent_valid'] >= self.h1c_thresh:
                                    pstats['pcomp'].append(_c)
                        else:
                            pstats.update({'1C': True})
                    # If no horizontal has sufficient data
                    else:
                        pstats.update({'1C': True, 'pcomp': [bstats['vcode']]})
        return pstats
                        
    def window_rtinststream(self, rtinststream, **kwargs):
        nsubmissions = 0
        for k1 in rtinststream.keys():
            # Alias risbranch
            _risbranch = rtinststream[k1]
            # Create new template in index if k1 is not present
            if k1 not in self.index.keys():
                self.index.update({k1:{deepcopy(self._template)}})
                _idxbranch = self.index[k1]
            else:
                _idxbranch = self.index[k1]
            # # # ASSESS IF NEW WINDOW CAN BE GENERATED # # #
            # If next_starttime is still template value
            if isinstance(_idxbranch['next_starttime'], type(None)):
                # Search for vertical component
                for _c in self.vert_codes:
                    if _c in _risbranch.keys():
                        # If vertical component has any data
                        if len(_risbranch[_c]) > 0:
                            # Assign `next_starttime` using buffer starttime
                            _buff_ts = _risbranch[_c].stats.starttime
                            _idxbranch['next_starttime'] = _buff_ts
                            # break iteration loop
                            break

            # Handle case if next_starttime already assigned
            if isinstance(_idxbranch['next_starttime'], UTCDateTime):
                ts = _idxbranch['next_starttime']
                te = ts + self.wsec

                pstats = self._assess_risbranch_windowing(
                    _risbranch,
                    starttime=ts,
                    endtime=te,
                    vert_codes=self.vert_codes,
                    hztl_codes=self.hztl_codes,
                    taper_type=self.ttype,
                    taper_sec=self.blindsec)
                
                # Update with pstats
                _idxbranch.update(pstats)

            # If flagged for windowing
            if isinstance(_idxbranch['pcomp'], list):
                _pcomp = _idxbranch['pcomp']
                # If not 3-C
                if len(_pcomp) <= 2:
                    # Get vertical buffer
                    _buff = _risbranch[_pcomp[0]]
                    _trZ = _buff.as_trace().trim(
                        starttime=ts,
                        endtime=te,
                        pad=True,
                        nearest_sample=True)
                    # If zero-padding rule
                    if self.ch_fill_rule == 'zeros':
                        _trH = _trZ.copy()
                        _trH.data *= 0
                    # If cloning rule
                    elif self.ch_fill_rule in ['cloneZ', 'cloneHZ']:
                        # If only vertical available, clone vertical
                        if len(_pcomp) == 1:
                            _trH = _trZ.copy()
                        # If horziontal available
                        elif len(_pcomp) == 2:
                            # For cloneZ, still clone vertical
                            if self.ch_fill_rule == 'cloneZ':
                                _trH = _trZ.copy()                              
                            # Otherwise, clone horizontal
                            else:
                                _buff = _risbranch[_pcomp[1]]
                                _trH = _buff.as_trace().trim(
                                    starttime=ts,
                                    endtime=te,
                                    pad=True,
                                    nearest_sample=True)
                    # Compose window dictionary
                    window ={'Z': _trZ,
                             'N': _trH,
                             'E': _trH}
                # If 3C
                elif len(_pcomp)==3:
                    window = {}
                    for _c,_a in zip(_pcomp, ['Z','N','E']):
                        _buff = _risbranch[_c]
                        _tr = _buff.as_trace().trim(
                            starttime=ts,
                            endtime=te,
                            pad=True,
                            nearest_sample=True)
                        window.update({_a:_tr})
                # Append processing metadata to window
                window.update(deepcopy(_idxbranch))

                # Append window to output queue
                self.queue.appendleft(window)

                # Advance next_window by stride seconds
                _idxbranch['next_starttime'] += self.ssec

                # Increase index for submitted
                nsubmissions += 1
        return nsubmissions
    
    def pulse(self, x):
        """
        :: INPUT ::
        :param x: [RtInstStream] populated realtime instrument
                    stream object

        :: OUTPUT ::
        :return y: [deque]
        """
        for _ in self.maxiter:
            ns = self.window_rtinststream(x)
            if ns == 0:
                break
        y = self.queue
        return y

                    






