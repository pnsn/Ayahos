from wyrm.wyrms.wyrm import Wyrm
from wyrm.structures.rtinststream import RtInstStream
from wyrm.structures.rtbufftrace import RtBuffTrace

class WindWyrm(Wyrm):
    
    def __init__(
            self,
            model_code='EQT',
            target_sr=100.,
            stride_pct=.3,
            vert_comp_pct=.95,
            hztl_1c_pct=.8,
            rule_1c='zeros',
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
        
        # Compatability check for rule_1c
        if isinstance(rule_1c, str):
            if rule_1c in ['zeros','cloneZ','cloneHZ']:
                self.rule_1c = rule_1c
            else:
                raise ValueError('rule_1c must be in: "zeros", "cloneZ", "cloneHZ"')
        else:
            raise TypeError('rule_1c must be type str')

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
        self._template = {'next_ts': None, '1C': False, 'z_code': False}

    def _index_rtinststream_branches(self, rtinststream):
        if not isinstance(rtinststream, RtInstStream):
            raise TypeError('rtinststream must be a wyrm.structures.rtinststream.RtInstStream')
        else:
            pass
    
        # Iterate across INST codes
        for k1 in rtinststream.keys():
            # Alias branch
            _rtsbranch = rtinststream[k1]
            
            # Initial checks for populating new self.index entry
            if k1 not in self.index.keys()
                # Test that all limbs in branch are RtBuffTrace's
                if not all(isinstance(_rtsbranch[_c], RtBuffTrace) for _c in _rtsbranch.keys()):
                    # Go to next _rtsbranch if untrue
                    continue
                else:
                    # Proceed if limbs are RtBuffTrace
                    pass

                # Test that a vertical component is present
                for _c in _rtsbranch.keys():
                    if _c.upper() in ['Z', '3']:
                        _zcode = _c.upper()
                        # Terminate inner loop if successful match
                        continue
                    else:
                        _zcode = False
                # If a vertical component is not present, go to next _rtsbranch
                if not _zcode:
                    continue
                # Otherwise, proceed
                else:
                    pass

                # If initial tests are passed, populate entry in index
                self.index.update({k1: deepcopy(self._template)})
                # Alias index branch
                _idxbranch = self.index[k1]
                _idxbranch['zcode'] = _zcode
            else:
                pass
        

            # Test for 1C processing flag
            # Get number of buffers present
            _nbuff = len(_rtsbranch)
            # if only vertical present -> flag as 1C
            if _nbuff == 1 and _idxbranch['zcode'] in _rtsbranch.keys():
                _idxbranch['1C'] = True
            # if two components, including vertical...
            elif _nbuff == 2 and _idxbranch['zcode'] in _rtsbranch.keys():
                # If a 0-pad or vertical-clone rule for partial data, treat as 1C data
                if self.rule_1c in ['zeros', 'cloneZ']:
                    _idxbranch['1C'] = True
                # Otherwise, do not treat as 1C data
                else:
                    _idxbranch['1C'] = False
            # If it appears to have 3-C data, do not treat as 1C at this point
            elif _nbuff == 3 and _idxbranch['zcode'] in _rtsbranch.keys():
                _idxbranch['1C'] = False
            else:
                print('something went wrong probing 1C status')
                breakpoint()
            
            # Assess window validity
            _buff_zts = _rtsbranch[_idxbranch['zcode']].stats.starttime
            _buff_zte = _rtsbranch[_idxbranch['zcode']].stats.endtime
            # Get max start-time of buffers
            if _idxbranch['1C']:
                _buff_tsmax = _rtsbranch[_idxbranch['zcode']].stats.starttime
            else:
                _buff_ts = [_rtsbranch[_c].stats.starttime for _c in _rtsbranch.keys()]
                _buff_tsmax = max(_buff_ts)
            # Get min end-time of buffers
            if _idxbranch['1C']
                _buff_temin = _rtsbranch[_idxbranch['zcode']].stats.endtime
            else:
                _buff_te = [_rtsbranch[_c].stats.endtime for _c in _rtsbranch.keys()]
                _buff_temin = min(_buff_te)





                        
