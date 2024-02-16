from wyrm.wyrms._base import Wyrm
from wyrm.message.trace import TraceMsg
from wyrm.structures.castings.sncl_dict import RtBuff_Dict
from copy import deepcopy
from collections import deque
import numpy as np
from obspy import UTCDateTime

class WindWyrm(Wyrm):
    def __init__(
        self,
        window_sec=60.0,
        stride_sec=18.0,
        sub_str="*.*.*.*",
        comp_map={"Z": "[Z3]", "N": "[N1]", "E": "[E2]"},
    ):
        # Initialize baseclass
        Wyrm.__init__(self)
        # Initialize profile dictionary
        self.profile = {}
        # Initialize template for profile sub-entries
        self._template = {_c: False for _c in comp_map.keys()}
        self._template.update(
            {"next_starttime": None, "1C": False}
        )
        # Initialize output queue
        self.queue = deque([])

        # Compatability checks for window_sec
        if isinstance(window_sec, float):
            self.wlen = window_sec
        elif isinstance(window_sec, int):
            self.wlen = float(window_sec)
        else:
            raise TypeError("window_sec must be type float or int")

        if self.wlen < 0:
            raise ValueError("window_sec must be positive")
        else:
            pass

        # Compatability checks for stride_sec
        if isinstance(stride_sec, float):
            self.slen = stride_sec
        elif isinstance(stride_sec, int):
            self.slen = float(stride_sec)
        else:
            raise TypeError("stride_sec must be type float or int")

        if self.slen < 0:
            raise ValueError("stride_sec must be positive")
        else:
            pass

        # Compatability checks for sub_str
        if not isinstance(sub_str, (str, None)):
            raise TypeError("sub_str must be type str or None")
        else:
            self._sub_str = sub_str

        # Compatability checks for comp_map
        if not isinstance(comp_map, dict):
            raise TypeError("comp_map must be type dict")
        else:
            if not all(isinstance(x, str) for x in comp_map.values()):
                raise TypeError("all values of comp_map must be type str")
            else:
                for _k, _v in comp_map.items():
                    if _k not in ["Z", "N", "E", "1", "2"]:
                        raise SyntaxError(
                            'Channel alias {_k} not in approved values: "Z", "N", "E", "1", "2"'
                        )
                    else:
                        pass
                    if len(_v) == 0:
                        raise SyntaxError(
                            "all comp_map values must have at least one character"
                        )
                    elif len(_v) > 1:
                        if _v[0] == "[" and _v[-1] == "]":
                            pass
                        else:
                            raise SyntaxError(
                                "Multiple entries per component must be bounded with [] (e.g., [Z3])"
                            )
                    else:
                        pass
        # If checks pass, complete initialization by preserving comp_map
        self._comp_map = comp_map

    def _get_sub_keys(self, rtbuff):
        if isinstance(self._sub_str, str):
            sub_keys = rtbuff.get_matching_keys(self._sub_str)
        elif isinstance(self._sub_str, type(None)):
            sub_keys = rtbuff.keys()
        else:
            raise TypeError("sub_str must be type str or None")
        return sub_keys


    def _fetch_rtbuff_entries(self, rtbuff, inst_code, deep=False):
        """
        Given an RtBuff_Dict and and an inst_code from self.profile
        that corresponds to the input rtbuff, return an alias/copy of
        the data from the rtbuff that correspond with the profiled
        inst_code

        :: INPUT ::
        :param rtbuff: [wyrm.structures.sncl_dict.RtBuff_Dict]
        :param inst_code: [str]
        :param deep: [bool] create deepcopy of queried rtbuff contents?

        :: OUTPUT ::
        :return out: [dict] composed of sub-dict entries from rtbuff
        """
        if not isinstance(rtbuff, RtBuff_Dict):
            raise TypeError("rtbuff must be tpe RtBuff_Dict")
        else:
            pass
        if inst_code not in self.profile.keys():
            raise KeyError("inst_code is not in self.profile keys")
        else:
            inst = self.profile[inst_code]
        out = {}
        for _c in self._comp_map.keys():
            if inst[_c]:
                if deep:
                    out.update({_c: deepcopy(rtbuff[inst[_c]])})
                else:
                    out.update({_c: rtbuff[inst[_c]]})
            else:
                out.update({_c: False})
        return out

    def _profile_rtbuffer(self, rtbuff):
        """
        Iterate across RtBuff_Dict keys and populate/update a
        reverse-look-up dictionary "self._profile" that to groups
        S.N.C.L codes into ordered dictionaries that are keyed by the
        "Instrument" code and have sub-keys that match the specified
        comp_order keys
        - i.e., Station.Network.Location.BandInstrument_Type
        for GNW.UW.BHZ. --> {'GNW.UW..BH':{'Z':'GNW.UW.BHZ.'},
                                         :{'N': None},
                                         :{'E': None}}
            where the N and E aliased components have yet to populate
            as known members of this dictionary

        :: INPUT ::
        :param rtbuff: [wyrm.structures.sncl_dict.RtBuff_Dict] Populated RtBuff_Dict
                        object, likely inherited from a WaveBuffWyrm object
        :: OUTPUT ::
        :return updates: [int] number of updated sub-dict entries from this profiling


        """
        sub_keys = self._get_sub_keys(rtbuff)

        nupdate = 0
        # Iterate across known SNCL
        for _sncl in sub_keys:
            # Compose S.N.L.BI "inst" code
            _s, _n, _l, _c = _sncl.split(".")
            # Get inst code
            _inst = f"{_s}.{_n}.{_l}.{_c[:2]}"
            # Get component code
            _ccode = _c[-1]
            # Identify component alias
            for _a, _b in self.comp_order.items():
                if _ccode in _b:
                    _acode = _a

            # If new INST instance, create new top-level {INST:self._template} entry in profile
            if _inst not in self._profile.keys():
                self.profile.update({_inst: deepcopy(self._template)})
            # Else, pass
            else:
                pass

            # Update inst/aliased component codes if value fills a None
            if not self.profile[_inst][_acode]:
                self.profile[_inst][_acode] = _sncl
                # And increase update counter
                nupate += 1
            else:
                pass
        
        # Run check on starttime assignments
        for _ic in self.profile.keys():
            icdata = self.profile[_ic]
            rtdata = self._fetch_rtbuff_entries(rtbuff, _ic, deep=False)
            # Run check on 1C status
            if all(icdata[_c] for _c in self._comp_map.keys()):
                # Update '1C' flag as False
                icdata["1C"] = False
            # If Z is mapped
            elif icdata["Z"]:
                # If both horizontals are unmapped
                if any(icdata[_c] for _c in self._comp_map.keys() if _c != "Z"):
                    # Update '1C' flag as True
                    icdata["1C"] = True
                # If one horizontal is also mapped
                else:
                    # Update "1C" flag as False
                    icdata["1C"] = False

            # Run check on next_starttime
            # If already assigned, pass - NOTE: striding is taken care of by another method.
            if isinstance(icdata['next_starttime'], UTCDateTime):
                pass
            # If next_starttime is unassigned (None), run cross-reference
            elif icdata['next_starttime'] is None:
                # If 1C...
                if icdata['1C']:
                    # See if Z component has non-masked data
                    _rttr = rtdata['Z'][rtbuff._mkey]
                    if np.ma.is_masked(_rttr.data):
                        nele = sum(_rttr.data.mask)
                    else:
                        nele = len(_rttr)
                    if nele > 0:
                        # If so, update
                        icdata['next_starttime'] = _rttr.stats.starttime
                    else:
                        pass
                # If not 1C
                elif not icdata['1C']:
                    # Iterate across components
                    for _c in self._comp_map.keys():
                        # Grab trace length, discounting masked values
                        _rttr = rtdata[_c][rtbuff._mkey]
                        if np.ma.is_masked(_rttr.data):
                            nele = sum(_rttr.data.mask)
                        else:
                            nele = len(_rttr)
                        
                        # If trace is non-empty
                        if nele > 0:
                            # If next_starttime is None in this iteration, assign
                            if icdata['next_starttime'] is None:
                                icdata['next_starttime'] = _rttr.stats.starttime
                            # If another non-empty component gives a later starttime, update
                            elif icdata['next_starttime'] < _rttr.stats.starttime:
                                icdata['next_starttime'] = _rttr.stats.starttime
                            # Otherwise, pass
                            else:
                                pass

        return nupdate



    def _generate_window(self, rtdata, icdata, mkey='rtbuff'):
        if icdata['1C']:
            clist = ['Z']
        elif not icdata['1C']:
            clist = [_c for _c in self._comp_map.keys() if icdata[_c]]

        if icdata['next_starttime'] is not None:
            nwts = icdata['next_starttime']
            nwte = icdata['next_starttime'] + self.wlen
            st = Stream()
            stmsg = StreamMsg(order=self._comp_map.keys())
            for _c in clist:
                st += rtdata[_c][mkey].copy()
            if all(_tr.stats.starttime <= nwts for _tr in st):
                if all(_tr.stats.endtime >= nwte for _tr in st):
                    wind = st.copy().trim(starttime=nwts, endtime=nwte)
                    for _tr in wind:
                        tracemsg = TraceMsg(_tr)
                        stmsg.append(tracemsg)
                else:
                    pass
            else:
                pass
            return stmsg                
        else:
            return StreamMsg()
    
        
                    





    # Fetch mapped data
    rt_data = self._fetch_rtbuff_entries(rtbuff, inst_code, deep=False)

    def _validate_inst_window(
        self,
        inst_profile,
        rt_data,
        mkey='rtbuff',
    ):
        # Compatability checks
        if not isinstance(rt_data, dict):
            raise TypeError("rtbuff must be type RtBuff_Dcit")
        else:
            pass

        if not isinstance(inst_profile, dict):
            raise TypeError("inst_profile must be type dict")
        elif inst_profile not in self.profile.values():
            raise ValueError("inst_value must be a value in self.profile")
        else:
            inst = inst_profile

        # -- Run update checks on 1C status from the profile side --
        # If all components are mapped
        if all(inst[_c] for _c in self._comp_map.keys()):
            # Update '1C' flag as False
            inst["1C"] = False
        # If Z is mapped
        elif inst["Z"]:
            # If both horizontals are unmapped
            if any(inst[_c] for _c in self._comp_map.keys() if _c != "Z"):
                # Update '1C' flag as True
                inst["1C"] = True
            # If one horizontal is also mapped
            else:
                # Update "1C" flag as False
                inst["1C"] = False
        # Otherwise, return false
        else:
            return False

        # -- Run cross reference on window times and trace bounds -- #
        if inst['1C']:
            _rttr = rt_data['Z'][mkey]
            if np.ma.is_masked(_rttr.data):
                nele = sum(_rttr.data.mask)
            else:
                nele = len(_rttr)
            if nele > 0:
                if inst['next_starttime'] is None:
                    inst['next_starttime'] = _rttr.stats.starttime
                    inst['next_endtime'] = inst['next_starttime'] + self.wlen
                else:
                    pass

        elif not inst['1C']:
            for _c in self._comp_map.keys():
                _rttr = rt_data[_c][mkey]
                if np.ma.is_masked(_rttr.data):
                    nele = sum(_rttr.data.mask)
                else:
                    nele = len(_rttr)
                if nele > 0:
                    if inst['next_starttime'] is None:
                        inst['next_starttime'] = _rttr.stats.starttime
                        inst['next_endtime'] = inst['next_starttime'] + self.wlen
                    elif inst['next_starttime'] < _rttr.stats.starttime:
                        inst['next_starttime'] = _rttr.stats.starttime
                        inst['next_endtime'] = inst['next_starttime'] + self.wlen
                    else:
                        pass

                        
        # -- Run cross reference on data -- #
        # If flagged as 1C, just operate on Z metadata
        if inst["1C"]:
            # Get realtime trace from Z only
            _rttr = rt_data["Z"][mkey]
            # Check if it's masked
            if np.ma.is_masked(_rttr.data):
                nele = sum(_rttr.data.mask)
            else:
                nele = len(_rttr)
            # If RtTrace has non-masked data
            if nele > 0:
                # If next_starttime is None, pull this from trace
                if inst_profile["next_starttime"] is None:
                    inst_profile["next_starttime"] = _rttr.stats.starttime
                    inst_profile["next_endtime"] = (
                        inst_profile["next_starttime"] + self.wlen
                    )
                else:
                    rtts = _rttr.stats.starttime
                    rtte = _rttr.stats.endtime
                    nwts = inst_profile['next_starttime']
                    nwte = inst_profile['next_endtime']
                    if rtts <= nwts and rtte >= nwte:
                        return True
                    else:
                        return False
            else:
                return False
        
        elif not inst['1C']:
            for _c in self._comp_map.keys():
                if inst[_c]:


    def _validate_unmasked_rtdata_window(self, rt_data, inst_profile, mkey='rtbuff', comp='Z'):
        _tr = rt_data[comp][mkey]
        if np.ma.is_masked(_tr.data):
            nele = sum(_tr.data.mask)
        else:
            nele = len(_tr)
        if 
        