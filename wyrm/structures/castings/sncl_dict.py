"""
:module: wyrm.structures.sncl_dict
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains the class definition for the 
    DEQ_Dict data structure used for organizing and 
    passing Msg objects between Wyrms


"""

import numpy as np
from collections import deque
import fnmatch
from operator import itemgetter
from copy import deepcopy
from obspy.realtime import RtTrace
from obspy import Trace
from wyrm import TraceMsg

class SNCL_Dict(dict):
    """
    Station.Network.Channel.Location Dictionary

    Flat dictionary data structure for Wyrm
    that uses S.N.C.L strings as reference keys
    and provides structured sub-dictionaries,
    and convenience methods for interacting with 
    collections.deque and obspy.realtime.RtTrace
    objects to buffer data

    """
    def __init__(self, key_list=None, main_field={'rtbuff': RtTrace(max_length=300)}, extra_fields={'age': 0}):
        # Call dict constructor
        super().__init__(self)

        # Compatability checks for main_field
        if not isinstance(main_field, dict):
            raise TypeError('main_field must be type dict')
        if len(main_field.keys()) != 1:
            raise IndexError('main_field must have exactly one key')
        else:
            self._main = main_field
            self._mkey = list(main_field.keys())[0]
            self._template = deepcopy(self._main)

        # Compatability check for key_list
        if not isinstance(key_list, (type(None), str, list, deque)):
            raise TypeError("key_list must be type None, string, list, or deque")
        elif isinstance(key_list, str):
            if len(key_list.split('.')) == 4:
                init_keys = [key_list]
            else:
                raise SyntaxError('string key_list must be S.N.C.L format')
        elif isinstance(key_list, (list, deque)):
            if all(self._validate_sncl(_k, verbose=False) for _k in key_list):
                init_keys = key_list
            else:
                raise SyntaxError('all keys in key_list must be S.N.C.L format')
        else:
            init_keys = []
        
        # Compatability check for extra_fields
        if not isinstance(extra_fields, (dict, type(None))):
            raise TypeError('extra_fields must be type dict or None')
        elif isinstance(extra_fields, dict):
            self._extra = extra_fields
            self._template.update(self._extra)           
        else:
            pass
        
        # Populate from provided init_keys
        for _k in init_keys:
            self.add_template_entry(_k, overwrite=True)


    ######################################
    ### __repr__ and supporting method ###
    ######################################

    def _repr_line(self, sncl):
        if not isinstance(sncl, str):
            raise TypeError('sncl must be type str')
        elif not self._validate_sncl(sncl):
            raise SyntaxError('sncl must be in SSSS.NN.CCC.LL format')
        elif sncl not in self.keys():
            raise KeyError('sncl not in self.keys()')
        else:
            pass
        # Generate Main Key segment with SNCL key
        main_key = self._mkey
        main_len = len(self[sncl][main_key])
        main_type = type(self[sncl][main_key])
        mtstr = f'{main_type}'
        mtstr = mtstr.split("'")[1].split('.')[-1]
        rstr = f'{sncl:<14} | {main_key}: {main_len} ele ({mtstr})'
        # Iterate across extra keys
        for _k in self._extra.keys():
            e_val = self[sncl][_k]
            e_type = type(self[sncl][_k])
            etstr = f'{e_type}'
            etstr = etstr.split("'")[1].split('.')[-1]
            if isinstance(e_val, (str, int, float)):
                e_mes = e_val
            else:
                try:
                    e_mes = f'{len(e_val)} ele'
                except TypeError:
                    e_mes = e_val
            rstr += f' | {_k}: {e_mes} ({etstr})'
        return rstr

    def __repr__(self, extended=False):
        rstr = f'SNCL_Dict containing {len(self)} member'
        if len(self) == 1:
            rstr += 's\n'
        else:
            rstr += '\n'

        for _i, _k in enumerate(self.keys()):
            if _i < 3:
                rstr += self._repr_line(_k)
                rstr += '\n'
            if not extended and len(self) > 6:
                if _i == 3:
                    rstr += f'       ({len(self) - 6} more)\n'
                if _i > len(self) - 4:
                    rstr += self._repr_line(_k) 
                    rstr += '\n'
                if _i == len(self) - 1:
                    rstr += (
                        'For a complete print, call "DEQ_Dict.__repr__(extended=True)"'
                    )
            elif _i >= 3:
                rstr += self._repr_line(_k)
        return rstr


    #####################################
    ### self.keys() validation method ###
    #####################################
            
    def _validate_sncl(self, sncl, verbose=True):
        if not isinstance(sncl, str):
            if verbose:
                print('sncl must be type string')
            return False
        else:
            if len(sncl.split('.')) == 4:
                _s, _n, _c, _l = sncl.split('.')
                if len(_s) > 4:
                    if verbose:
                        print(f'{sncl} station must be 4 or less characters')
                    return False
                else:
                    pass
                if len(_n) > 2:
                    if verbose:
                        print(f'{sncl} network must be 2 characters or less')
                    return False
                else:
                    pass
                if len(_c) > 3:
                    if verbose:
                        print(f'{sncl} channel must be 3 characters or less')
                    return False
                else:
                    pass
                if len(_l) > 2:
                    if verbose:
                        print(f'{sncl} location must be 2 characters or less')
                    return False
                else:
                    pass
                return True
            else:
                if verbose:
                    print('sncl needs 4 "." delimited fields')
                return False


    ################################################
    ### METHOD FOR ADDING A NEW SNCL KEYED ENTRY ###
    ################################################
    
    def add_template_entry(self, sncl, overwrite=False):
        """
        Add a new entry keyed with input sncl and default
        formatting from self._template
        i.e., {sncl: self._template}

        :: INPUT ::
        :param sncl: [str] SSSS.NN.CCC.LL formatted string
        :param overwrite: [bool] overwrite existing sncl key with defaults?

        """
        if isinstance(sncl, str):
            if self._validate_sncl(sncl):
                if sncl not in self.keys():
                    self.update({sncl: deepcopy(self._template)})
                else:
                    if not overwrite:
                        raise KeyError(f'key {sncl} already exists')
                    else:
                        self.update({sncl: deepcopy(self._template)})
                        print(f'key {sncl} already exists, overwriting')
            else:
                raise SyntaxError('key must be a SSSS.NN.CCC.LL formatted string')
        else:
            raise TypeError('key must be type str')   
    ################################################
    ### METHODS FOR MODIFYING TEMPLATE STRUCTURE ###
    ################################################

    def add_extra_key(self, key, default_value):
        if key not in self._template.keys():
            self._template.update({key: default_value})
            self._extra.update({key: default_value})
            for _k in self.keys():
                self[_k].update({key: default_value})
        else:
            raise KeyError(f'key {key} already exists in _template, cannot add')
    
    def del_extra_key(self, key, only_defaults=True):
        if key not in self._template.keys():
            raise KeyError(f'key {key} does not exist in _template, cannot del')
        elif key == self._mkey:
            raise KeyError(f'Main key {self._mkey} cannot be deleted')
        else:
            pass
        
        bools = {}
        if only_defaults:    
            # Iterate across keys
            for _sncl in self.keys():
                # If sncl keyed entry has a default value for key, save True
                if self[_sncl][key] == {key:self._template[key]}:
                    bools.update({_sncl: True})
                # Otherwise save False
                else:
                    bools.update({_sncl: False})
            # If all entries are defaults, pop this key
            if all(bools.values()):
                for _sncl in self.keys():
                    self[_sncl].pop(key)
                self._template.pop(key)
                self._extra.pop(key)
            else:
                print(f'SNCL_Dict contains non-default entries for {key} - no modifications allowed')
        else:
            for _sncl in self.keys():
                # If sncl keyed entry has a default value for key, save True
                if self[_sncl][key] == {key:self._template[key]}:
                    self[_sncl].pop(key)
                    bools.update({_sncl: True})
                # Otherwise save False
                else:
                    bools.update({_sncl: False})
            # If all entries are defaults, pop this key
            if all(bools.values()):
                self._template.pop(key)
                self._extra.pop(key)
            else:
                print(f'SNCL_Dict contains non-default entries for {key} - default entries were removed from SNCL entries')        
    
    
    #############################################
    ### SELF.KEY SORTING / SUBSETTING METHODS ###
    #############################################
    
    def sort_sncl(self, reverse=False, dcopy=False):
        if dcopy:
            x = deepcopy(dict(sorted(self.items(), reverse=reverse)))
        else:
            x = dict(sorted(self.items(), reverse=reverse))
        return x
    
    def unique(self, element='sta', return_count=False):

        if not isinstance(element, str):
            raise TypeError('element must be type str')
        elif element.lower() not in ['s','n','c','l','sta','net','cha','chan','loc','station','network','channel','location']:
            raise ValueError('element must be in approved list of aliases for a SNCL element')
        else:
            pass

        if not isinstance(return_count, bool):
            raise TypeError('return_count must be type bool')
        else:
            pass

        if return_count:
            udict = {}
        else:
            ulist = []
    
        for _k in self.keys():
            _s, _n, _c, _l = _k.split('.')
            if element.lower() in ['s','sta','station']:
                if _s not in ulist:
                    if return_count:
                        udict.update({_s:1})
                    else:
                        ulist.append(_s)
                else:
                    if return_count:
                        udict[_s] += 1
                    else:
                        pass
            elif element.lower() in ['n','net','network']:
                if _n not in ulist:
                    if return_count:
                        udict.update({_n:1})
                    else:
                        ulist.append(_n)
                else:
                    if return_count:
                        udict[_n] += 1
                    else:
                        pass
            elif element.lower() in ['c','cha','chan','channel']:
                if _c not in ulist:
                    if return_count:
                        udict.update({_c:1})
                    else:
                        ulist.append(_c)
                else:
                    if return_count:
                        udict[_c] += 1
                    else:
                        pass
            elif element.lower() in ['l','loc','location']:
                if _l not in ulist:
                    if return_count:
                        udict.update({_l:1})
                    else:
                        ulist.append(_l)
                else:
                    if return_count:
                        udict[_l] += 1
                    else:
                        pass
        
        if return_count:
            return udict
        else:
            return ulist


    #############################################
    ### METHODS FOR CREATING SUBSETS / SLICES ###
    #############################################
                
    def get_matching_keys(self, fnmatch_str):
        keys = fnmatch.filter(self.keys(), fnmatch_str)
        return keys
    
    def get_subset_items(self, fnmatch_str, deep=False):
        key_list = self.get_matching_keys(fnmatch_str)
        if deep:
            items = deepcopy(itemgetter(*key_list)(self))
        else:
            items = itemgetter(*key_list)(self)
        return items

    def get_subset(self, fnmatch_str, deep=False):
        """
        Retrieve a subset of this DEQ_Dict using a fnmatch
        filter string. Provide an option to have subset
        DEQ_Dict be a deepcopy or a slice of contents of
        this DEQ_Dict
        :: INPUTS ::
        :param fnmatch_str: [str] unix wild-card compliant
                            SSSS.NN.CCC.LL code
        :param deep: [bool] return a deepcopy?

        :: OUTPUT ::
        :return subset: [DEQ_Dict]
        """
        key_list = self.get_matching_keys(fnmatch_str)
        subset = SNCL_Dict(main_field=self._main, extra_fields=self._extra)
        for _k in key_list:
            if deep:
                subset.update({_k: deepcopy(self[_k])})
            else:
                subset.update({_k: self[_k]})
        return subset

    

    ################################
    ### DEQUE APPEND/POP METHODS ###
    ################################

    def _append_pop_deqkey(self, value, sncl, operation='append', side='left', deqkey='q'):
        """
        ~~~ PRIVATE METHOD ~~~
        Shared root method for pop/append left/right operations on deque objects
        contained within this SNCL_Dict.

        :: INPUTS ::
        :param value: [object] value to append to deque
        :param sncl: [str] SSSS.NN.CCC.LL formatted string
        :param operation: [str] 'append' or 'pop'
        :param side: [str] 'left' or 'right'
        :param deqkey: [str] key in self._template for a deque object

        :: OUTPUT ::
        :return x: [None] - in the case of append or pop on empty deque
                   [object] - in the case of a pop on a non-empty deque
        """
        
        # Check SNCL formatting
        if not self._validate_sncl(sncl, verbose=False):
            raise SyntaxError('sncl must be SSSS.NN.CCC.LL format')
        else:
            pass
        # Check operation name
        if not operation.lower() in ['append','pop']:
            raise AttributeError(f'{operation} unapproved. Must be a pop/append method')
        else:
            pass
        # Check side
        if not side.lower() in ['left', 'right']:
            raise SyntaxError('side must be "left" or "right"')
        else:
            pass
        # Check that target is a deque per the object._template def.
        if not isinstance(self._template[deqkey], deque):
            raise KeyError(f'target deqkey {deqkey} must be type deque')
        else:
            pass
        
        if operation.lower() == 'append':
            # If sncl is not in list, add a template entry
            if sncl not in self.keys():
                self.add_template_entry(sncl)
            else:
                pass
            if side.lower() == 'left':
                self[sncl][deqkey].appendleft(value)
                x = None
            elif side.lower() == 'right':
                self[sncl][deqkey].append(value)
                x = None
            else:
                raise SyntaxError('not sure how we got here...')
        
        elif operation.lower() == 'pop':
            if sncl not in self.keys():
                print(f'{sncl} not in keys')
                x = None
            else:
                if len(self[sncl][deqkey]) == 0:
                    # print(f'DEQ_Dict[{sncl}][{deqkey}] is empty')
                    x = None
                else:
                    pass
            
                if side.lower() == 'left':
                    x = self[sncl][deqkey].popleft()
                elif side.lower() == 'right':
                    x = self[sncl][deqkey].pop()

        return x            



#########################
##### CHILD CLASSES #####
#########################    


class DEQ_Dict(SNCL_Dict):
    """
    SNCL_Dict where the main field is fixed as a collections.deque object
    and convenience methods are provided for the pop/append methods associated
    with left/right modifications of the deque objects.
    """
    def __init__(self, key_list=None, main_key = 'q', extra_fields={'age': 0}):
        if not isinstance(main_key,str):
            raise TypeError('main_key must be type str')
        else:
            _main = {main_key: deque([])}
        super().__init__(key_list=key_list, main_field=_main, extra_fields=extra_fields)

    def __repr__(self, extended=False):
        rstr = 'DEQ_Dict '
        rstr += super().__repr__(self, extended=extended)
        return rstr

    def append(self, value, sncl):
        """
        append(right) method for target DEQ_Dict[sncl][deqkey]
        :: INPUTS ::
        :param value: [object] value to append to deque
        :param sncl: [str] SSSS.NN.CCC.LL formatted string
        :param deqkey: [str] key in self._template for a deque object
    
        :: OUTPUT ::
        :return x: [None] Placeholder single entry output
        """
        x = self._append_pop_deqkey(value, sncl, operation='append', side='right', deqkey=self._mkey)
        return x
    
    def appendleft(self, value ,sncl):
        """
        appendleft method for target DEQ_Dict[sncl][deqkey]
        :: INPUTS ::
        :param value: [object] value to append to deque
        :param sncl: [str] SSSS.NN.CCC.LL formatted string
        :param deqkey: [str] key in self._template for a deque object
    
        :: OUTPUT ::
        :return x: [None] Placeholder single entry output
        """
        x = self._append_pop_deqkey(value, sncl, operation='append', side='left', deqkey=self._mkey)
    
    def append_set(self, sncl_value_dict, side='left'):
        if not isinstance(sncl_value_dict, dict):
            raise TypeError('sncl_value_dict must be type dict')
        else:
            pass
        for _k in sncl_value_dict.keys():
            value = sncl_value_dict[_k]
            if not isinstance(value, deque):

            if side.lower() == 'left':
                self.appendleft(self, sncl_value_dict[_k], _k, deqkey=self._mkey)
            elif side.lower() == 'right':
                self.append(self, sncl_value_dict[_k], _k, deqkey=self._mkey)
            else:
                raise ValueError('side must be "left" or "right"')

    def pop(self, sncl):
        """
        pop(right) method for target DEQ_Dict[sncl][deqkey]
        :: INPUTS ::
        :param sncl: [str] SSSS.NN.CCC.LL formatted string
        :param deqkey: [str] key in self._template for a deque object
    
        :: OUTPUT ::
        :return x: [None] if target is empty deque
                   [object] if target is non-empty deque
        """
        x = self._append_pop_deqkey(None, sncl, operation='pop', side='right', deqkey=self._mkey)
        return x
    
    def popleft(self, sncl):
        """
        pop(right) method for target DEQ_Dict[sncl][deqkey]
        :: INPUTS ::
        :param sncl: [str] SSSS.NN.CCC.LL formatted string
        :param deqkey: [str] key in self._template for a deque object
    
        :: OUTPUT ::
        :return x: [None] if target is empty deque
                   [object] if target is non-empty deque
        """
        x = self._append_pop_deqkey(None, sncl, operation='pop', side='left', deqkey=self._mkey)
        return x
    

class RtBuff_Dict(SNCL_Dict):
    """
    SNCL_Dict where the main field is fixed as an obspy.realtime.rttrace.RtTrace
    object and the append method is used to append obspy.Trace-like objects to
    the RtTrace in each sub_dir
    """
    def __init__(self, key_list=None, main_key='rtbuff', buff_sec=300, extra_fields={'age': 0}):
        if not isinstance(main_key, str):
            raise TypeError('main_key must be type str')
        else:

        if not isinstance(buff_sec, (int, float)):
            raise TypeError('buff_sec must be type int or float')
        elif buff_sec < 0: 
            raise ValueError('buff_sec must be positive')
        else:
            pass
        _main = {main_key: RtTrace(max_length=buff_sec)}
        
        super().__init__(key_list=key_list, main_field=_main, extra_fields=extra_fields)

    
    def append(self, trace):
        # Compatability check for trace
        if not isinstance(trace, Trace):
            raise TypeError('trace must be type Trace or child class (RtTrace, TraceMsg, etc.)')
        else:
            pass
        
        # Do compatability and existance checks for sncl
        if not isinstance(trace, TraceMsg):
            _s = trace.stats.station
            _n = trace.stats.network
            _c = trace.stats.channel
            _l = trace.stats.location
            sncl = f'{_s:.4}.{_n:.2}.{_c:.3}.{_l:.2}'
        else:
            sncl = trace.sncl
        
        if not self._validate_sncl(sncl):
            raise SyntaxError('trace sncl does not comply with SSSS.NN.CCC.LL format')
        elif sncl not in self.keys():
            self.add_template_entry(sncl)
        else:
            pass

        # Append trace data
        self[sncl][self._mkey].append(trace)



class Windowing_SNCL_Dict(RtBuff_Dict):

    def __init__(self, key_list=None, main_key='rtbuff', buff_sec=300,):
        if not isinstance(main_key, str):
            raise TypeError('main_key must be type str')
        else:

        if not isinstance(buff_sec, (int, float)):
            raise TypeError('buff_sec must be type int or float')
        elif buff_sec < 0: 
            raise ValueError('buff_sec must be positive')
        else:
            pass
        _main = {main_key: RtTrace(max_length=buff_sec)}
        
        super().__init__(
            key_list=key_list,
            main_field=_main,
            extra_fields={'next_starttime': None, 'next_endtime': None}
            )

        if not isinstance(comp_order, (list, tuple, str)):
            raise TypeError('comp_order must be type str, list or tuple')
        if isinstance(comp_order, (list, tuple)):
            if not all(isinstance(x, str) for x in comp_order):
                raise TypeError('all members of list-like comp_order must be type str')
            else:
                for x in comp_order:
                    if len(x) == 0:
                        raise SyntaxError('all comp_order entries must have at least one character')
                    elif len(x) > 1:
                        if x[0] == '[' and x[-1] == ']':
                            pass
                        else:
                            raise SyntaxError('Multiple entries per component must be bounded with [] (e.g., [Z3])')
                    else:
                        pass
        else:
            self.comp_order = comp_order
        self._profile = {}


    

## SHIFT THIS TO WINDOW_WYRM
    def window_traces(self, window_sec=60., sub_str=None):
        if isinstance(sub_str, str):
            sub_keys = self.get_matching_keys(sub_str)
        elif isinstance(sub_str, type(None)):
            sub_keys = self.keys()
        else:
            raise TypeError('sub_str must be type str or None')
        # Iterate across each instrument
        for inst in self._profile.keys():
            # Alias instrument dict lookup
            _inst_dlu = self._profile[inst]
            # Alias component entry values as list
            _comp_dlu = list(_inst_dlu.values())
            
            # Assess completeness of instrument component mapping
            if all(isinstance(_c, str) for _c in _comp_dlu):
                # Flag as a candidate non-1-component instrument
                one_c = False
                pass
            elif any(isinstance(_c, str) for _c in _comp_dlu):
                # If vertical is mapped - candidate 1-component instrument
                if _inst_dlu['Z'] is not False:
                    one_c = True
                    pass
                else:
                    # Skip to next iteration - no vertical component
                    continue
            else:
                # Skip to next iteration - no components mapped
                continue
            
            # Assess if there is data in the buffer
            has_data = {}
            for _c, _sncl in _inst_dlu.items():
                # Get number of unmaksed values
                _rttr =self[_sncl][self._mkey]
                if np.ma.ismasked(_rttr.data):
                    nele = sum(_rttr.mask)
                else:
                    nele = len(_rttr)
                if nele > 0:
                    has_data.update({_c:True})
            # If all components have data, proceed
            if all(has_data.values()):
                pass
            # If some components have data, including vertical, proceed 
            elif has_data['Z']:
                # and flag as one_c
                one_c = True
            else:
                # Otherwise skip to next iteration
                continue

            # Assess if there is enough data to window
            for _c, _sncl in _inst_dlu.items():
                

                
            # Assess if all traces have data




            for _c, _sncl in _inst_dlu.items():
                # Does component have data?
                if len(self[_sncl][self._mkey]) > 0:

                # 



                nwts = self[_sncl]['next_starttime']
                rtts = self[_sncl][self._mkey].stats.starttime
                if isinstance(nwts, UTCDateTime):
                    if rtts < nwts:
                        ts_compat.append(True)
                    
                # Window times
                nwts.update({_c: self[_sncl]['next_starttime']})
                nwte.update({_c: self[_sncl]['next_endtime']})
                # Buffer times
                rtts.update({_c: self[_sncl][self._mkey].stats.starttime})
                rtte.update({_c: self[_sncl][self._mkey].stats.endtime})
                rtnp.update({_c: len(self[_sncl][self._mkey])})
            


            




        subset_keys = self.get_matching_keys(sub_str)
        sub_SNLBI = {}
        for _key in subset_keys:
            _s, _n, _c, _l = _key.split('.')
            _snlbi = f'{_s}.{_n}.{_l}.{_c[:2]}'
            if _snlbi not in sub_SNLBI.keys():
                # Create new S.N.bi entry with {_comp: SNCL}
                sub_SNLBI.update({_snlbi:{_c[-1]: _key}})
            else:
                sub_SNLBI[_snlbi].update({_c[-1]: _key})

        for _snlbi in sub_SNLBI.keys

        
        stream = Stream()
        for sncl in subset_keys:
            _rttr = self[sncl][self._mkey]
            # If this is the initial
            if self[sncl]['next_starttime'] is None:
                if len(_rttr) > 0:
                    self[sncl]['next_starttime'] = _rttr.stats.starttime
                else:
                    continue

            elif _rttr.stats.starttime >= self[sncl]['next_starttime']:
                self[sncl]['next_starttime'] = _rttr.stats.starttime
            else:
                continue
            if _rttr.stats

