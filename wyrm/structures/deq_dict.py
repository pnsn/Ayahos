"""
:module: wyrm.structures.deq_dict
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains the class definition for the 
    DEQ_Dict data structure  used for buffering
    and passing Msg objects between Wyrms

"""

import numpy as np
from collections import deque
import fnmatch
from operator import itemgetter
from copy import deepcopy


class DEQ_Dict(dict):
    """
    Double Ended Queue Dictionary
    Flat message buffer data structure for Wyrm
    that uses S.N.C.L strings as reference keys

    """
    def __init__(self, key_list=None, extra_fields={'age':0}):
        # Call dict constructor
        super().__init__(self)

        # Define template for a unit entry
        self._template = {'q': deque([])}

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

    def __repr__(self, extended=False):
        rstr = f'DEQ_Dict containing {len(self)} queue'
        if len(self) > 1:
            rstr += 's\n'
        else:
            rstr += '\n'
        for _i, _k in enumerate(self.keys()):
            if _i < 3:
                rstr += f'{_k:<14} | {len(self[_k]["q"])} elements'
                for _l in self._template.keys():
                    if _l != "q":
                        rstr += f' | {_l}: {self[_k][_l]}'
                rstr += '\n'
            if not extended and len(self) > 6:
                if _i == 3:
                    rstr += f'       ({len(self) - 6} more)\n'
                if _i > len(self) - 4:
                    rstr += f'{_k:<14} | {len(self[_k]["q"])} elements'
                    for _l in self._template.keys():
                        if _l != "q":
                            rstr += f' | {_l}: {self[_k][_l]}'
                    rstr += '\n'                
                if _i == len(self) - 1:
                    rstr += (
                        'For a complete print, call "DEQ_Dict.__repr__(extended=True)"'
                    )
            elif _i >= 3:
                rstr += f'{_k:<14} | {len(self[_k]["q"])} elements'
                for _l in self._template.keys():
                    if _l != "q":
                        rstr += f' | {_l}: {self[_k][_l]}'
                rstr += '\n'
        return rstr

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
        elif key == 'q':
            raise KeyError('Required key "q" cannot be deleted')
        else:
            pass
        
        bools = {}
        if only_defaults:    
            # Iterate across keys
            for _sncl in self.keys():
                # If sncl keyed entry has a default value for key, save True
                if self[_sncl][key] == {key:self._template[key]}
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
                print(f'DEQ_Dict contains non-default entries for {key} - no modifications allowed')
        else:
            for _sncl in self.keys():
                # If sncl keyed entry has a default value for key, save True
                if self[_sncl][key] == {key:self._template[key]}
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
                print(f'DEQ_Dict contains non-default entries for {key} - default entries were removed from SNCL entries')        

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

    def get_subset_DEQ_Dict(self, fnmatch_str, deep=False):
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
        subset = DEQ_Dict(extra_fields=self._extra)
        for _k in key_list:
            if deep:
                subset.update({_k: deepcopy(self[_k])})
            else:
                subset.update({_k: self[_k]})
        return subset

    def _append_pop_queue(self, value, sncl, operation='append', side='left', queue='q'):
        """
        ~~~ PRIVATE METHOD ~~~
        Shared root method for pop/append left/right operations on deque objects
        contained within this DEQ_Dict.

        :: INPUTS ::
        :param value: [object] value to append to deque
        :param sncl: [str] SSSS.NN.CCC.LL formatted string
        :param operation: [str] 'append' or 'pop'
        :param side: [str] 'left' or 'right'
        :param queue: [str] key in self._template for a deque object

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
        if not isinstance(self._template[queue], deque):
            raise KeyError(f'target queue {queue} must be type deque')
        else:
            pass
        
        if operation.lower() == 'append':
            # If sncl is not in list, add a template entry
            if sncl not in self.keys():
                self.add_template_entry(sncl)
            else:
                pass
            if side.lower() == 'left':
                self[sncl][queue].appendleft(value)
                x = None
            elif side.lower() == 'right':
                self[sncl][queue].append(value)
                x = None
            else:
                raise SyntaxError('not sure how we got here...')
        
        elif operation.lower() == 'pop':
            if sncl not in self.keys():
                print(f'{sncl} not in keys')
                x = None
            else:
                if len(self[sncl][queue]) == 0:
                    # print(f'DEQ_Dict[{sncl}][{queue}] is empty')
                    x = None
                else:
                    pass
            
                if side.lower() == 'left':
                    x = self[sncl][queue].popleft()
                elif side.lower() == 'right':
                    x = self[sncl][queue].pop()

        return x            

    def append(self, value, sncl, queue='q'):
        """
        append(right) method for target DEQ_Dict[sncl][queue]
        :: INPUTS ::
        :param value: [object] value to append to deque
        :param sncl: [str] SSSS.NN.CCC.LL formatted string
        :param queue: [str] key in self._template for a deque object
    
        :: OUTPUT ::
        :return x: [None] Placeholder single entry output
        """
        x = self._append_pop_queue(value, sncl, operation='append', side='right', queue=queue)
        return x
    
    def appendleft(self, value ,sncl, queue='q'):
        """
        appendleft method for target DEQ_Dict[sncl][queue]
        :: INPUTS ::
        :param value: [object] value to append to deque
        :param sncl: [str] SSSS.NN.CCC.LL formatted string
        :param queue: [str] key in self._template for a deque object
    
        :: OUTPUT ::
        :return x: [None] Placeholder single entry output
        """
        x = self._append_pop_queue(value, sncl, operation='append', side='left', queue=queue)
    
    def append_set(self, sncl_value_dict, queue='q', side='left'):
        if not isinstance(sncl_value_dict, dict):
            raise TypeError('sncl_value_dict must be type dict')
        else:
            pass
        for _k in sncl_value_dict.keys():
            value = sncl_value_dict[_k]
            if not isinstance(value, deque):
            if side.lower() == 'left':
                self.appendleft(self, sncl_value_dict[_k], _k, queue=queue)
            elif side.lower() == 'right':
                self.append(self, sncl_value_dict[_k], _k, queue=queue)
            else:
                raise ValueError('side must be "left" or "right"')

    def pop(self, sncl, queue='q'):
        """
        pop(right) method for target DEQ_Dict[sncl][queue]
        :: INPUTS ::
        :param sncl: [str] SSSS.NN.CCC.LL formatted string
        :param queue: [str] key in self._template for a deque object
    
        :: OUTPUT ::
        :return x: [None] if target is empty deque
                   [object] if target is non-empty deque
        """
        x = self._append_pop_queue(None, sncl, operation='pop', side='right', queue=queue)
        return x
    
    def popleft(self, sncl, queue='q'):
        """
        pop(right) method for target DEQ_Dict[sncl][queue]
        :: INPUTS ::
        :param sncl: [str] SSSS.NN.CCC.LL formatted string
        :param queue: [str] key in self._template for a deque object
    
        :: OUTPUT ::
        :return x: [None] if target is empty deque
                   [object] if target is non-empty deque
        """
        x = self._append_pop_queue(None, sncl, operation='pop', side='left', queue=queue)
        return x

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


                


        
                













