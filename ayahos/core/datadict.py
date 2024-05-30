import fnmatch, os, obspy, logging

class DataDict(dict):
    """Dictionary-like object for holding various data class objects indexed by a unique key based
    on the Network.Station.Location.Channel.Model.Weight naming convention described in :class:`~ayahos.core.mltrace.MLTrace`

    This class serves as the base-class for the :class:`~ayahos.core.dictstream.DictStream` and :class:`~ayahos.core.windowstream.WindowStream`
    classes and provides the underlying view/select/copy methods

    :param objects: list of objects with identical type with an attribute matching 'key_attr', defaults to None
    :type objects: list of objects, optional
    :param key_attr: name of the attribute to use as dictionary keys, defaults to 'id'
    :type key_attr: str, optional
    :param **options: key-word arguments passed to the objects' __add__ method
    :type **options: kwargs
    """
    
    def __init__(self, objects=None, key_attr='id', **options):
        super.__init__()
        if not isinstance(key_attr, str):
            raise TypeError('key_attr must be type str')
        else:
            self.key_attr = key_attr
        if objects is not None:
            if all(isinstance(e, type(objects[0])) for e in objects):
                if all(key_attr in e.__dir__() for e in objects):
                    for e in objects:
                        self.append(e, **options)
                else:
                    raise AttributeError
            else:
                raise TypeError
        else:
            pass
        
    def append(self, other, **options):
        if self.key_attr not in other.__dir__():
            raise AttributeError
        else:
            key = getattr(other, self.key_attr)
            if key in self.keys():
                self[key].__add__(other)
            else:
                self.__setitem__(key, other)
    
    def __iter__(self):
        """return a robust iterator for this DataDict that iterates across its values

        :return: list of values
        :rtype: list
        """        
        return list(self.values()).__iter__()

    def __getitem__(self,index):
        if isinstance(index, int):
            value = self[list(self.keys())[index]]
        elif isinstance(index, slice):
            keyslice = list(self.keys()).__getitem__(index)
            values = [self.values[_k] for _k in keyslice]
            out = self.__class__(objects=values)
        elif isinstance(index, str):
            if index in self.keys():
                out = self[index]
            else:
                raise KeyError
        elif isinstance(index, list):
            if all(isinstance(e_, str) and e_ in self.keys() for e_ in index):
                values = [self[k_] for k_ in index]
                out = self.__class__(objects=values)
            else:
                raise KeyError
        else:
            raise TypeError
        return out
    
    def __setitem__(self, index, object):
        if self.ref_type is not None:
            if not isinstance(object, self.ref_type):
                raise TypeError(f'cannot mix object types - must be type {self.ref_type}')
        if isinstance(index, int):
            key = list(self.keys())[index]
        elif isinstance(index, str):
            key = index
        else:
            raise TypeError
        self.update({key: object})