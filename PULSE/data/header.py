"""
:module: PULSE.data.header
:auth: Nathan T. Stevens
:org: Pacific Northwest Seismic Network
:email: ntsteven (at) uw.edu
:license: AGPL-3.0
:purpose: This module holds class definitions for metadata header objects that build off the
    ObsPy :class:`~obspy.core.util.attribdict.AttribDict` and :class:`~obspy.core.trace.Stats` classes
    that are used for the following classes in :mod:`~PULSE`
     - :class:`~PULSE.data.mltrace.MLTrace` and decendents (:class:`~PULSE.data.mltracebuff.MLTraceBuff`) use :class`~PULSE.data.header.MLStats`
     - :class:`~PULSE.data.dictstream.DictStream` uses :class`~PULSE.data.header.DictStreamStats`
     - :class:`~PULSE.data.window.Window` uses :class`~PULSE.data.header.WindowStats`
     - :class:`~PULSE.mod.base.BaseMod` and decendents (i.e., all :mod:`~PULSE.mod` classes) uses :class`~PULSE.data.header.PulseStats`
     
"""
import copy
from math import inf
from obspy import UTCDateTime
from obspy.core.trace import Stats
from obspy.core.util.attribdict import AttribDict

###################################################################################
# Machine Learning Stats Class Definition #########################################
###################################################################################

class MLStats(Stats):
    """Extends the :class:`~obspy.core.mltrace.Stats` class to encapsulate additional metadata associated with machine learning enhanced time-series processing.

    Added/modified defaults are:
     - 'location' = '--'
     - 'model' - name of the ML model associated with a MLTrace
     - 'weight' - name of the ML model weights associated with a MLTrace

    :param header: initial non-default values with which to populate this MLStats object
    :type header: dict
    """
    # set of read only attrs
    readonly = ['endtime']
    # add additional default values to obspy.core.mltrace.Stats's defaults
    defaults = copy.deepcopy(Stats.defaults)
    defaults.update({
        'location': '--',
        'model': '',
        'weight': '',
        'processing': []
    })

    # dict of required types for certain attrs
    _types = copy.deepcopy(Stats._types)
    _types.update({
        'model': str,
        'weight': str
    })

    def __init__(self, header={}):
        """Create a :class:`~PULSE.data.mltrace.MLStats` object

        :param header: initial non-default values with which to populate this MLStats object
        :type header: dict
        """        
        super(Stats, self).__init__(header)
        if self.location == '':
            self.location = self.defaults['location']

    def __str__(self):
        """
        Return better readable string representation of this :class:`~PULSE.data.mltrace.MLStats` object.
        """
        prioritized_keys = ['model','weight','station','channel', 'location', 'network',
                          'starttime', 'endtime', 'sampling_rate', 'delta',
                          'npts', 'calib']
        return self._pretty_str(prioritized_keys)

    def utc2nearest_index(self, utcdatetime, ref='starttime'):
        """Return the integer index value for the nearest time
        of an input UTCDateTime object in the time index defined
        by this MLStats's sampling_rate and (starttime OR endtime).

        :param utcdatetime: reference utcdatetime
        :type utcdatetime: obspy.core.utcdatetime.UTCDateTime or None.
        :param ref: reference attribute, defaults to starttime
            Supported values: 'starttime','endtime'
        :type ref: str, optional
        :return:
         - **index** (*int*) - integer index value position
        """        
        if ref not in ['starttime','endtime']:
            raise ValueError(f'ref value {ref} not supported.')
        if utcdatetime is None:
            if ref == 'starttime':
                index = 0
            elif ref == 'endtime':
                index = self.stats.npts
        if isinstance(utcdatetime, UTCDateTime):
            index = round((utcdatetime - self[ref])*self.stats.sampling_rate)
            index += self.stats.npts
        else:
             raise TypeError('utcdatetime must be type obspy.core.utcdatetime.UTCDateTime or None')
        return index
    

###################################################################################
# Dictionary Stream Stats Class Definition ########################################
###################################################################################

class DictStreamStats(AttribDict):
    """A class to contain metadata for a :class:`~PULSE.data.dictstream.DictStream` object of the based on the
    ObsPy :class:`~obspy.core.util.attribdict.AttribDict` class and operates like the ObsPy :class:`~obspy.core.trace.Stats` class.
    
    This DictStream header object contains metadata on the minimum and maximum starttimes and endtimes of :class:`~PULSE.data.mltrace.MLTrace`
    objects contained within a :class:`~PULSE.data.dictstream.DictStream`, along with a Unix-wildcard-inclusive string representation of 
    all trace keys in **DictStream.traces** called **common_id**

    """
    defaults = {
        'common_id': '*',
        'min_starttime': None,
        'max_starttime': None,
        'min_endtime': None,
        'max_endtime': None,
        'processing': []
    }

    _types = {'common_id': str,
              'min_starttime': (type(None), UTCDateTime),
              'max_starttime': (type(None), UTCDateTime),
              'min_endtime': (type(None), UTCDateTime),
              'max_endtime': (type(None), UTCDateTime)}

    def __init__(self, header={}):
        """Initialize a DictStreamStats object

        A container for additional header information of a PULSE :class:`~PULSE.data.dictstream.DictStream` object


        :param header: Non-default key-value pairs to include with this DictStreamStats object, defaults to {}
        :type header: dict, optional
        """        
        super(DictStreamStats, self).__init__()
        self.update(header)
    
    def _pretty_str(self, priorized_keys=[], hidden_keys=[], min_label_length=16):
        """
        Return tidier string representation of this :class:`~PULSE.data.dictstream.DictStreamStats` object

        Based on the :meth:`~obspy.core.util.attribdict.AttribDict._pretty_str` method, and adds
        a `hidden_keys` argument

        :param priorized_keys: Keys of current AttribDict which will be
            shown before all other keywords. Those keywords must exists
            otherwise an exception will be raised. Defaults to [].
        :type priorized_keys: list, optional
        :param hidden_keys: Keys of current AttribDict that will be hidden, defaults to []
                        NOTE: does not supercede items in prioritized_keys.
        :param min_label_length: Minimum label length for keywords, defaults to 16.
        :type min_label_length: int, optional
        :return: String representation of object contents.
        """
        keys = list(self.keys())
        # determine longest key name for alignment of all items
        try:
            i = max(max([len(k) for k in keys]), min_label_length)
        except ValueError:
            # no keys
            return ""
        pattern = "%%%ds: %%s" % (i)
        # check if keys exist
        other_keys = [k for k in keys if k not in priorized_keys and k not in hidden_keys]
        # priorized keys first + all other keys
        keys = priorized_keys + sorted(other_keys)
        head = [pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)


    def __str__(self):
        prioritized_keys = ['common_id',
                            'min_starttime',
                            'max_starttime',
                            'min_endtime',
                            'max_endtime',
                            'processing']
        return self._pretty_str(prioritized_keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def update_time_range(self, trace):
        """
        Update the minimum and maximum starttime and endtime attributes of this :class:`~PULSE.data.dictstream.DictStreamStats` object using timing information from an obspy Trace-like object.

        :param trace: trace-like object with :attr:`stats` from which to query starttime and endtime information
        :type trace: obspy.core.trace.Trace
        """
        if self.min_starttime is None or self.min_starttime > trace.stats.starttime:
            self.min_starttime = trace.stats.starttime
        if self.max_starttime is None or self.max_starttime < trace.stats.starttime:
            self.max_starttime = trace.stats.starttime
        if self.min_endtime is None or self.min_endtime > trace.stats.endtime:
            self.min_endtime = trace.stats.endtime
        if self.max_endtime is None or self.max_endtime < trace.stats.endtime:
            self.max_endtime = trace.stats.endtime



###############################################################################
# WindowStats Class Definition ##########################################
###############################################################################

class WindowStats(DictStreamStats):
    """Child-class of :class:`~PULSE.data.dictstream.DictStreamStats` that extends
    contained metadata to include a set of reference values and metadata that inform
    pre-processing, carry metadata cross ML prediction operations using SeisBench 
    :class:`~seisbench.models.WaveformModel`-based models, and retain processing information
    on outputs of these predictions.

    :param header: collector for non-default values (i.e., not in WindowStats.defaults)
        to use when initializing a WindowStats object, defaults to {}
    :type header: dict, optional

    also see:
     - :class:`~PULSE.data.dictstream.DictStreamStats`
     - :class:`~obspy.core.util.attribdict.AttribDict`
    """    
    # NTS: Deepcopy is necessary to not overwrite _types and defaults for parent class
    _readonly = ['target_endtime']
    _refresh_keys = ['target_starttime','target_npts','target_sampling_rate']
    _types = copy.deepcopy(DictStreamStats._types)
    _types.update({'primary_component': str,
                   'primary_threshold': float,
                   'secondary_threshold': float,
                   'fold_threshold_level': float,
                   'target_starttime': (UTCDateTime, type(None)),
                   'target_npts': (int, type(None)),
                   'target_sampling_rate': (float, type(None)),
                   'target_endtime': (UTCDateTime, type(None))})
    defaults = copy.deepcopy(DictStreamStats.defaults)
    defaults.update({'primary_component': 'Z',
                     'primary_threshold': 0.95,
                     'secondary_threshold': 0.8,
                     'fold_threshold_level': 1.,
                     'target_starttime': None,
                     'target_sampling_rate': None,
                     'target_npts': None,
                     'target_endtime': None})
    
    def __init__(self, header={}):
        """Initialize a WindowStats object

        :param header: collector for non-default values (i.e., not in WindowStats.defaults)
            to use when initializing a WindowStats object, defaults to {}
        :type header: dict, optional

        also see:
         - :class:`~PULSE.data.dictstream.DictStreamStats`
         - :class:`~obspy.core.util.attribdict.AttribDict`
        """        
        # Initialize super + updates to class attributes
        super(WindowStats, self).__init__()
        # THEN update self with header inputs
        self.update(header)


    def __setitem__(self, key, value):
        if key in ['primary_threshold','secondary_threshold']:
            # Primary Threshold
            if 0 < value <= 1:
                super(DictStreamStats,self).__setitem__(key,value)
            else:
                raise ValueError(f'{key} must be in (0, 1]. {value} is out of bounds.')
        if key == 'fold_threshold_level':
            if 0 <= value:
                super(DictStreamStats, self).__setitem__(key,value)
            else:
                raise ValueError(f'{key} must be non-negative. {value} is out of bounds')
        if key in self._refresh_keys:
            if key == 'target_sampling_rate':
                # Target Sampling Rate
                if isinstance(value, float):
                    if inf > value > 0:
                        super(DictStreamStats, self).__setitem__(key, value)
                    else:
                        raise ValueError(f'{key} must be a positive, rational float-like value or NoneType')
            # FIXME: Not updating target_endtime...
            if not any(isinstance(self[_e], type(None)) for _e in ['target_starttime','target_sampling_rate','target_npts']):
                timediff = self.target_npts/self.target_sampling_rate
                self.__dict__['target_endtime'] = self.target_starttime + timediff
            return
        super(DictStreamStats, self).__setitem__(key, value)


    def __str__(self):
        prioritized_keys = ['common_id',
                            'primary_component',
                            'primary_threshold',
                            'secondary_threshold',
                            'fold_threshold_level',
                            'target_starttime',
                            'target_sampling_rate',
                            'target_npts',
                            'target_endtime',
                            'processing']

        hidden_keys = ['min_starttime',
                       'max_starttime',
                       'min_endtime',
                       'max_endtime']

        return self._pretty_str(prioritized_keys, hidden_keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def get_unique_secondary_components(self):
        """Return a list of unique secondary_components characters

        :return: 
         - **uniques** (*list*) -- list of unique elements.
        """
        uniques = []   
        if self.secondary_components is None:
            pass
        elif (self.secondary_components, str):
            uniques = []
            for _c in self.secondary_components:
                if _c not in uniques:
                    uniques.append(_c)
        else:
            raise TypeError('secondary_components must be type str or NoneType')
        return uniques

###############################
# PulseStats Class Definition #
###############################

class PulseStats(AttribDict):
    """A :class:`~obspy.core.util.attribdict.AttribDict` child-class for holding metadata
    from a given call of :meth:`~PULSE.mod.base.BaseMod.pulse`. 
    
    :var modname: name of the associated module
    :var starttime: POSIX start time of the last call of **pulse**
    :var endtime: UTC end time of the last call of **pulse**
    :var niter: number of iterations completed
    :var in0: input size at the start of the call
    :var in1: input size at the end of the call
    :var out0: output size at the start of the call
    :var out1: output size at the end of the call
    :var runtime: number of seconds it took for the call to run
    :var pulse rate: iterations per second
    :var stop: Reason iterations stopped

    Explanation of **stop** values
       - 'max' -- :meth:`~PULSE.mod.BaseMod.pulse` reached the **max_pulse_size** iteration limit
       - 'early0' -- flagged for early stopping before executing the unit-process in an iteration
       - 'early1' -- flagged for early stopping after executing the unit-process in an iteration
     """    
    readonly = ['pulse rate','runtime']
    _refresh_keys = {'starttime','endtime','niter'}
    defaults = {'modname': '',
                'starttime': 0,
                'endtime': 0,
                'stop': '',
                'niter': 0,
                'in0': 0,
                'in1': 0,
                'out0': 0,
                'out1': 0,
                'runtime':0,
                'pulse rate': 0}
    _types = {'modname': str,
              'starttime':float,
              'endtime':float,
              'stop': str,
              'niter':int,
              'in0':int,
              'in1':int,
              'out0':int,
              'out1':int,
              'runtime':float,
              'pulse rate':float}
    

    def __init__(self, header={}):
        """Create an empty :class:`~PULSE.mod.base.PulseStats` object"""
        super(PulseStats, self).__init__(header)

    def __setitem__(self, key, value):
        if key in self._refresh_keys:
            if key == 'starttime':
                value = float(value)
            elif key == 'endtime':
                value = float(value)
            elif key == 'niter':
                value = float(value)
            # Set current key
            super(PulseStats, self).__setitem__(key, value)
            # Set derived value: runtime
            self.__dict__['runtime'] = self.endtime - self.starttime
            # Set derived value: pulse rate
            if self.runtime > 0:
                self.__dict__['pulse rate'] = self.niter / self.runtime
            else:
                self.__dict__['pulse rate'] = 0.
            return
        if isinstance(value, dict):
            super(PulseStats, self).__setitem__(key, AttribDict(value))
        else:
            super(PulseStats, self).__setitem__(key, value)


    __setattr__ = __setitem__

    def __getitem__(self, key, default=None):
        return super(PulseStats, self).__getitem__(key, default)

    def __str__(self):
        prioritized_keys = ['modname','pulse rate','stop','niter',
                            'in0','in1','out0','out1',
                            'starttime','endtime','runtime']
        return self._pretty_str(priorized_keys=prioritized_keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
