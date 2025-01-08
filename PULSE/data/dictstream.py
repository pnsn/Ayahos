"""
:module: PULSE.data.dictstream
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

Purpose
=======
This module contains the class definition for :class:`~PULSE.data.dictstream.DictStream` data objects
and :class:`~PULSE.data.dictstream.DictStreamStats` metadata objects that host and coordinate functionalities 
of sets of :class:`~PULSE.data.foldtrace.FoldTrace` objects. 

:class:`~PULSE.data.dictstream.DictStream` objects use a :class:`~dict` to hold traces as the **traces** attribute,
rather than the :class:`~list` used by the ObsPy :class:`~obspy.core.stream.Stream` **traces** attribute. 
This accelerates sorting, slicing, and querying individual traces (or sets of traces) via the hash-table functionality that 
underlies python dictionaries.

In exchange, this places more restrictions on the objects that can be placed in a
:class:`~PULSE.data.dictstream.DictStream` object in comparison to it's ObsPy forebearer
as dictionaries require unique keys for each entry.

The **stats** attribute added to :class:`~PULSE.data.dictstream.DictStream` objects
(:class:`~PULSE.data.dictstream.DictStreamStats`) that provides summary metadata on the 
contents of the DictStream. It is modeled after the ObsPy :class:`~obspy.core.trace.Stats` class.

 * TODO: cleanup extraneous (developmental) methods that are commented out
"""
import fnmatch, warnings

from obspy import Trace, Stream, Inventory
from PULSE.data.foldtrace import FoldTrace
from PULSE.util.header import MLStats

###################################################################################
# Dictionary Stream Class Definition ##############################################
###################################################################################

class DictStream(Stream):
    """:class:`~obspy.core.stream.Stream`-like object hosting
    :class:`~PULSE.data.foldtrace.FoldTrace` objects in a :class:`~dict` **trace** attribute,
    rather than the :class:`~list` **trace** attribute used by :class:`~obspy.core.stream.Stream`.
    
    The :class:`~.DictStream` **traces** attribute uses a prescribed key attribute to automatically
    generate keys. If traces have identical keys they are merged structure trades off the flexibility
    of having multiple, identically labeled Traces are merged using the :class:`~PULSE.data.foldtrace.FoldTrace.__iadd__`
    method. 

    In exchange for this restriction, the dictionary-based storage of trace objects allows for
    accelerated searching/splitting/referencing of trace objects by leveraging the hash mapping
    underlying python :class:`~dict` (O[1]) object compared to the :class:`~list` (O[n]) object
    underlying the :var:`~obspy.core.stream.Stream.traces` attribute.

    This is particularly useful for working with large collections of traces.

    Most of the changes in :class:`~.DictStream` functionalities relative to :class:`~obspy.core.stream.Stream`
    are affectuated using dunder-methods (e.g., __getattr__) to keep the syntax of this
    class and it's parent nearly identical.

    Parameters
    ----------
    :param traces: initial list of ObsPy Trace-like objects, defaults to []
        All ingested traces are converted to :class:`~PULSE.data.foldtrace.FoldTrace` objects
        unless they already meet this requirement. 
        E.g., :class:`~PULSE.data.foldtracebuff.FoldTraceBuff` objects are not altered.
    :type traces: list of :class:`~obspy.core.trace.Trace` (or children) objects, optional
    :param key_attr: attribute of each trace in traces to use for their unique key values 
        such that a :val:`~.DictStream.traces` entry is {tr.key_attr: tr}, defaults to 'id'
        Supported values are provided in :var:`~PULSE.data.foldtrace.FoldTrace.id_keys`
    :type key_attr: str, optional
    :param **options: key word argument gatherer that passes optional kwargs to
        :meth: `~PULSE.data.dictstream.DictStream.__iadd__` for merging input
        traces that have the same `key_attr`
        NOTE: This is how DictStream objects handle matching trace ID's (key_attr values)
        for multiple traces.
    :type **options: kwargs

    :var traces: A container :class:`~dict` for :class:`~PULSE.data.foldtrace.FoldTrace`-type objects.
    :var key_attr: Key attribute name used by this :class:`~PULSE.data.dictstream.DictStreamStats` object.

    .. rubric:: Basic Usage
    >>> from PULSE.data.dictstream import DictStream
    >>> from obspy import read
    >>> ds = DictStream(traces=read())
    >>> ds
    3 FoldTrace(s) in DictStream
    BW.RJOB.--.EHZ.. : BW.RJOB.--.EHZ.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    BW.RJOB.--.EHN.. : BW.RJOB.--.EHN.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    BW.RJOB.--.EHE.. : BW.RJOB.--.EHE.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples

    Iterable sets of ObsPy :class:`~obspy.core.trace.Trace`-like objects and :class:`~obspy.core.stream.Stream` objects can be
    loaded into a :class:`~PULSE.data.dictstream.DictStream` when initializing the object.
    
    >>> ds = DictStream(traces=read(), key_attr='component')
    >>> ds
    3 FoldTrace(s) in DictStream
    Z : BW.RJOB.--.EHZ.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    N : BW.RJOB.--.EHN.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    E : BW.RJOB.--.EHE.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples

    Users can specify the key attribute used for auto-generating keys in **DictStream.traces**
    
    .. rubric:: Indexing and Slicing
    >>> ds[0]
    BW.RJOB.--.EHZ.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples | Fold: [1] 1.00

    Indexing with an integer value returns the :class:`~PULSE.data.foldtrace.FoldTrace`
    at the specified position. This matches the syntax for getting items from
    :class:`~obspy.core.stream.Stream` objects.

    **Note**: input ObsPy :class:`~obspy.core.trace.Trace` object(s) are automatically
    converted into :class:`~PULSE.data.foldtrace.FoldTrace` objects.
    
    >>> ds['BW.RJOB.--.EHN..']
    BW.RJOB.--.EHN.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples | Fold: [1] 1.00

    Indexing with a string value that matches a key in **DictStream.traces** returns a view of the 
    corresponding :class:`~PULSE.data.foldtrace.FoldTrace` object.

    >>> ds[1:]
    2 FoldTrace(s) in DictStream
    BW.RJOB.--.EHN.. : BW.RJOB.--.EHN.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    BW.RJOB.--.EHE.. : BW.RJOB.--.EHE.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    
    Indexing with a slice returns a new :class:`~PULSE.data.dictstream.DictStream` object containing a view of the
    :class:`~PULSE.data.foldtrace.FoldTrace` objects at the specified slice position(s). This matches the
    behavior of slice indexing with ObsPy :class:`~obspy.core.stream.Stream` objects.

    >>> ds[['BW.RJOB.--.EHZ..','BW.RJOB.--.EHN..']]
    2 FoldTrace(s) in DictStream
    BW.RJOB.--.EHZ.. : BW.RJOB.--.EHZ.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    BW.RJOB.--.EHN.. : BW.RJOB.--.EHN.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples

    Indexing with a list of keys returns a returns a new :class:`~PULSE.data.dictstream.DictStream` object containing a view
    of matching keyed :class:`~PULSE.data.foldtrace.FoldTrace` objects.
    
    **Note**: slice and key-list indexing produce new DictStream objects, but the FoldTraces contained within are views of the original FoldTraces.
    I.e., the FoldTraces are the same in-memory objects, the DictStreams referencing them are not.
    
    .. rubric:: Supported Operations
    ``DictStream = DictStreamA + DictStreamB``
        Merges all foldtraces within the two DictStream objects
    
    ``DictStreamA += DictStreamB``
        Extends the DictStream object ``DictStreamA`` with all traces from ``DictStreamB``.
    
    **NOTES**:

    * Not all inherited methods from ObsPy :class:`~obspy.core.stream.Stream` are gauranteed to be supported. 
    
    * Please submit a bug report if you find one that you'd like to use that hasn't been supported!

    DictStream Class Methods
    ========================
    """
    supported_keys = dict(MLStats().id_keys).keys()

    def __init__(self, traces=[], **options):
        """Initialize a :class:`~PULSE.data.dictstream.DictStream` object

        Parameters
        ----------
        :param traces: ObsPy Trace-like object, or iterable collections thereof, defaults to [].
        :type traces: obspy.core.trace.Trace-like or list-like collections thereof, optional. See Notes.
        :param header: Non-default parameters to pass to the initialization of this DictStream's :class:`~PULSE.data.dictstream.DictStreamStats` attribute, defaults to {}
        :type header: dict, optional
        :param key_attr: Attribute of each FoldTrace (or ObsPy Trace-like object converted into a PULSE FoldTrace) to use for **DictStream.traces** key values, defaults to 'id'
        :type key_attr: str, optional
        """        
        # initialize as empty stream
        super().__init__()
        self.traces = {}
        # Process key_attr out of options, rather than as an explicit kwarg
        if 'key_attr' in options.keys():
            key_attr = options.pop('key_attr')
            if key_attr in self.supported_keys:
                self.key_attr = key_attr
            else:
                raise KeyError(f'key_attr "{key_attr}" not in supported_keys.')
        if not hasattr(self, 'key_attr'):
            self.key_attr = 'id'
        # Extend to add traces
        self.extend(traces, **options)

    # PROPERTIES
        
    def get_keys(self):
        return list(self.traces.keys())
    
    keys = property(get_keys)

    def get_values(self):
        return list(self.traces.values())
    
    values = property(get_values)

    #####################################################################
    # DUNDER METHOD UPDATES #############################################
    #####################################################################

    def __eq__(self, other):
        if not isinstance(other, DictStream):
            return False
        if self.traces != other.traces:
            return False
        return True

    def __iter__(self):
        """
        Return a robust iterator for DictStream.traces.values()
        :returns: 
         - **output** (*list*) -- list of values in DictStream.traces.values
        """
        return list(self.traces.values()).__iter__()
    
    def __getitem__(self, index):
        """
        Fusion of the :meth:`~dict.__getitem__` and :meth:`~list.__getitem__` methods.

        This accepts integer and slice indexing to access items in DictStream.traces,
        as well as str-type key values. 

        __getitem__ calls that retrieve a single trace return a FoldTrace object whereas
        calls that retrieve multiple traces return a DictStream object

        Explainer
        ---------
        The DictStream class defaults to using trace.id values for keys (which are str-type),
        so this approach removes the ambiguity in the expected type for self.traces' keys.

        Parameters
        ----------
        :param index: indexing value with behaviors based on index type
        :type index: int, slice, str, list
        
        :returns:
             - for *int* index -- returns the i^{th} trace in list(self.traces.values())
             - for *slice* index -- returns a DictStream with the specified slice from
                list(self.traces.values()) and associated keys
             - for *str* index -- returns the trace corresponding to self.traces[index]
             - for *list* of *str* -- returns a DictStream containing the traces as specified by a list of trace keys
        """
        # Handle single item fetch
        if isinstance(index, int):
            if index > len(self) - 1:
                raise ValueError(f'index {index} exceeds bounds of this DictStream [0, {len(self) - 1}]')
            trace = self.traces[list(self.traces.keys())[index]]
            out = trace
        # Handle slice fetch
        elif isinstance(index, slice):
            keyslice = list(self.traces.keys()).__getitem__(index)
            traces = [self.traces[_k] for _k in keyslice]
            out = self.__class__(traces=traces)
        # Preserve dict.__getitem__ behavior for string arguments
        elif isinstance(index, str):
            if index in self.traces.keys():
                out = self.traces[index]
            else:
                raise KeyError(f'index {index} is not a key in this DictStream\'s traces attribute')
        # Handle lists & sets of keys (dictionary slice)
        elif isinstance(index, (list, set)):
            # purely strings
            if all(isinstance(_e, str) and _e in self.traces.keys() for _e in set(index)):
                traces = [self.traces[_k] for _k in set(index)]
                out = self.__class__(traces=traces, key_attr=self.key_attr)
            # purely integers
            elif all(isinstance(_e, int) and _e <= len(self) - 1 for _e in set(index)):
                traces = [self.traces[_k] for _k in set(index)]
                out = self.__class__(traces=traces, key_attr=self.key_attr)
            # Mixed
            else:
                raise IndexError('index elements are not uniformly type int or str')
        else:
            raise TypeError(f'index type {type(index)} not supported.')
        return out
    
    def __setitem__(self, key, value):
        """Provides options to __setitem__ for string and int type indexing 
        consistent with :meth:`~PULSE.data.dictstream.DictStream.__getitem__`
        behaviors.

        Parameters
        ----------
        :param key: key to assign value object to, either a string-type
            key value or int-type key value
        :type key: int or str
        :param value: Trace-like object to add, will be converted into a 
            :class:`~PULSE.data.foldvalue.FoldTrace` object if necessary
        :type value: obspy.core.value.Trace
        """
        # Handle empty DictStream case
        if len(self) == 0:
            raise AttributeError('DictStream is empty. Use DictStream.extend to add values.')
        
        # Compatability check/conversion on value
        if not isinstance(value, Trace):
            raise TypeError(f'input object value must be type obspy.core.value.Trace or child. Not {type(value)}')
        elif not isinstance(value, FoldTrace):
            value = FoldTrace(value)
        elif isinstance(value, FoldTrace):
            pass
        else:
            raise TypeError('Shouldn\'t have gotten here...')
        # Allow integer values
        if isinstance(key, int):
            if key >= len(self):
                raise ValueError(f'int-type input for "key" outside range of this DictStream [0, {len(self) - 1}]')
            else:
                key = self.keys[key]
        elif isinstance(key, str):
            pass
        else:
            raise TypeError('key must be type int or str')
        
        # Ensure valid mapping
        if key != value.id_keys[self.key_attr]:
            raise KeyError(f'provided/implicit key mismatches expected key for provide value: {key} != {value.id_keys[self.key_attr]}')
        # Use update to alter/add entry
        self.traces.update({key: value})



    def __delitem__(self, index):
        """Provides options to __delitem__ for string and int type indexing
        consistent with :meth:`~PULSE.data.dictstream.DictStream.__getitem__`
        behaviors

        :param index: integer position value or string key of an element in this
            DictStream's **traces** attribute
        :type index: str or int
        :raises TypeError: _description_
        :return:
            - **output** (*PULSE.data.foldtrace.FoldTrace*) -- removed element
        """        
        if isinstance(index, str):
            key = index
        elif isinstance(index, int):
            key = list(self.traces.keys())[index]
        else:
            raise TypeError(f'index type {type(index)} not supported. Only int and str')   
        return self.traces.__delitem__(key)

    def __getslice__(self, i, j, k=1):
        """
        Updated __getslice__ that leverages the 
        :meth:`~PULSE.data.dictstream.DictStream.__getitem__` method
        for retrieving integer-indexed slices of DictStream.traces values. 

        :param i: leading index value, must be non-negative
        :type i: int
        :param j: trailing index value, must be non-negative
        :type j: int
        :param k: index increment, defaults to 1
        :type k: int, optional

        :return: view of this DictStream
        :type: PULSE.data.dictstream.DictStream
        """
        return self.__class__(traces=self[max(0,i):max(0,j):k])

    def __add__(self, other, **options):
        """Add the contents of this DictStream object and another iterable
        set of :class:`~obspy.core.trace.Trace`-like objects into a list
        and initialize a new :class:`~PULSE.data.dictstream.DictStream` object.

        Parameters
        ----------
        :param other: Trace-like object or iterable comprising several Traces
        :type other: :class:`~obspy.core.trace.Trace`-like, or list-like thereof
        :param **options: key-word argument gatherer that passes to DictStream.__init__
        :type **options: kwargs
            # NOTE: If key_attr is not specified in **options, 
            #     the new DictStream uses key_attr = self.key_attr
        :raises TypeError: If other's type does not comply
        :return: new DictStream object containing traces from self and other
        :rtype: PULSE.data.dictstream.DictStream
        """
        out = self.copy()
        if isinstance(other, (Trace, Stream)):
            out.extend(other)
        else:
            raise TypeError('other must be type obspy.core.trace.Trace or obspy.core.stream.Stream')


        if isinstance(other, Trace):
            other = [other]
        if not all(isinstance(tr, Trace) for tr in other):
            raise TypeError
        traces = [tr for tr in self.traces] + other
        # Allow for specifiying a different key_attr
        if 'key_attr' not in options.keys():
            options.update({'key_attr': self.key_attr}) 
        return self.__class__(traces=traces, **options)

    def __iadd__(self, other, **options):
        """
        Alias for the :meth:`~PULSE.data.dictstream.DictStream.extend`
        method, allowing rich use of the ``+=`` operator.
        """
        self.extend(other, **options) 
        return self           

    def _add_trace(self, other: Trace, **options):
        if isinstance(other, Trace):
            if isinstance(other, FoldTrace):
                pass
            else:
                other = FoldTrace(other)
        else:
            raise TypeError('other must be type obspy.core.trace.Trace')
        
        key = other.id_keys[self.key_attr]
        if key not in self.traces.keys():
            self.traces.update({key: other})
        else:
            try:
                self.traces[key].__iadd__(other, **options)
            except Exception:
                raise 

    def extend(self, other, **options):
        """Core method for adding :class:`~PULSE.data.foldtrace.FoldTrace` objects
        to this DictStream. In addition to directly incorporating PULSE 
        :class:`~PULSE.data.foldtrace.FoldTrace` and :class:`~PULSE.data.foldtracebuff.FoldTraceBuff`,
        and iterable groups thereof, this method also accepts any inputs of ObsPy
        :class:`~obspy.core.trace.Trace` objects and iterable sets thereof.
        In this case, Trace-like objects are converted into :class:`~PULSE.data.foldtrace.FoldTrace` objects.

        :param other: waveform (meta)data object, or iterable sets thereof, to add to this DictStream
        :type other: obspy.core.trace.Trace and iterable groups thereof
        :param options: key-word argument collector that is passed to :meth:`~PULSE.data.foldtrace.FoldTrace.__add__` in the event
            that a trace-like object in **other** has the same key as an existing value in **DictStream.traces**
        :type options: kwargs

        **Notes**:

        * Iterable groupings include ObsPy :class:`~obspy.core.stream.Stream`, child-classes like PULSE's :class:`~PULSE.data.dictstream.DictStream`.
        and Python iterable objects: :class:`~list`, :class:`~tuple`, and :class:`~set`.

        * This method uses the **DictStream.key_attr** value to generate keys for items in **other** and resolves matching
        keys in **DictStream.traces** and thos arising from **other** using :meth:`~PULSE.data.foldtrace.FoldTrace.__add__`.
        
        Also see :meth:`~PULSE.data.dictstream.DictStream.__iter__`.
        """
        # Explicit safety catch for options
        if options.keys() <= set(['method','fill_value','idtype']):
            pass
        else:
            misfits = options.keys() - set(['method','fill_value','idtype'])
            raise SyntaxError(f'DictStream.extend() given unexpected keyword argument(s) for internal call of FoldTrace.__add__: {misfits}')
        # Run wrapper around _add_trace
        try:
            if isinstance(other, Trace):
                self._add_trace(other, **options)
            elif isinstance(other, (Stream, list)):
                for tr in other:
                    self._add_trace(tr, **options)
            else:
                raise TypeError('other must be type obspy.core.trace.Trace or obspy.core.stream.Stream')
        except:
            raise

    def __str__(self, short=False):
        """string representation of the full module/class path of
        :class:`~PULSE.data.dictstream.DictStream`

        :return rstr: representative string
        :rtype rstr: str
        """
        if short:
            rstr = 'DictStream'
        else:
            rstr = 'PULSE.data.dictstream.DictStream'
        return rstr

    def __repr__(self, extended=False):
        """string representation of the contents of this PULSE.data.dictstream.DictStream` object
        Always shows the representation of the DictStream.stats object,
        Truncates the representation of Trace-like objects in the DictStream
        if there are more than 20 Trace-like objects.

        :param extended: print full list of Trace-likes? Defaults to False
        :type extended: bool, optional
        :return rstr: representative string
        :rtype rstr: str

        ..rubric:: Use

        >> dst = DictStream(read('example_file.mseed'))
        >> print(dst.__repr__(extended=True))
        >> print(dst.__repr__(extended=False))
        

        """        
        rstr = ''
        if len(self.traces) > 0:
            id_length = max(len(_tr.id) for _tr in self.traces.values())
        else:
            id_length=0
        if len(self.traces) > 0:
            rstr += f'\n{len(self.traces)} {type(self[0]).__name__}(s) in {type(self).__name__}\n'
        else:
            rstr += f'\nNothing in {type(self).__name__}\n'
        if len(self.traces) <= 20 or extended is True:
            for _l, _tr in self.traces.items():
                rstr += f'{_l:} : {_tr.__str__(id_length)}\n'
        else:
            _l0, _tr0 = list(self.traces.items())[0]
            _lf, _trf = list(self.traces.items())[-1]
            rstr += f'{_l0:} : {_tr0.__repr__(id_length=id_length)}\n'
            rstr += f'...\n({len(self.traces) - 2} other traces)\n...\n'
            rstr += f'{_lf:} : {_trf.__repr__(id_length=id_length)}\n'
            rstr += f'[Use "print({type(self).__name__}.__repr__(extended=True))" to print all labels and FoldTraces]'
        return rstr

    ######################
    ## SUB-VIEW METHODS ##
    ######################

    def split_on(self, id_key='instrument') -> dict:
        """Split this DictStream into a dictionary
        of subset DictStreams based on one of the id_keys
        key names in the :class:`~PULSE.util.header.MLStats`
        **id_keys** attribute. 
        
        .. note::
        FoldTrace objects in subset DictStreams are views of the original data.
        Use :meth:`~.DictStream.copy` to create independent copies.

        Keys
        ----
        'nslc' - N.S.L.C SEED channel naming convention
        'sncl' - S.N.C.L Earthworm channel naming convention
        'id' - N.S.L.C(.M.W) - MLStats extension that may include
                model and weight names.
        'site' - Site defining elements of the SEED channel naming
                convention (N.S)
        'instrument' - Instrument defining elements of the SEED channel
                naming convention (N.S.L.C [minus component code])
        'modwt' - Model + Weight elements of the MLStats extension
                to the SEED naming convention
        'network' - SEED Network code (N)
        'station' - SEED Station code (S)
        'location' - SEED Location code (L)
        'channel' - SEED Channel code (C)
        'component' - SEED Component character of Channel Code
        'model' - MLStats model name
        'weight' - MLSTats weight name
                
        :param id_key: FoldTrace.id_keys key value to
            used to split this DictStream, defaults to 'instrument'
        :type id_key: str, optional
        :return: 
         - **output** (*dict*) -- dictionary of subset
            DictStream views keyed by unique **id_key**
            values from the originating DictStream
        """
        if id_key not in self.supported_keys:
            raise KeyError(f'id_key {id_key} not supported. See DictStream().supported_keys')
        else:
            output = {}
            for _ft in self:
                # Get trace key
                _k = _ft.id_keys[id_key]
                if _k not in output:
                    output.update({_k: DictStream(key_attr=self.key_attr)})
                output[_k].extend(_ft)
        return output

    #####################################################################
    # SEARCH METHODS ####################################################
    #####################################################################
 
        

    def select(self, method='intersection', inverse=False, **kwargs):
        """Revised version of :meth:`~obspy.core.stream.Stream.select`
        that uses set-logic to create a subset view of this
        DictStream's contents.
        
        :param method: method for joining sets created by each
            specified key-word argument (see below), defaults to 'intersection'
            Supported:
             - 'intersection': common traces to all search criteria
                "inner join" in SQL parlance
                accepted aliases: "&"
             - 'symmetric_difference': unique traces across all search criteria
                "exclusive full join" in SQL parlance
                accepted aliases: "^"
             - 'union': all traces 
                "inclusive full join" in SQL parlance
                accepted aliasees: "|"
            all are methods of :class:`~set`
        :type method: str, optional
        :param inverse: should the inverse set of the final subset be returned?
            Defaults to False
        :type inverse: bool, optional
        :param kwargs: key-word argument collector. The order of kwargs
            dictates the order in which individual sets are joined

        Accepted key-word arguments include:
          - id (*str*)
          - nslc (*str*)
          - sncl (*str*)
          - network (*str*)
          - station (*str*)
          - location (*str*)
          - channel (*str*)
          - component (*str*)
          - model (*str*)
          - weight (*str*)
          - site (*str*)
          - inst (*str*)
          - mod (*str*)
          - sampling_rate (*float*)
          - delta (*float*)
          - calib (*float*)
          - npts (*int*)

        In Development
          - inventory (*obspy.core.inventory.Inventory*)

        All *str*-type values are Unix-style wildcard compliant,
        All *float* and *int* values are tolerant to mismatched
        numeric types

        :returns:
         - **ds_view** (*PULSE.data.dictstream.DictStream**) - subset view
            of this DictStream based on search parameters (or its inverse)
        """
        # Try to run subset
        try:      
            match_set = self._search(method=method, **kwargs)
        # Telegraph internal errors to this level.
        except: 
            raise
        # Apply inverse if True
        if inverse:
            match_set = self._inverse_set(match_set)
        ds_view = self[match_set]
        return ds_view
    
    def _inverse_set(self, subset):
        """Return the inverse set of FoldTrace IDs from this DictStream
        for a given subset of FoldTrace IDs.

        :param subset: subset to difference from the set of IDs in this DictStream
        :type subset: set
        :return:
         - **inverse_set** (*set*) -- inverse set of FoldTrace ID keys
        """
        subset = self._check_subset(subset)
        inverse_set = set(self.get_keys()).difference(subset)
        return inverse_set
    
    def _search(self, method='intersection', **kwargs):
        """
        PRIVATE METHOD

        Create a subset of **traces.keys()** from this
        DictStream based on one or more

        :param method: set joining method, defaults to 'intersection'
            Supported:
             - 'intersection': traces meet all criteria
                "inner join" in SQL parlance
                accepted aliases: "&"
             - 'union': all traces that meet any criteria
                "inclusive full join" in SQL parlance
                accepted aliasees: "|"
            In Development:
            - 'difference': traces meet only one criterion
                "exclusive full join" in SQL parlance
                accepted aliases: "^"
        :type method: str, optional
        :param inverse: return the inverse set of the final subset?
            Defaults to False.
        :type inverse: bool, optional.
        :return:
         - **ds_view** (*PULSE.data.dictstream.DictStream*) -- 
            subset view of this dictstream (or it's inverse)
        """
        inverse = False
        # Parse operator aliases
        if method == '&':
            method = 'intersection'
        elif method == '|':
            method = 'union'
        elif method in ['^', 'difference']:
            raise NotImplementedError('difference method in development')
            # method = 'intersection'
            # inverse = True
        elif method in ['intersection','union']:
            pass
        else:
            raise ValueError(f'method "{method}" not supported.')

        # Handle null-search
        if len(kwargs) == 0:
            match_set = set(self.traces.keys())
            warnings.warn('No search parameters provided. Returning full set.')
        # Reducing methods
        elif method.lower() == 'intersection':
            match_set = set(self.traces.keys())
            subset = match_set
        # Agglomorating method
        elif method.lower() == 'union':
            match_set = set()
            subset = None
        else:
            raise ValueError(f'method {method} not supported.')
        
        # Iterate across kwargs
        for _k, _v in kwargs.items():
            # Clear out None-type kwargs
            if _v is None:
                kwargs.pop(_k)
                continue
            elif _k == 'inventory':
                iset = self._match_inventory(_v, subset=subset)
            elif _k in self.supported_keys:
                iset = self._search_ids(_k, _v, subset=subset)
            elif _k in ['sampling_rate','npts','delta','calib']:
                iset = self._match_stats(_k, _v, subset=subset)
            else:
                raise SyntaxError(f'Unexpected key-word argument "{_k}".')
            match_set = getattr(match_set, method)(iset)

        if inverse:
            match_set = self._inverse_set(match_set)

        return match_set        

    
    def _check_subset(self, subset=None):
        """Run formatting/compatability checks on `subset`

        :param subset: subset of keys in **traces**, defaults to None
        :type subset: NoneType, set, list, tuple, dict_keys, optional
        :raises TypeError: If subset type is non-conforming
        :return:
         - **subset** (*set*) -- formatted subset
           if None is input, returns the set-type representation of
           all keys in this DictStream's **traces** attribute
        """        
        if subset is None:
            subset = set(self.get_keys())
        elif isinstance(subset, (list, tuple, type(dict().keys()))):
            subset = set(subset)
        elif isinstance(subset, set):
            pass
        else:
            raise TypeError('subset must be set-like')
        return subset

    def _match_stats(self, key, value, subset=None):
        """Search for the id's of FoldTraces in this DictStream that match
        the specified stats.attributes subset values

        Supported keys:
         - sampling_rate (*float*)
         - npts (*int*)
         - calib (*float*)
         - delta (*float*)

        :param key: attribute key to query
        :type key: str
        :param value: sought attribute value
        :type key: int or float
        :param subset: subset list of **traces** keys to truncate
            the search, defaults to None
            None input starts search with all keys of the **traces**
            attribute of this DictStream
        :type subset: set or NoneType, optional.
        :return:
         - **match_set** (*set*) - set of matching keys from
            DictStream.traces
        """
        subset = self._check_subset(subset=subset)

        # Create output holder
        match_set = set()
        # Clear out NoneType kwargs
        if key not in ['npts','sampling_rate','calib','delta']:
            raise KeyError(f'key {key} not supported.')
        elif key == 'npts':
            if isinstance(value, (int, float)):
                value = int(value)
            else:
                raise ValueError('value for key "npts" must be int-like')
        else:
            if isinstance(value, (int, float)):
                value = float(value)
            else:
                raise ValueError(f'value for key "{key}" must be float-like')
        match_set = {_ft.id_keys[self.key_attr] for _ft in self[subset] if _ft.stats[key] == value}
        return match_set
    
    def _search_ids(self, key, value, subset=None):
        """Search for matching id_key values to a 
        specified pattern using :meth:`~fnmatch.fnmatch`

        Includes an option to subset

        :param id_key: id_key key to use for subset
        :type id_key: str
        :param pat: pattern to match id_key values to
        :type pat: str
        :param subset: subset of trace keys (DictStream.traces.keys)
           to limit this subset to, defaults to None.
           None uses the full set of traces.keys().
        :type subset: set or NoneType, optional
        :return: matched set of traces.keys value
        :rtype: set
        """ 
        if not isinstance(value, str):
            raise ValueError('value must be type str')
        if key not in self.supported_keys:
            raise KeyError(f'key "{key}" not supported. See DictStream().supported_keys')
        # If checking full set
        if subset is None:
            # If running on key_attr, leverage get_keys() to get pre-existing set
            if key == self.key_attr:
                match_set = set(fnmatch.filter(self.get_keys(), value))
            # Otherwise run iteration
            else:
                match_set = {_ft.id_keys[self.key_attr] for _ft in self if fnmatch.fnmatch(_ft.id_keys[key], value)}
        else:
            subset = self._check_subset(subset)
            if key == self.key_attr:
                match_set = set(fnmatch.filter(self[subset].get_keys(), value))
            else:
                match_set = {_ft.id_keys[self.key_attr] for _ft in self[subset] if fnmatch.fnmatch(_ft.id_keys[key], value)}
        return match_set    
        
        
    



    ## TRACE PROCESSING METHODS ##

    def trim(self, starttime=None, endtime=None, pad=False,
             fill_value=None, nearest_sample=True,
             keep_empty_traces=True, apply_fill=True):
        """Trim the :class:`~PULSE.data.foldtrace.FoldTrace` objects in this
        :class:`~.DictStream` using a uniform set of inputs for
        :meth:`~PULSE.data.foldtrace.FoldTrace.trim`

        Modifications are made in-place. If you want to preserve the source data
        use the :meth:`~.DictStream.copy` method.

        :param starttime: start time for traces, defaults to None
        :type starttime: obspy.core.utcdatetime.UTCDateTime, optional
        :param endtime: end time for traces, defaults to None
        :type endtime: obspy.core.utcdatetime.UTCDateTime, optional
        :param pad: option to allow padding, defaults to False
        :type pad: bool, optional
        :param fill_value: fill value to assign to masked/padding values, defaults to None
        :type fill_value: scalar, optional
        :param nearest_sample: should trimming go to the nearest sample (True) or 
            strictly samples within specified starttime and endtime bounds (False)?
            Defaults to True
        :type nearest_sample: bool, optional
        :param keep_empty_traces: should empty traces be kept? Defaults to True
        :type keep_empty_traces: bool, optional
        :param apply_fill: should masked values be filled with fill_value? Defaults to True
        :type apply_fill: bool, optional
        """        
        for _id, _ft in self.traces.items():
            _ft.trim(starttime=starttime,
                     endtime=endtime,
                     pad=pad,
                     fill_value=fill_value,
                     nearest_sample=nearest_sample,
                     apply_fill=apply_fill)
            if _ft.count() > 0:
                self[_id] = _ft
            elif keep_empty_traces:
                self[_id] = _ft
            else:
                continue
        return self
    
    def view(self, starttime=None, endtime=None, keep_empty_traces=True):
        """Apply the :meth:`~PULSE.data.foldtrace.FoldTrace.view` method
        to all :class:`~PULSE.data.foldtrace.FoldTrace` objects in this
        :class:`~.DictStream` with identical arguments

        Views reference the **data** and **fold** values of FoldTraces
        in the originating DictStream, so any modifications made to the
        output of this method are made to the source data.

        :param starttime: start time for the view, defaults to None
        :type starttime: obspy.UTCDateTime or None, optional
        :param endtime: end time for the view, defaults to None
        :type endtime: obspy.UTCDateTime or None, optional
        :param keep_empty_traces: should dataless traces be kept in
            this view? Defaults to True
        :type keep_empty_traces: bool, optional
        :returns: **view** (*PULSE.data.dictstream.DictStream**) -- view
            of specified data and metadata
        """        
        out = self.__class__(key_attr=self.key_attr)
        for _ft in self:
            view = _ft.view(starttime=starttime, endtime=endtime)
            if view.count() > 0:
                out.extend(view)
            elif keep_empty_traces:
                out.extend(view)
            else:
                continue
        return out

    def normalize(self, norm='max', global_norm=False):
        """Apply the :meth:`~PULSE.data.foldtrace.FoldTrace.normalize` method
        to all FoldTraces in this :class:`~.DictStream` object, with the added
        option to use the global maximum norm for all traces

        Alterations are made in-place. If you want to preserve the initial data
        use the :meth:`~.DictStream.copy` method

        :param norm: norm type to use, defaults to 'max'
            Supported values: 
             - 'max' - maximum amplitude
             - 'std' - standard deviation               
        :type norm: str, optional
        :param global_norm: Should the largest norm be used to normalize all traces?
            Defaults to False
        :type global_norm: bool, optional
        """        
        if norm in ['max','minmax','peak','std','standard','sigma']:
            pass
        elif isinstance(norm, (int, float)):
            if norm > 0:
                pass
            else:
                raise ValueError('scalar numeric norms must be positive values')
        else:
            raise TypeError(f'norm {norm} of type {type(norm)} not supported')
        
        if global_norm:
            if norm in ['max','minmax','peak']:
                norm = max([_ft.max() for _ft in self])
            elif norm in ['std','standard']:
                norm = max([_ft.data.std() for _ft in self])
            else:
                pass

        for _ft in self:
            _ft.normalize(norm=norm)
        return self
    
    def blind(self, npts):
        """Apply blinding to all FoldTraces in this DictStream
        using :meth:`~.FoldTrace.blind`
        """
        for _ft in self:
            _ft.blind(npts)


    # def filter(self, *args, **kwargs):
    #     Stream.filter(self, *args, **kwargs)

    
        # def update(self, updatedict):
    #     """update method for DictStream
    #     Waps the :meth:`~.DictStream.__setitem__` method

    #     :param updatedict: _description_
    #     :type updatedict: _type_
    #     """        
    #     for _k, _v in updatedict:
    #         self.__setitem__(_k, _v)

    # def get_key_set(self):
    #     return list(self.traces.keys())
    
    # keys = property(get_key_set)


               # def select(self, **kwargs):
    #     # Compatability checks
    #     for _k, _v in kwargs.items():
    #         if _k in self.supported_keys:
    #             if isinstance(_v, str):
    #                 pass
    #             else:
    #                 raise TypeError(f'Type for {_k} is not supported. Must be type str')
    #         elif _k in MLStats._types.keys():
    #             if isinstance(_v, MLStats._types[_k]):
    #                 pass
    #             else:
    #                 raise TypeError(f'Type for {_k} is not supported')
    #         else:
    #             raise KeyError(f'Unexpected key-word argument {_k}')    


    # def id_subset(self, id=None, subset=None):
    #     """Basic subset routine using :meth:`~fnmatch.filter` on
    #     sets of trace keys (IDs) from this DictStream's traces and
    #     the subset matching an input idstring.

    #     idstring accepts UNIX wildcard statements

    #     .. Note::
    #         The :class:`~PULSE.data.foldtrace.FoldTrace` objects contained in
    #         the **output** :class:`~.DictStream` object are views of the original
    #         data. Thus, changes made to the contents of **output** change the
    #         contents of the originating DictStream. Use the :meth:`~.DictStream.copy`
    #         method on the output or source to make new in-memory copies of their
    #         contents.

    #     :param idstring: wildcard-compliant ID string to use for subsetting
    #         this DictStream, defaults to '*'
    #     :type idstring: str, optional
    #     :return:
    #         - **keyset** (*set*) - subset FoldTrace key_attr values (DictStream.traces.keys())
    #             matching the provided idstring
    #     """
    #     if subset is None:
    #         subset = set(self.traces.keys())
    #     elif isinstance(subset, set):
    #         pass
    #     else:
    #         raise TypeError('subset must be type set or NoneType') 
    #     if id is None:
    #         match_set = subset
    #     elif isinstance(id, str):
    #         match_set = set(fnmatch.filter(subset, id))
    #     else:
    #         raise TypeError('id must be type str or NoneType')
    #     return match_set



    # def _match_id_keys(self, id_key, pat=None, subset=None):
    #     """Search for matching id_key values to a 
    #     specified pattern using :meth:`~fnmatch.fnmatch`

    #     Includes an option to subset

    #     :param id_key: id_key key to use for subset
    #     :type id_key: str
    #     :param pat: pattern to match id_key values to
    #     :type pat: str
    #     :param subset: subset of trace keys (DictStream.traces.keys)
    #        to limit this subset to, defaults to None.
    #        None uses the full set of traces.keys().
    #     :type subset: set or NoneType, optional
    #     :return: matched set of traces.keys value
    #     :rtype: set
    #     """        
    #     # Compatability check on key
    #     if id_key not in self.supported_keys:
    #         raise KeyError(f'{id_key} not included in FoldTrace.id_keys.')
        
    #     # Compatability check on pat
    #     if not isinstance(pat, str):
    #         raise TypeError('pat must be type str')
        
    #     # Compatability check on subset
    #     if subset is None:
    #         subset = set(self.traces.keys())
    #     elif isinstance(subset, set):
    #         pass
    #     else:
    #         raise TypeError('subset must be type set or NoneType')

    #     # if id_key matches the key_attr for this DictStream, 
    #     # use faster fnmatch.filter method
    #     if id_key == self.key_attr:
    #         match_set = set(fnmatch.filter(self[subset].traces.keys(), pat))
    #     # otherwise iterate over traces and find matches
    #     # with the fnmatch.fnmatch method
    #     else:
    #         match_set = set()
    #         for _k in subset:
    #             if fnmatch.fnmatch(self[_k].id_keys[id_key], pat):
    #                 match_set.update([_k])

    #     return match_set
        



    #     # id_key_dict = {_k: _ft.id_keys[key] for _k, _ft in self.traces.items()}
    #     # for _id, _ft in self.traces.items():
    #     #     if fnmatch.fnmatch(_ft.id_keys[key])
    #     # for _id, _ft in self.traces.items():
    #     #     if fnmatch.fnmatch(_ft.id_keys[key], value):
    #     #         matches.update(_id)
    
    # def _match_inventory(key_set, inv):
    #     if not isinstance(inv, Inventory):
    #         raise TypeError('inv must be type obspy.core.inventory.Inventory')
    #     raise NotImplementedError('Work In Progress')
    #     # if isinstance(inventory, Inventory):
    #     #     contents = inventory.get_contents()
    #     #     if len(contents['channels']) > 0:




    # def key_search(self, strings, key_attr=None, ascopy=False, inverse=False):
    #     """Return a subset of FoldTraces that match at least one unix-wildcard-compliant
    #     string contained in strings as a new :class:`~PULSE.data.dictstream.DictStream` object.

    #     Options are provided for conducting inverse searches and whether the subset
    #     :class:`~PULSE.data.foldtrace.FoldTrace` object(s) are views of the originals or
    #     deepcopy copies of the originals.
        
    #     :param strings: Unix-wildcard-compliant string, or iterable set thereof, to use for searching for matching keys
    #     :type strings: str or list-like set thereof
    #     :param ascopy: should the subset of FoldTrace(s) be deepcopies of the originals? Defaults to False,
    #         which provides a view of the original FoldTrace objects (i.e., identical in-memory objects).
    #     :type ascopy: bool, optional
    #     :param inverse: should the subset be composed of items that *do not* match anything in **strings**? Defaults to False.
    #     :type inverse: bool, optional
    #     :return:
    #         **out** (*PULSE.data.dictstream.DictStream*) -- new DictStream object containing a subset view or copy
    #                 selected FoldTrace objects.
        
    #     .. rubric:: Selecting and Inverse Selecting
    #     >>> dst.key_select('*.*.*.??[NE].*')
    #     --Stats--
    #         common_id: BW.RJOB.--.EH?..
    #     min_starttime: 2009-08-24T00:20:03.000000Z
    #     max_starttime: 2009-08-24T00:20:03.000000Z
    #         min_endtime: 2009-08-24T00:20:32.990000Z
    #         max_endtime: 2009-08-24T00:20:32.990000Z
    #         processing: []
    #     -------
    #     2 FoldTrace(s) in DictStream
    #     BW.RJOB.--.EHN.. : BW.RJOB.--.EHN.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    #     BW.RJOB.--.EHE.. : BW.RJOB.--.EHE.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    #     >>> dst.fnselect('*.*.*.??[NE].*', inverse=True)
    #     --Stats--
    #         common_id: BW.RJOB.--.EHZ..
    #     min_starttime: 2009-08-24T00:20:03.000000Z
    #     max_starttime: 2009-08-24T00:20:03.000000Z
    #         min_endtime: 2009-08-24T00:20:32.990000Z
    #         max_endtime: 2009-08-24T00:20:32.990000Z
    #         processing: []
    #     -------
    #     1 FoldTrace(s) in DictStream
    #     BW.RJOB.--.EHZ.. : BW.RJOB.--.EHZ.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
        
    #     Subset views of **DictStream.traces** contents can also be generated quickly using unix-wildcard-compliant
    #     search strings using the :meth:`~PULSE.data.dictstream.DictStream.key_select` method. This tends to be faster
    #     than than ObsPy Stream objects' :meth:`~obspy.core.stream.Stream.select` method, particularly for large sets
    #     of (ML)Trace objects. See :meth:`~PULSE.data.dictstream.DictStream.key_select` documentation for more information.

    #     """
    #     # Compatability check for `strings`
    #     if isinstance(strings, str):
    #         strings = [strings]
    #     elif isinstance(strings, (list, tuple)):
    #         if all(isinstance(_e, str) for _e in strings):
    #             pass
    #         else:
    #             raise TypeError('All elements of a list-like `strings` must be type str')
    #     else:
    #         raise TypeError('strings must be type str or a list-like thereof.')
    #     if not isinstance(inverse, bool):
    #         raise TypeError('inverse must be type bool')

    #     # Initialize matches as a set
    #     matches = set()
    #     # Get traces keys as a set
    #     tkeys = set(self.traces.keys())
    #     # Iterate across each string in strings
    #     for _e in strings:
    #         # Get items that match the current string
    #         imatches = fnmatch.filter(tkeys, _e)
    #         # Update matches set with output of iterative match
    #         matches.update(imatches)
    #     # If doing inverse search, use :meth:`~set.difference_update` to remove all matches from the traces keys set
    #     if inverse:
    #         matches = tkeys.difference_update(matches)

    #     # Iterate across (inverse)matched keys
    #     traces = []
    #     for _m in matches:
    #         if ascopy:
    #             traces.append(self.traces[_m].copy())
    #         else:
    #             traces.append(self.traces[_m])

    #     # Initialize new DictStream
    #     out = self.__class__(traces=traces, header=self.stats.copy(), key_attr = self.key_attr)
    #     return out














        
        # for _m in matches

        # matches = fnmatch.filter(self.traces.keys(), strings)
        # out = self.__class__(header=self.stats.copy(), key_attr = self.key_attr)
        # for _m in matches:
        #     if ascopy:
        #         _tr = self.traces[_m].copy()
        #     else:
        #         _tr = self.traces[_m]
        #     out.extend(_tr, key_attr=key_attr)
        # out.stats.common_id = out.get_common_id()
        # return out
    
    # def fnpop(self, fnstr, key_attr=None):
    #     """Use a fnmatch.filter() search for keys that match the provided
    #     fn string and pop them off this DictStream-like object into a new 
    #     DictStream-like object

    #     :param fnstr: unix wildcard compliant string for matching to keys that will be popped off this DictStream object
    #     :type fnstr: str
    #     :param key_attr: alternative key attribute to use for the output DictStream object, defaults to None
    #     :type key_attr: NoneType or str, optional
    #     :return out: popped items hosted in a new DictStream-like object
    #     :rtype: PULSE.data.dictstream.DictStream-like
    #     """        
    #     matches = fnmatch.filter(self.traces.keys(), fnstr)
    #     out = self.__class__(header=self.stats.copy(), key_attr=self.key_attr)
    #     for _m in matches:
    #         out.extend(self.pop(_m), key_attr = key_attr)
    #     out.stats.common_id = out.get_common_id()
    #     return out
            
    


    # def isin(self, iterable, key_attr=None, ascopy=False):
    #     """Return a subset view (or copy) of the contents of this
    #     DictStream with keys that conform to an iterable set
    #     of strings.

    #     Generally based on the behavior of the pandas.series.Series.isin() method

    #     :: INPUTS ::
    #     :param iterable: [list-like] of [str] - strings to match
    #                         NOTE: can accept wild-card strings 
    #                             (also see DictStream.fnselect)
    #     :param key_attr: [str] - key attribute for indexing this view/copy
    #                     of the source DictStream (see DictStream.__init__)
    #                      [None] - defaults to the .key_attr attribute
    #                      value of the source DictStream object
    #     :param ascopy: [bool] return as an independent copy of the subset?
    #                     default - False
    #                         NOTE: this creates a view that can
    #                         alter the source DictStream's contents
    #     :: OUTPUT ::
    #     :return out: [PULSE.data.dictstream.DictStream] subset view/copy

    #     TODO: Merge this into select
    #     """
    #     out = self.__class__(header=self.stats.copy(), key_attr = self.key_attr)
    #     matches = []
    #     for _e in iterable:
    #         matches += fnmatch.filter(self.traces.keys(), _e)
    #     for _m in matches:
    #         if ascopy:
    #             _tr = self.traces[_m].copy()
    #         else:
    #             _tr = self.traces[_m]
    #         out.extend(_tr, key_attr=key_attr)
    #     out.stats.common_id = out.get_common_id()
    #     return out

    # def fnexclude(self, iterable, ascopy=False):
    #     """Create a new :class:`~PULSE.data.dictstream.DictStream` object containing
    #     and inverse-search set of :class:`~PULSE.data.foldtrace.FoldTrace` objects using
    #     unix-wildcard-compliant search string(s) to match keys in **DictStream.traces**
    #     using the :meth:`~fnmatch.filter` method.

    #     This is the inverse of the :meth:`~PULSE.data.dictstream.DictStream.fnselect` method

    #     :param iterable: one or many fnmatch-compliant strings
    #     :type iterable: str, or list-like thereof
    #     :param ascopy: should the output DictStream contain deepcopys of the original FoldTraces
    #         that do not match string(s) in **iterable**? Defaults to False.
    #     :type ascopy: bool, optional
    #     :returns:
    #         **out** -- (*PULSE.data.dictstream.DictStream*) - inverse subset of FoldTraces

    #     .. rubric:: E.g., exclude North and East components
    #     >>> dst = DictStream(traces=read())
    #     >>> dst.fnexclude('*.*.*.??[NE].*')
    #     --Stats--
    #         common_id: BW.RJOB.--.EHZ..
    #     min_starttime: 2009-08-24T00:20:03.000000Z
    #     max_starttime: 2009-08-24T00:20:03.000000Z
    #       min_endtime: 2009-08-24T00:20:32.990000Z
    #       max_endtime: 2009-08-24T00:20:32.990000Z
    #        processing: []
    #     -------
    #     1 FoldTrace(s) in DictStream
    #     BW.RJOB.--.EHZ.. : BW.RJOB.--.EHZ.. | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples

    #     """        
    #     out = self.__class__(header=self.stats.copy(), key_attr = self.key_attr)
    #     matches = []
    #     if isinstance(iterable, str):
    #         iterable = [iterable]
    #     elif isinstance(iterable, (list, tuple)):
    #         if all(isinstance(_e, str) for _e in iterable):
    #             pass
    #         else:
    #             raise TypeError(f'list-like iterable must comprise only string elements')
    #     else:
    #         raise TypeError('iterable must be type str or a list-like thereof.')
    #     for _e in iterable:
    #         matches += fnmatch.filter(self.traces.keys(), _e)
    #     for _k in self.traces.keys():
    #         if _k not in matches:
    #             if ascopy:
    #                 _tr = self.traces[_k].copy()
    #             else:
    #                 _tr = self.traces[_k]
    #             out.extend(_tr, key_attr=self.key_attr)
    #     out.stats.common_id = out.get_common_id()
    #     return out


    # def _get_unique_id_elements(self):
    #     """Compose a dictionary containing lists of unique id elements: Network, Station, Location, Channel, Model, Weight in this DictStream

    #     :return:
    #       - **out** (*dict* -- output dictionary keyed by the above elements and valued as lists of strings
    #     """
    #     N, S, L, C, M, W = [], [], [], [], [], []
    #     for _tr in self:
    #         hdr = _tr.stats
    #         if hdr.network not in N:
    #             N.append(hdr.network)
    #         if hdr.station not in S:
    #             S.append(hdr.station)
    #         if hdr.location not in L:
    #             L.append(hdr.location)
    #         if hdr.channel not in C:
    #             C.append(hdr.channel)
    #         if hdr.model not in M:
    #             M.append(hdr.model)
    #         if hdr.weight not in W:
    #             W.append(hdr.weight)
    #     out = dict(zip(['network','station','location','channel','model','weight'],
    #                    [N, S, L, C, M, W]))
    #     return out
    
    # def _get_common_id_elements(self):
    #     """
    #     Return a dictionary of strings that are 
    #     UNIX wild-card representations of a common
    #     id for all traces in this DictStream. I.e.,
    #         ? = single character wildcard
    #         * = unbounded character count widlcard

    #     :: OUTPUT ::
    #     :return out: [dict] dictionary of elements keyed
    #                 with the ID element name
    #     """
    #     ele = self._get_unique_id_elements()
    #     out = {}
    #     for _k, _v in ele.items():
    #         if len(_v) == 0:
    #             out.update({_k:'*'})
    #         elif len(_v) == 1:
    #             out.update({_k: _v[0]})
    #         else:
    #             minlen = 999
    #             maxlen = 0
    #             for _ve in _v:
    #                 if len(_ve) < minlen:
    #                     minlen = len(_ve)
    #                 if len(_ve) > maxlen:
    #                     maxlen = len(_ve)
    #             _cs = []
    #             for _i in range(minlen):
    #                 _cc = _v[0][_i]
    #                 for _ve in _v:
    #                     if _ve[_i] == _cc:
    #                         pass
    #                     else:
    #                         _cc = '?'
    #                         break
    #                 _cs.append(_cc)
    #             if all(_c == '?' for _c in _cs):
    #                 _cs = '*'
    #             else:
    #                 if minlen != maxlen:
    #                     _cs.append('*')
    #                 _cs = ''.join(_cs)
    #             out.update({_k: _cs})
    #     return out

    # def get_common_id(self):
    #     """
    #     Get the UNIX wildcard formatted common common_id string
    #     for all traces in this DictStream

    #     :: OUTPUT ::
    #     :return out: [str] output stream
    #     """
    #     ele = self._get_common_id_elements()
    #     out = '.'.join(ele.values())
    #     return out

    # def update_stats_timing(self):
    #     for tr in self:
    #         self.stats.update_time_range(tr)
    #     return None                
    
    # def split_on_key(self, key='instrument', **options):
    #     """
    #     Split this DictStream into a dictionary of DictStream
    #     objects based on a given element or elements of the
    #     constituient traces' ids.

    #     :: INPUTS ::
    #     :param key: [str] name of the attribute to split on
    #                 Supported:
    #                     'id', 'site','inst','instrument','mod','component',
    #                     'network','station','location','channel','model','weight'
    #     :param **options: [kwargs] key word argument gatherer to pass
    #                     kwargs to DictStream.__add__()
    #     :: OUTPUT ::
    #     :return out: [dict] of [DictStream] objects
    #     """
    #     if key not in FoldTrace().key_opts.keys():
    #         raise ValueError(f'key {key} not supported.')
    #     out = {}
    #     for _tr in self:
    #         key_opts = _tr.key_opts
    #         _k = key_opts[key]
    #         if _k not in out.keys():
    #             out.update({_k: self.__class__(traces=_tr)})
    #         else:
    #             out[_k].__add__(_tr, **options)
    #     return out
    

    # #####################################################################
    # # UPDATED METHODS FROM OBSPY STREAM #######################################
    # #####################################################################
    # # @_add_processing_info
    # def trim(self,
    #          starttime=None,
    #          endtime=None,
    #          pad=True,
    #          keep_empty_traces=True,
    #          nearest_sample=True,
    #          fill_value=None):
    #     """
    #     Slight adaptation of :meth: `~obspy.core.stream.Stream.trim` to accommodate to facilitate the dict-type self.traces
    #     attribute syntax.

    #     see obspy.core.stream.Stream.trim() for full explanation of the arguments and behaviors
        
    #     :: INPUTS ::
    #     :param starttime: [obspy.core.utcdatetime.UTCDateTime] or [None]
    #                     starttime for trim on all traces in DictStream
    #     :param endtime: [obspy.core.utcdatetime.UTCDateTime] or [None]
    #                     endtime for trim on all traces in DictStream
    #     :param pad: [bool]
    #                     should trim times outside bounds of traces
    #                     produce masked (and 0-valued fold) samples?
    #                     NOTE: In this implementation pad=True as default
    #     :param keep_empty_traces: [bool]
    #                     should empty traces be kept?
    #     :param nearest_sample: [bool]
    #                     should trim be set to the closest sample(s) to 
    #                     starttime/endtime?
    #     :param fill_value: [int], [float], or [None]
    #                     fill_value for gaps - None results in masked
    #                     data and 0-valued fold samples in gaps

    #     """
    #     if not self:
    #         return self
    #     # select start/end time fitting to a sample point of the first trace
    #     if nearest_sample:
    #         tr = self[0]
    #         try:
    #             if starttime is not None:
    #                 delta = compatibility.round_away(
    #                     (starttime - tr.stats.starttime) *
    #                     tr.stats.sampling_rate)
    #                 starttime = tr.stats.starttime + delta * tr.stats.delta
    #             if endtime is not None:
    #                 delta = compatibility.round_away(
    #                     (endtime - tr.stats.endtime) * tr.stats.sampling_rate)
    #                 # delta is negative!
    #                 endtime = tr.stats.endtime + delta * tr.stats.delta
    #         except TypeError:
    #             msg = ('starttime and endtime must be UTCDateTime objects '
    #                    'or None for this call to Stream.trim()')
    #             raise TypeError(msg)
    #     for trace in self:
    #         trace.trim(starttime, endtime, pad=pad,
    #                    nearest_sample=nearest_sample, fill_value=fill_value)
    #         self.stats.update_time_range(trace)
    #     if not keep_empty_traces:
    #         # remove empty traces after trimming
    #         self.traces = {_k: _v for _k, _v in self.traces.items() if _v.stats.npts}
    #         self.stats.update_time_range(trace)
    #     self.stats.common_id = self.get_common_id()
    #     return self
    
    # # 
    # def normalize_traces(self, norm_type ='peak'):
    #     """Normalize traces in this :class:`~PULSE.data.dictstream.DictStream`, using :meth:`~PULSE.data.foldtrace.FoldTrace.normalize` on each trace

    #     :param norm_type: normalization method, defaults to 'peak'
    #     :type norm_type: str, optional
    #         Supported values: 'peak', 'std'
    #     """        
    #     for tr in self:
    #         tr.normalize(norm_type=norm_type)
    
    # #####################################################################
    # # I/O METHODS #######################################################
    # #####################################################################


    # # TODO: Determine if this method is sufficient for development purposes
    # def write(self, base_path='.', path_structure='foldtraces', name_structure='{wfid}_{iso_start}', **options):
    #     """
    #     Write a DictStream object to disk as a series of MSEED files using the FoldTrace.write() method/file formatting
    #     in a prescribed directory structure

    #     :: INPUTS ::
    #     :param base_path: [str] path to the directory that will contain the save file structure. If it does
    #                 not exist, a directory (structure) will be created
    #     :type base_path: str
    #     :param path_structure: [None] - no intermediate path structure
    #                             [str] - format string based on the metadata of individual FoldTrace objects contained
    #                                     in this DictStream. In addition to standard kwargs in the FoldTrace.stats
    #                                     that can be used as elements of this format string, additional options are
    #                                     provided for datetime information:
    #                                         epoch_start - starttimes converted into a timestamp
    #                                         epoch_end - endtimes converted into timestamps
    #                                         iso_start - starttimes converted into isoformat strings
    #                                         iso_ends - endtimes converted into isoformat strings
    #                                         wfid - waveform ID (Net.Sta.Loc.Chan.Mod.Wgt)
    #                                         site - Net.Sta code string
    #                                         inst - Loc.Chan (minus the component character) code string
    #                                         mod - Mod.Wgt code string
    #                                         instrument - site.inst (as defined above) code string
    #     :type path_structure: str or NoneType
                                        
    #     :param name_structure: [str] - format string with the opions as described for path_structure
    #     :param **options: [kwargs] optional key word argument collector for 

    #     :ATTRIBUTION: Based on path sturcturing and syntax from the ObsPlus WaveBank class
        
    #     """
    #     # # Ensure OS-appropriate path formatting
    #     # base_parts = os.path.split(base_path)
    #     # base_path = os.path.join(base_parts)

    #     # Get elements of the save directory structure as an OS-agnostic list of directory names
    #     if path_structure is None:
    #         path_parts = [base_path]
    #     else:
    #         path_parts = [base_path] + path_structure.split('/')
    #     # Iterate across traces in this DictStream
    #     for tr in self.traces.values():
    #         # Ge the formatting dictionary 
    #         fmt_dict = {'wfid': tr.id,
    #                     'epoch_start': tr.stats.starttime.timestamp,
    #                     'iso_start': tr.stats.starttime.isoformat(),
    #                     'epoch_end': tr.stats.endtime.timestamp,
    #                     'iso_end': tr.stats.endtime.isoformat()}
    #         fmt_dict.update(tr.stats)
    #         if isinstance(tr, FoldTrace):
    #             fmt_dict.update({'component': tr.comp,
    #                              'site': tr.site,
    #                              'inst': tr.inst,
    #                              'mod': tr.mod,
    #                              'instrument': tr.instrument})

    #         save_path = os.path.join(*path_parts).format(**fmt_dict)
    #         save_name = f'{name_structure.format(**fmt_dict)}.mseed'
    #         file_name = os.path.join(save_path, save_name)
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)
    #         tr.write(file_name=file_name, **options)

    # def read(foldtrace_mseed_files):
    #     """
    #     Read a list-like set of file names for mseed files created by a
    #     FoldTrace.write() call (i.e., MSEED files containing a data and fold trace)

    #     also see :meth:`~PULSE.data.foldtrace.FoldTrace.write`
    #              :meth:`~PULSE.data.foldtrace.FoldTrace.read`
    #     """
    #     if isinstance(foldtrace_mseed_files, str):
    #         foldtrace_mseed_files = [foldtrace_mseed_files]
    #     dst = DictStream()
    #     for file in foldtrace_mseed_files:
    #         if not os.path.isfile(file):
    #             raise FileExistsError(f'file {file} does not exist')
    #         else:
    #             mltr = FoldTrace.read(file)
    #         dst.extend(mltr)
    #     return dst


    


    # #######################
    # # VISUALIZATION TOOLS #  
    # #######################
    # # TODO: Determine if this section should be migrated to a visualization submodule
    # def _to_vis_stream(self, fold_threshold=0, normalize_src_traces=True, attach_mod_to_loc=True):
    #     """PRIVATE METHOD - prepare a copy of the traces in this DictStream for visualization

    #     :param fold_threshold: _description_, defaults to 0
    #     :type fold_threshold: int, optional
    #     :param normalize_src_traces: _description_, defaults to True
    #     :type normalize_src_traces: bool, optional
    #     :param attach_mod_to_loc: _description_, defaults to True
    #     :type attach_mod_to_loc: bool, optional
    #     :return: _description_
    #     :rtype: _type_
    #     """        
    #     st = obspy.Stream()
    #     for mltr in self:
    #         tr = mltr.copy()
    #         if normalize_src_traces:
    #             if mltr.stats.weight == mltr.stats.defaults['weight']:
    #                 tr = tr.normalize(norm_type='max')
    #         st += tr.to_trace(fold_threshold=fold_threshold,
    #                           attach_mod_to_loc=attach_mod_to_loc)
    #     return st
    
    # def plot(self, fold_threshold=0, attach_mod_to_loc=True, normalize_src_traces=False, **kwargs):
    #     """Plot the contents of this DictStream using the obspy.core.stream.Stream.plot backend
        
    #     """
    #     st = self._to_vis_stream(fold_threshold=fold_threshold,
    #                              normalize_src_traces=normalize_src_traces,
    #                              attach_mod_to_loc=attach_mod_to_loc)
    #     outs = st.plot(**kwargs)
    #     return outs
    
    # def snuffle(self, fold_threshold=0, attach_mod_to_loc=True, normalize_src_traces=True,**kwargs):
    #     """Launch a snuffler instance on the contents of this DictStream

    #     NOTE: Imports from Pyrocko and runs :meth:`~pyrocko.obspy_compat.plant()`

    #     :param fold_threshold: fold_threshold for "valid" data (invalid data are masked), defaults to 0
    #     :type fold_threshold: float, optional
    #     :param attach_mod_to_loc: should model and weight names be appended to the trace's location string, defaults to True
    #     :type attach_mod_to_loc: bool, optional
    #     :param normalize_src_traces: normalized traces for traces that have default weight codes, defaults to True
    #     :type normalize_src_traces: bool, optional
    #     :return: standard output from :meth: `obspy.core.stream.Stream.snuffle` which is added to Stream
    #         via :class:`pyrocko.obspy_compat`
    #     :rtype: tuple
    #     """        
    #     if 'obspy_compat' not in dir():
    #         from pyrocko import obspy_compat
    #         obspy_compat.plant()
    #     st = self._to_vis_stream(fold_threshold=fold_threshold,
    #                              normalize_src_traces=normalize_src_traces,
    #                              attach_mod_to_loc=attach_mod_to_loc)
    #     outs = st.snuffle(**kwargs)
    #     return outs
                    
    

    # ####################
    # # Group Triggering #
    # ####################
    # # TODO: Determine if this should be migrated into a method associated with triggering
    # def prediction_trigger_report(self, thresh, exclude_list=None, **kwargs):
    #     """Wrapper around the :meth:`~PULSE.data.foldtrace.FoldTrace.prediction_trigger_report`
    #     method. This method executes the Trace-level method on each trace it contains using
    #     shared inputs

    #     :TODO: Update to the `notin` method (develop notin from the code in here)

    #     :param thresh: trigger threshold value
    #     :type thresh: float
    #     :param exclude_list: :meth: `~PULSE.data.dictstream.DictStream.notin` compliant strings
    #                     to exclude from trigger processing, defaults to None
    #     :type exclude_list: list of str, optional
    #     :param **kwargs: gatherer for key word arguments to pass to :meth: `~PULSE.data.foldtrace.FoldTrace.prediction_trigger_report`
    #     :type **kwargs: key word arguments, optional
    #     :return df_out: trigger report
    #     :rtype df_out: pandas.core.dataframeDataFrame
    #     """        
    #     df_out = pd.DataFrame()
    #     if 'include_processing_info' in kwargs.keys():
    #         include_proc = kwargs['include_processing_info']
    #     else:
    #         include_proc = False
    #     if include_proc:
    #         df_proc = pd.DataFrame()
    #     if exclude_list is not None:
    #         if isinstance(exclude_list, list) and all(isinstance(_e, str) for _e in exclude_list):
    #             view = self.exclude(exclude_list)
    #         else:
    #             raise TypeError('exclude_list must be a list of strings or NoneType')
    #     else:
    #         view = self
    #     for tr in view.traces.values():
    #         out = tr.prediction_trigger_report(thresh, **kwargs)
    #         # Parse output depening on output type
    #         if not include_proc and out is not None:
    #             idf_out = out
    #         elif include_proc and out is not None:
    #             idf_out = out[0]
    #             idf_proc = out[1]
    #         # Concatenate outputs
    #         if out is not None:
    #             df_out = pd.concat([df_out, idf_out], axis=0, ignore_index=True)
    #             if include_proc:
    #                 df_proc = pd.concat([df_proc, idf_proc], axis=0, ignore_index=True)
    #         else:
    #             continue
    #     if len(df_out) > 0:
    #         if include_proc:
    #             return df_out, df_proc
    #         else:
    #             return df_out
    #     else:
    #         return None


# TODO: Determine if this method should be removed
# def read_foldtraces(data_files, obspy_read_kwargs={}, add_options={}):
#     """
#     Wrapper around :meth:`~PULS# .data.foldtrace.FoldTrace.read_foldtrace`
#     to reconstitute multiple FoldTrace objects from the _DATA, _FOLD, _PROC
#     files generated by the FoldTrace.write() method and populate a DictStream
#     object. 

#     :: INPUTS ::
#     :param data_files: [str] file name of a 
#                     file that contains FoldTrace data and header information
#                        [list] list of valid data_file name strings
#                     also see PULS# .data.foldtrace.read_foldtrace()
#     :obspy_read_kwargs: [dict] dictionar of keyword arguments to pass to
#                     obspy.core.stream.read()
#     :add_options: [dict] dictionary of keyword arguments to pass to
#                     PULS# .data.dictstream.DictStream.__add__

#     :: OUTPUT ::
#     :return dst: [PULS# .data.dictstream.DictStream] assembled DictStream object
    
#     {common_name}_DATA.{extension} 
#     """
#     if isinstance(data_files, str):
#         data_files = [data_files]
#     elif not isinstance(data_files, list):
#         raise TypeError
#     else:
#         if not all(isinstance(_e, str) for _e in data_files):
#             raise TypeError
    
#     dst = DictStream()
#     for df in data_files:
#         mlt = read_foldtrace(df, **obspy_read_kwargs)
#         dst.__add__(mlt, **add_options)
#     return dst
        

# TODO: Determine if this class method should be removed
# def write_to_mseed(
#         self,
#         savepath='.',
#         save_fold=True,
#         fmt_str='{ID}_{t0:.3f}_{t1:.3f}_{sampling_rate:.3f}sps',
#         save_processing=True,
#         save_empty=False,
#         **kwargs):
#     split_outs = {}
#     for _mlt in self.traces.items():
#         if not save_empty and _mlt.stats.npts == 0:
#             continue
#         else:
#             out = _mlt.write_to_tiered_directory(
#                 savepath=savepath,
#                 save_fold=save_fold,
#                 save_processing=save_processing,
#                 fmt_str = fmt_str,
#                 **kwargs)
#             split_outs.update({_mlt.id: out})
#     return split_outs

# TODO: Determine if this class method should be removed
# def write(
#         self,
#         save_path='.',
#         fmt='sac',
#         save_fold=True,
#         save_processing=True,
#         filename_format='{ID}_{starttime}_{sampling_rate:.3f}sps',
#         **obspy_write_options):
#     """
#     Wrapper around the PULSE# .data.foldtrace.FoldTrace.write() method
#     that saves data as 
    
#     """
#     for _mlt in self.traces.values():
#         _mlt.write(save_path=save_path,
#                    fmt=fmt,
#                    save_fold=save_fold,
#                    save_processing=save_processing,
#                    filename_format=filename_format,
#                    **obspy_write_options)
        


    # def _add_trace(self, other, **options):
    #     """
    #     Add a trace-like object `other` to this DictStream using elements from
    #     the trace's id as the dictionary key in the DictStream.traces dictionary

    #     :: INPUTS ::
    #     :param other: ObsPy Trace-like object to add
    #     :type: other: obspy.core.trace.Trace-like or dict
    #     :param **options:key-word argument gatherer to pass to the 
    #                     FoldTrace.__add__() or FoldTraceBuffer.__add__() method
    #     :type **options: kwargs
    #     """
    #     # If potentially appending a wave
    #     if isinstance(other, dict):
    #         try:
    #             other = wave2foldtrace(other)
    #         except SyntaxError:
    #             pass
    #     # If appending a trace-type object
    #     elif isinstance(other, Trace):
    #         # If it isn't an FoldTrace, __init__ one from data & header
    #         if not isinstance(other, FoldTrace):
    #             other = FoldTrace(data=other.data, header=other.stats)
    #         else:
    #             pass
    #     # Otherwise
    #     else:
    #         raise TypeError(f'other {type(other)} not supported.')
        
    #     if isinstance(other, FoldTrace):
    #         # Get id of FoldTrace "other"
    #         key_opts = other.key_opts
    #         # If id is not in traces.keys() - use dict.update
    #         key = key_opts[self.key_attr]
    #         # If new key
    #         if key not in self.traces.keys():
    #             self.traces.update({key: other})
    #         # If key is in traces.keys() - use __add__
    #         else:
    #             self.traces[key].__add__(other, **options)
    #         self.stats.update_time_range(other)
    #         self.stats.common_id = self.get_common_id()

    # def _add_stream(self, stream, **options):
    #     """
    #     Supporting method to iterate across sets of trace-like objects in a stream-like object
    #     and apply the :meth:`~PULSE.data.dictstream.DictStream._add_trace` method.

    #     :param stream: iterable object containing obspy Trace-like objects
    #     :type stream: obspy.core.stream.Stream or other list-like of obspy.core.trace.Trace-like objects
    #     :param **options: optional key-word argument gatherer
    #                     to pass kwargs to the :meth:`~PULSE.data.dictstream.DictStream._add_trace` method
    #     :type **options: kwargs, optional
    #     """
    #     for _tr in stream:
    #         self._add_trace(_tr, **options)