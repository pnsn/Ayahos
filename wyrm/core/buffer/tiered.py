from wyrm.buffer.trace import TraceBuff
import wyrm.util.input_compatability_checks as icc
from obspy import Stream, Trace
import inspect
import fnmatch
from copy import deepcopy

class TieredBuffer(dict):
    """
    Standardized structure and search methods for an 2-tiered dictionary with 
    the structure
        tier0 - a primary key-set, generally an instrument code string
            tier1 - a secondary key-set, generally an instrument component code or model name
                buff - some data holding class object

    The rationale of this structure is to accelerate searching by limiting the number
    of keys that are searched in a given list of keys, and leveraging the query speed
    of hash-tables underlying the Python `dict` class. Earlier tests showed that this
    structured system was significantly faster at retrieving data from large sets of
    objects compared to an unstructured list-like holder.

    NOTE
    This version of TieredBuffer limits the structure to 10 tiers maximum to have
    kwarg values for most methods conform to some form of the string '*T?'.
    
    If you need more, your probably want to and know how to code your own
    structure and can base it on the code here.

    """
    def __init__(self, buff_class=TraceBuff, **buff_init_kwargs):
        # Inherit dict properties
        super().__init__()
        # Compatability check for buff_class input
        if not isinstance(buff_class, type):
            raise TypeError('buff_class must be a class-defining type')
        # Ensure that buff_class has an append method
        elif 'append' not in dir(buff_class):
            raise TypeError('buff_class does not have an "append" method - incompatable')
        else:
            self.buff_class = buff_class

        # Compatability check for buff_init_kwargs input
        bcargs = inspect.getfullargspec(self.buff_class).args[1:]
        bcdefs = inspect.getfullargspec(self.buff_class).defaults
        bdefaults = dict(zip(bcargs, bcdefs))
        emsg = ''
        for _k in buff_init_kwargs.keys():
            if _k not in bcargs:
                if emsg == '':
                    emsg += f'The following kwargs are not compabable with {buff_class}:'
                emsg += f'\n{_k}'
        # If emsg has any information populated in it, kick error with emsg report
        if emsg != '':
            raise TypeError(emsg)
        # Otherwise clear init_kwargs
        else:
            for _k in bcargs:
                if _k not in buff_init_kwargs and _k != 'self':
                    buff_init_kwargs.update({_k: bdefaults[_k]})
            self.bkwargs = buff_init_kwargs
        # Create a template buffer (some methods have larger overhead, so making a template saves time)
        self._template_buff = self.buff_class(**self.bkwargs)
        self.nbranches = 0
        self.nitems = 0

    def copy(self):
        """
        Return a deepcopy of this wyrm.buffer.structures.TieredBuffer
        """
        return deepcopy(self)    

    def add_branch(self, TK0, TK1=None):
        """
        Add a new branch to this tiered buffer with specified tier-0 and tier-1 keys

        :: INPUTS ::
        :param TK0: [str] tier-0 key for candidate branch
        :param TK1: [str] or None
                    tier-1 key for candidate branch
                    If None, an empty branch keyed to TK0 is
                    added if tier-0 key TK0 does not already exist
        
        :: OUTPUT ::
        :return self: [TieredBuffer] representation of self to enable cascading
        """
        if TK0 not in self.keys():
            if TK1 is not None:
                self.update({TK0: {TK1: self._template_buff.copy()}})
                self.nbranches += 1
                self.nitems += 1
            else:
                self.update({TK0: {}})
                self.nbranches += 1

        elif TK1 is not None:
            if TK1 not in self[TK0].keys():
                self[TK0].update({TK1: self.buff_class(**self.bkwargs)})
                self.nitems +=1
            else:
                print(f'Warning - buffer [{TK0}][{TK1}] already initialized')
        return self

    def apply_buffer_method(self, TK0='', TK1='', method='__repr__', **inputs):
        """
        Apply specified buff_class class method to entries that match (regex) strings
        for tier-0 and tier-1 keys as self[TK0][TK1].method(**inputs)

        NOTE: Operations are applied to data contained within each buff_class object as
                specified by the method and are expected to act on data in-place. 

                e.g., For buff_class = TraceBuff
                self.apply_buffer_method(method='filter', TK0='UW.*.EH', TK1='Z', type='bandpass', freqmin=1, freqmax=45)
                    would apply a 1-45 Hz Butterworth filter to analog, 1-component seismometers in the UW network


        :: INPUTS ::
        :param method: [str] string representation of a valid classmethod
                        for self.buff_class objects
                        default: "__repr__"
        :param TK0: [str] (regex-compatable) string to search for among tier-0 keys
        :param TK1: [str] (regex-compatable) string to search for among tier-1 keys
                            belonging to matched tier-0 key branches
        :param **inputs: [kwargs] inputs for `method`. Any arg-type inputs should be
                            presented as kwargs to ensure correct assignment
        
        :: OUTPUT ::
        :return self: [TieredBuffer] representation of self to enable cascading
        """
        # Check that method is compatable with buff_class
        if method not in dir(self.buff_class):
            raise SyntaxError(f'specified "method" {method} is not compatable with buffer type {self.buff_class}')
        # Conduct search on tier-0 keys
        for _k0 in fnmatch.filter(self.keys(), TK0):
            # Conduct search on tier-1 keys
            for _k1 in fnmatch.filter(self[_k0], TK1):
                # Compose eval str to enable inserting method
                eval_str = f'self[_k0][_k1].{method}(**inputs)'
                eval(eval_str); # <- run silent
        return self

    def append(self, obj, TK0, TK1, **options):
        """
        Append a single object to a single buffer object in this TieredBuffer using
        the buffer_object.append() method

        :: INPUTS ::
        :param obj: [object] object to append to a self.buff_class object
            Must be compatable with the self.buff_class.append() method
        :param TK0: [str] key string for tier-0 associated with `obj`
        :param TK1: [str] key string for tier-1 associated with `obj`

        :param **options: [kwargs] key-word arguments to pass to 
                    self.buff_class.append()
        
        :: OUTPUT ::
        :return self: [wyrm.buffer.structures.TieredBuffer] return
                    alias of self to enable cascading.
        """
        # Add new structural elements a needed
        self.add_branch(TK0, TK1=TK1)
        # Append obj to buffer [TK0][TK1]
        self[TK0][TK1].append(obj, **options)
        # Return self
        return self
    

    def __str__(self):
        """
        Provide a representative string documenting the initialization of
        this TieredBuffer
        """
        rstr = f'wyrm.buffer.structures.TieredBuffer(buff_class={self.buff_class}'
        for _k, _v in self.bkwargs.items():
            rstr += f', {_k}={_v}'
        rstr += ')'
        return rstr
    
    def _repr_line(self, _k0):
        """
        Compose a single line string that represents contnts of 
        branch [_k0]
        """
        rstr =  f'{_k0:16} '
        for _k1 in self[_k0].keys():
            rstr += f'| [{_k1}] {self[_k0][_k1].__str__(compact=True)} '
        return rstr 
    
    def __repr__(self, extended=False):
        """
        String representation of the contents of this TieredBuffer
        similar to the format of obspy.core.stream.Stream objects

        :: INPUT ::
        :param extended: [bool] Show all tier-0 entries and their
                        compact representation of tier-1 labels and
                        buffer contents?

        :: OUTPUT ::
        :return rstr: [str] representative string
        """
        _cl = 3
        # Create header line
        rstr = f'TieredBuffer comprising {self.nbranches} branches holding {self.nitems} {self.buff_class} buffers\n'
        # If extended, iterate across all tier-0 keys
        if extended:
            for _k0 in self.keys():
                # And print tier-1 keys plus compact buffer representations
                rstr += self._repr_line(_k0)
                rstr += '\n'
        # If not extended produce a similar trucated format as 
        # obspy.core.stream.Stream.__str__(extended=False)
        else:
            # Iterate across tier-0 keys
            for _i, _k0 in enumerate(self.keys()):
                # If the first 2 or last 2 lines
                if _i < _cl or _i > len(self.keys()) - _cl - 1:
                    rstr += self._repr_line(_k0)
                    rstr += '\n'
                elif _i in [_cl, len(self.keys()) - _cl - 1]:
                    rstr == '    ...    \n'
                elif _i == _cl + 1:
                    rstr += f' ({self.nbranches - _cl*2} other branches)\n'
            if not extended and len(self) > _cl*2:
                rstr += 'To see full contents, use print(tieredbuffer.__repr__(extended=True))'
        return rstr

    def append_stream(self, stream):
        """
        Convenience method for appending the contents of obspy.core.stream.Stream to
        a TieredBuffer with buff_class=wyrm.buffer.trace.TraceBuff, using the 
        following key designations

        TK0 = trace.id[:-1]
        TK1 = trace.id[-1]

        :: INPUT ::
        :param stream: [obspy.core.stream.Stream] or [obspy.core.trace.Trace]
                Trace or Stream to append to this TieredBuffer
        :: OUTPUT ::
        :return self: [wyrm.buffer.structures.TieredBuffer] allows cascading
        """
        # Ensure the buff_class is TraceBuff
        if self.buff_class != TraceBuff:
            raise AttributeError('Can only use this method when buff_class is wyrm.buffer.trace.TraceBuff')
        # Ensure the input is an obspy Trace or Stream
        if not isinstance(stream, (Trace, Stream)):
            raise TypeError
        # If this is a trace, turn it into a 1-element list
        if isinstance(stream, Trace):
            stream = [stream]
        # Iterate across traces in stream
        for _tr in stream:
            # Generate keys
            TK0 = _tr.id[:-1]
            TK1 = _tr.id[-1]
            # Append to TraceBuff, generating branches as needed
            self.append(_tr, TK0=TK0, TK1=TK1)
        return self

    def append_predarray(self, predarray):
        """
        Convenience method for appending a wyrm.buffer.prediction.PredArray to
        a TieredBuffer with buff_class=wyrm.buffer.prediction.PredBuff using
        the following key designations

        TK0 = predarray.id
        TK1 = predarray.weight_name
        """
        if self.buff_class != PredBuff:
            raise AttributeError('Can only use this method when buff_class is wyrm.buffer.prediction.PredBuff')
        if not isinstance(predarray, PredArray):
            raise TypeError
        TK