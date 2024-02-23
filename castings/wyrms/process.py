
"""
:module: wyrm.wyrms.process
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains the class definition for ProcWyrm objects.
    ProcWyrm's (Processing Wyrms) pulse applys a standard sequence of
    classmethods to a target object type that take the form
    
    obj.classmethod1(args1, kwargs1) 
    obj.classmethod2(args2, kwargs2)
    
    wherein obj's data are altered by applying 'classmethod', and 
    concludes this unit processing pipeline by applying an output
    classmethod of the form

    out = obj.classmethod(args, kwargs)
    
    except in the special case (out_eval_string = None) where

    out = obj

    Objects are pop'd from an input deque and outputs are appendleft'd
    to a deque attached to the ProcWyrm object (ProcWyrm.queue).

    These class methods are applied to objects using the eval notation:
    eval(f'{obj}{classmethod_string}')
    and
    out = eval(f'{obj}{output_classmethod_string}')

    
TODO: Clean up hanging try - except in def pulse():
"""
from wyrm import Wyrm
from collections import deque
from wyrm.structures.window import MLInstWindow
from wyrm.util.stacking import semblance
from wyrm.wyrms.window import WindowWyrm
import wyrm.util.input_compatability_checks as icc
import inspect


class EvalWyrm(Wyrm):
    """
    Wyrm that applies a series of class methods to an input object
    and returns that altered object. Provides 
    """
    def __init__(self,
                 target_class=InstWindow,
                 class_method_strs=['._preproc_example'',
                 max_pulse_size=1,
                 debug=False)


class ProcWyrm(Wyrm):
    """
    This worm provides a pulse method that applys a series of
    class-methods that operate in-place on objects' data. Expects
    objects to arrive in a deque
    """

    def __init__(
        self,
        target_class=MLInstWindow,
        self_eval_string_list=["._preproc_example(fill_value=0)"],
        out_eval_string='.to_torch()',
        max_pulse_size=10000,
        debug=False
    ):
        """
        Initialize a ProcWyrm object with specified processing targets and steps

        :: INPUTS ::
        :param target_class: [type] expected type for objects being processed
                                    by this ProcWyrm.
                                    E.g., obspy.Trace
        :param self_eval_string_list: [list-like] set of strings taking the form
                                    ".classmethod(args, kwargs)" to be applied
                                    to expected objects.
                                    e.g., '.resample(100, no_filter=True)'
        :param out_eval_string: [str] string representation of a target_class
                                    method that produces an output
                                    e.g., '.copy()'
                                [None] if None, the output is the altered
                                    object itself rather than the output
                                    of a classmethod enacted on the object
        :param max_pulse_size: [int-like] maximum number of objects to attempt
                                    to evaluate in a given pulse
        :param debug: [bool] should this Wyrm be run in debug mode?
        """

        # Run super__init__ from Wyrm
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)
        # Compatability check for target_class
        try:
            type(target_class)
            self.target_class = target_class
        except NameError:
            raise NameError(f"{target_class} is not defined")

        # Compatability checks for self_eval_string_list
        if isinstance(self_eval_string_list, str):
            self_eval_string_list = [self_eval_string_list]
        elif isinstance(self_eval_string_list, list):
            if all(isinstance(_es, str) for _es in self_eval_string_list):
                self_eval_string_list = self_eval_string_list
            else:
                raise TypeError("not all etries in eval_strigns are type str")
        else:
            raise TypeError("self_eval_string_list must be type str or list of str")

        # Syntax checks for all self_eval_string_list elements
        all_bool_list = [
            self._check_class_method_string_formatting(_e) for _e in self_eval_string_list
        ]
        if all(all_bool_list):
            self.self_eval_string_list = self_eval_string_list
        # Otherwise
        else:
            # Iterate across checks
            for _i, _e in enumerate(self_eval_string_list):
                # Highlight specific bad eval_string entries
                if not all_bool_list[_i]:
                    print(
                        f'self_eval_string_list entry "{_e}" does not pass formatting checks'
                    )
            # Then raise SyntaxError with advice on formatting
            raise SyntaxError(
                'self_eval_string_list entries must have basic form ".classmethod()"'
            )

        # Compatabiliy checks for out_eval_string
        # Type check
        if out_eval_string is None:
            self.out_eval_string = out_eval_string
        elif not isinstance(out_eval_string, str):
            raise TypeError("out_eval_string must be type str")
        # Syntax check
        if self._check_class_method_string_formatting(out_eval_string):
            self.out_eval_string = out_eval_string
        else:
            raise SyntaxError(
                '"out_eval_string" did not pass format checks - see ProcWyrm docs.'
            )

        # Initialize output double ended queue attribute
        self.queue = deque([])

    ## Compatability checking methods ##

    def _check_class_method_string_formatting(self, string):
        """
        Check if the proposed class-method string for eval()
        execution has the fundamentally correct formatting
        expected i.e.,: ".classmethod()" and that the proposed
        "classmethod" is an attribute or classmethod of
        self.target_class

        Tests:
        1) starts with '.'?
        2) contains at least one '('
        3) ends with ')'
        4) equal number of '(' and ')'
        5) 'classmethod' is in dir(self.target_class)

        :: INPUT ::
        :param string: [str] .classmethod() formatted string
                    e.g.,
                    for a numpy array 'arr' with shape (12,)
                    string = '.reshape(3,4)'
                    would pass this test (i.e., return True), HOWEVER,
                    string = '.reshape(4,4)' would also pass this test,
                    but fail at execution due to the arguments provided.
        :: OUTPUT ::
        :return status: [bool] does this string pass all tests?
        """
        # Run formtting checks on out_eval_string
        bool_set = []
        # Check that starts with '.'
        bool_set.append(string[0] == ".")
        # Check that splitting on "(" gives at least 2 parts
        bool_set.append(len(string.split("(")) >= 2)
        # Check that string ends with ')'
        bool_set.append(string[-1] == ")")
        # Check that splitting on ")" gives the same amount of elements as "(" split
        bool_set.append(len(string.split("(")) == len(string.split(")")))
        # Check that proposed method is in target class's class methods
        bool_set.append(string.split("(")[0][1:] in dir(self.target_class))
        status = all(bool_set)
        return status

    def _check_obj_compat(self, obj):
        """
        Check if input object is self.target_class
        :: INPUT ::
        :param obj: [object] candidate object

        :: OUTPUT ::
        :return [bool]: is this an instance of self.target_class? 
        """
        if isinstance(obj, self.target_class):
            return True
        else:
            return False

    def _run_self_eval_statements(self, obj):
        """
        Run self_eval_string_list elements in sequence on target object
        :: INPUT ::
        :param obj: [object] candidate object

        :: OUTPUT ::
        :return obj: [object] candidate object with self-altering
                              class methods applied
        """
        for _e in self.self_eval_string_list:
            eval(f"obj{_e}")
        return obj

    def _run_out_eval_statement(self, obj):
        """
        Run out_eval_string command on target object
        :: INPUT ::
        :param obj: [object] candidate object
        
        :: OUTPUT ::
        :return out: [std_out] standard output of eval('obj.out_eval_string')
        """
        if self.out_eval_string is None:
            out = obj
        else:
            out = eval(f"{obj}{self.out_eval_string}")
        return out

    def pulse(self, x):
        """
        For an input deque of objects, sequentially pop off objects,
        assess if they are type self.target_class, apply sequential
        self_eval_string_list to those matching objects, and finally apply
        out_eval_string to the final altered object, collecting the
        output in self.queue with .appendleft(). 

        Objects popped off input deque 'x' that do not match self.target_class
        are re-appended to the input deque using .appendleft().

        Pulse will run until hitting max_pulse_size iterations, with the following
        early termination criteria:
        1) the input deque 'x' is empty
        2) number of iterations exceeds the original length of the input deque 'x'

        :: INPUT ::
        :param x: [deque] ideally consisting of only self.target_class objects

        :: OUTPUT ::
        :return y: [deque] alias to self.queue
        """
        nproc = 0
        if not isinstance(x, deque):
            raise TypeError(f'input "{x}" is not a deque')
        elif len(x) == 0:
            print("Empty input deque. Ending iterations.")
            y = self.queue
        else:
            x_queue_length = len(x)
        # If nonempty deque present, iterate
        for _i in range(self.max_pulse_size):
            # Handle iteration break if deque is now empty
            if len(x) == 0:
                if self.debug:
                    print(f"Input deque emptied after {_i} iterations.")
                    print("Ending iterations.")
                break
            elif _i > x_queue_length:
                if self.debug:
                    omsg = 'All input deque entries have been assessed.'
                    omsg += f'\nx_queue status: {len(x)}:{x_queue_length}'
                    print(omsg)
                break
            # Otherwise keep plucking
            else:
                # Pop off object from head of deque
                obj = x.pop()
                # If object is non-compatable, replace at end of deque
                if not self._check_obj_compat(obj):
                    x.appendleft(obj)
                # Otherwise, process
                else:
                    # Run self_eval statements on object with catch in case of Error
                    try:
                        self._run_self_eval_statements(obj)
                    except:
                        breakpoint() # Placeholer for dev
                        # raise ValueError.__traceback__ # Is this syntax right? - want to 

                    # Run output_eval statement
                    _yi = self._run_out_eval_statement(obj)
                    # Append output to this Wyrm's queue
                    self.queue.appendleft(_yi)
                    nproc += 1
        y = self.queue
        return y
                

class CRFWyrm(Wyrm):
    """
    Characteristic Response Function Wyrm
    A wyrm for applying established, characteristic response function analyses
    on timeseries data that provides outputs in comparable formats to those
    produced by Machine Learning workflows (i.e., wyrm.wyrms.predict.MachineWyrm)
    to allow ensemble detection/classification that include both "classic" and "ML"
    data products
    """

    def __init__(self, method, max_pulse_size=10000, debug=False, **options):
        # Inherit Wyrm
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)



        self.method = method
        self.mkws = options
        

    def pulse(self, x):
        """
        Iterate across data contained in iterable/poppable structure x
        and apply the 
        """


class SembWyrm(WindowWyrm):
    
    def __init__(self, inst_fn_str='*', model_fn_str='*', wgt_fn_str='*', label_dict={'P': ['P','p'], 'S': ['S', 's'], 'D': ['Detection','STALTA']}, samp_rate=100, **semblance_kwargs):
        """
        Pull windows from a TieredBuffTree at the end of a CanWyrm object
        and conduct semblance
        """
        # Inherit from WindowWyrm
        super().__init__()


        # Compat check for instrument name fnmatch.filter string
        if not isinstance(inst_fn_str, str):
            raise TypeError('inst_fn_str must be type str')
        else:
            self.ifs = inst_fn_str
        # Compat check for ML model name fnmatch.filter string
        if not isinstance(model_fn_str, str):
            raise TypeError('model_fn_str must be type str')
        else:
            self.mfs = model_fn_str
        # Compat check for pretrained model weight name fnmatch.filter string
        if not isinstance(wgt_fn_str, str):
            raise TypeError('wgt_fn_str must be type str')
        else:
            self.wfs = wgt_fn_str
        # Compat checks for sampling rate
        self.samp_rate = icc.bounded_floatlike(
            samp_rate,
            name='samp_rate',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # Compatability check for semblance kwarg gatherer
        sargs = inspect.getfullargspec(semblance).args
        emsg = ''
        for _k in semblance_kwargs.keys():
            if _k not in sargs:
                if emsg == '':
                    emsg += 'The following kwargs are not compabable with wyrm.util.stacking.semblance:'
                emsg += f'\n{_k}'
        if emsg == '':
            raise TypeError(emsg)
        else:
            self.skwargs = semblance_kwargs
        
        self.queue = deque([])

    def _gather