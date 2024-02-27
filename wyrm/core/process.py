"""
:module: wyrm.core.window
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module houses class definitions related to generating and
    handling windowed data (i.e. defined temporal and/or sample length)

    Classes:
        
        WindowWyrm - a submodule for sampling a BufferTree that buds with
            TraceBuffer objects for a specific window size and advancement
            increment to generate InstrumentWindow objects

            BufferTree(TraceBuffer) -> Windowyrm -> deque(InstrumentWindow)

        ProcWyrm - a submodule for applying class methods to objects that
            alter those objects' data in-place via eval() execution of
            string-representations of class methods and their arguments.

            Each class method is executed independently, using the outpu

            deque(class_obj) -> ProcWyrm -> deque(class_obj)

        WavePredictWyrm - a submodule for conducting predictions with machine
            learning model architectures based on the seisbench.models.WaveformModel
            base class, with the option to predict using a list of pretrained weights.
            Based on methods in Yuan et al. (2023).

            deque()

        SembWyrm - a submodule for conducting semblance-based combinations
            of time-series data transforms/predictions to derive concensus
            estimates. Based on methods in Yuan et al. (2023).


"""
import numpy as np
import torch
import seisbench.models as sbm
import wyrm.util.compatability as wuc
import wyrm.core.data as wcd
from obspy import UTCDateTime
from wyrm.core._base import Wyrm


###################################################################################
# WINDOW WYRM CLASS DEFINITION ####################################################
###################################################################################

class WindowWyrm(Wyrm):
    """
    The WindowWyrm class takes windowing information from an input
    seisbench.models.WaveformModel object and user-defined component
    mapping and data completeness metrics and provides a pulse method
    that iterates across entries in an input RtInstStream object and
    generates windowed copies of data therein that pass data completeness
    requirements. Windowed data are formatted as MLInstWindow objects and
    staged in the WindWyrm.queue attribute (a deque) that can be accessed
    by subsequent data pre-processing and ML prediction Wyrms.
    """

    def __init__(
        self,
        code_map={"Z": "Z3", "N": "N1", "E": "E2"},
        completeness={"Z": 0.95, "N": 0.8, "E": 0.8},
        missing_component_rule="zeros",
        model_name="EQTransformer",
        target_samprate=100.0,
        target_npts=6000,
        target_overlap=1800,
        target_blinding=500,
        target_order="ZNE",
        max_pulse_size=20,
        debug=False,
    ):
        """
        Initialize a WindowWyrm object

        :: INPUTS ::
        -- Window Generation --
        :param code_map: [dict] dictionary with keys "Z", "N", "E" that have
                        iterable sets of character(s) for the SEED component
                        codes that should be aliased to respectiv keys
        :param trace_comp_fract :[dict] dictionary of fractional thresholds
                        that must be met to consider a given component sufficiently
                        complete to trigger generating a new InstrumentWindow
        :param missing_component_rule: [str]
                        string name for the rule to apply in the event that
                        horizontal data are missing or too gappy
                            Supported: 'zeros', 'clonez','clonehz'
                        also see wyrm.core.window.instrument.InstrumentWindow
        :param model_name: [str]
                        Model name to associate this windowing parameter set to
                        e.g., EQTransformer, PhaseNet
        :param target_samprate: [float]
                        Target sampling rate to pass to InsturmentWindow init
        :param target_npts: [int]
                        Target temporal samples to pass to InstrumentWindow init
        :param target_overlap: [int]
                        Target overlap between sequential windows. Used alongside
                        target_samprate and target_npts to determine window advances
                        between pulse iterations
        :param target_blinding:  [int] left and right blinding sample amounts
                        for stacking ML predictions. This is also used to determine
                        trace completeness. 
        :param target_order: [str]
                        Target component order to pass to InstrumentWindow init
        :param max_pulse_size: [int]
                        Maximum number of sweeps this Wyrm should conduct over
                        pulse input x (TieredBuffer) buffers to check if each
                        buffer can produce a new window. Generally this will be
                        a low number [1, 20].
        :param debug: [bool] - run in debug mode?
        """
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)

        if not isinstance(code_map, dict):
            raise TypeError('code_map must be type dict')
        elif not all(_c in 'ZNE' for _c in code_map.keys()):
            raise KeyError('code_map keys must comprise "Z", "N", "E"')
        elif not all(isinstance(_v, str) and _k in _v for _k, _v in code_map.items()):
            raise SyntaxError('code_map values must be type str and include the key value')
        else:
            self.code_map = code_map

        if not isinstance(completeness, dict):
            raise TypeError('completeness must be type dict')
        elif not all(_c in 'ZNE' for _c in completeness.keys()):
            raise KeyError('completeness keys must comprise "Z", "N", "E"')
        elif not all (0 <= _v <= 1 for _v in completeness.values()):
            raise ValueError('completeness values must fall in the range [0, 1]')
        else:
            self.completeness = completeness
        
        if not isinstance(missing_component_rule, str):
            raise TypeError('missing_component_rule must be type str')
        elif missing_component_rule.lower() not in ['zeros','clonez','clonehz']:
            raise ValueError(f'missing_component_rule {missing_component_rule} not supported')
        else:
            self.mcr = missing_component_rule.lower()

        if not isinstance(model_name, str):
            raise TypeError('model_name must be type str')
        else:
            self.model_name = model_name
        # Target sampling rate compat check
        self.target_samprate = wuc.bounded_floatlike(
            target_samprate,
            name='target_samprate',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # Target window number of samples compat check
        self.target_npts = wuc.bounded_intlike(
            target_npts,
            name='target_npts',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # Target window number of sample overlap compat check
        self.target_overlap = wuc.bounded_intlike(
            target_overlap,
            name='target_overlap',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # Target (symmetrical) blinding number of samples compat check
        self.target_blinding = wuc.bounded_intlike(
            target_blinding,
            name='target_blinding',
            minimum=0,
            maximum=self.target_npts/2,
            inclusive=True
        )
        # Target component order compat check
        if not isinstance(target_order, str):
            raise TypeError('target_order must be type str')
        elif not all(_c.upper() in 'ZNE' for _c in target_order):
            raise ValueError('target_order must comprise "Z", "N", "E"')
        else:
            self.target_order = target_order.upper()

         # Set Defaults and Derived Attributes
        # Calculate window length, window advance, and blinding size in seconds
        self.window_sec = self.target_npts/self.target_samprate
        self.advance_sec = (self.target_npts - self.target_overlap)/self.target_samprate
        self.blinding_sec = self.target_blinding/self.target_samprate

        # Create dict for holding instrument window starttime values
        self.default_starttime = None
        self.window_tracker = {}

        # Create queue for output collection of windows
        self.queue = wcd.deque([])

        # Gather general instrumentwindow creation instructions 
        self.windowing_attr = {
            'model_name': self.model_name,
            'component_order': self.target_order,
            'target_npts': self.target_npts,
            'target_samprate': self.target_samprate,
            'target_blinding': self.target_blinding,
            'missing_component_rule': self.mcr}

    ################
    # PULSE METHOD #
    ################
        
    def pulse(self, x, **options):
        """
        Conduct up to the specified number of iterations of
        self._process_windows on an input RtInstStream object
        and return access to this WindWyrm's queue attribute

        Includes an early termination trigger if an iteration
        does not generate new windows.

        :: INPUT ::
        :param x: [wyrm.core.buffer.structure.TieredBuffer] terminating
                    with [wyrm.core.buffer.trace.TraceBuffer] objects
        :param options: [kwargs] optional kwargs to pass to self.process_windows()

        :: OUTPUT ::
        :return y: [deque] deque of 
                    [wyrm.core.window.instrument.InstrumentWindow' objects
                    loaded with the appendleft() method, so the oldest messages
                    can be removed with the pop() method in a subsequent step
        """
        # Iterate for up to self.max_pulse_size sweeps across x
        for _ in range(self.max_pulse_size):
            nnew = self.process_windows(x, **options)
            # Early stopping trigger - if process_windows produces no new windows
            if nnew == 0:
                break
        # Return y as access to WindWyrm.queue attribute
        y = self.queue
        return y

    # ################ #
    # CORE SUBROUTINES #
    # ################ #

    def process_windows(self, input, pad=True, extra_sec=0., wgt_taper_len='blinding', wgt_taper_type='cosine'):
        """
        Primary coordinating processes for WindowWyrm - conducts cross-checks with window_tracker to initialize
        and update window start times for each branch of a TieredBuffer terminating in 
        TraceBuffer objects and calls the subroutine "branch2iwind()", which does the instrument
        window validation and generation.

        :: INPUTS ::
        :param input: [wyrm.core.data.BufferTree] terminating in [wyrm.core.data.TraceBuffer] objects
        """
        # pad compat check
        if not isinstance(pad, bool):
            raise TypeError('pad must be type bool')
        
        # wgt_taper_len compat check
        if isinstance(wgt_taper_len, str):
            if wgt_taper_len.lower() != 'blinding':
                raise TypeError('The only supported str-type wgt_taper_len input is "blinding"')
        elif not isinstance(wgt_taper_len, (int,float)):
            raise TypeError('wgt_taper_len must be type int, float, or str "blinding"')
        if not isinstance(wgt_taper_type, str):
            raise TypeError('wgt_taper_type must be type str')
        
        # Start number of new windows counter
        nnew = 0
        # Iterate across limbs
        for k0 in input.keys():
            if k0 not in self.window_tracker.keys():
                self.window_tracker.update({k0: {}})
            for k1 in input[k0].keys():
                branch = input[k0][k1]
                if k1 not in self.window_tracker[k0].keys():
                    self.window_tracker[k0].update({k1: self.default_starttime})
                
                # Get next starttime
                next_ts = self.window_tracker[k0][k1]
                # If next starttime is the default value, scrape starttime from branch's Z-component
                if next_ts == self.default_starttime:
                    for _c in self.code_map['Z']:
                        if _c in branch.keys():
                            if len(branch[_c]) > 0:
                                first_ts = branch[_c].stats.starttime
                                self.window_tracker[k0].update({k1: first_ts})
                                next_ts = first_ts
                                break
                # If a valid next starttime is present (or was scraped), proceed
                if isinstance(next_ts, UTCDateTime):
                    data_ts = None
                    data_te = None
                    # Iterate across entries in branch
                    for k2 in branch.keys():
                        # If a branch entry has data
                        if len(branch[k2]) > 0:
                            # Grab the starttime, endtime and buffer length
                            data_ts = branch[k2].stats.starttime
                            data_te = branch[k2].stats.endtime
                            buff_length = branch[k2].max_length
                            break
                    # if data encompasses the next starttime, proceed
                    if data_ts <= next_ts < data_te:
                        pass
                    # if data ends before the next starttime - run safety catch if there's run-away behavior
                    # in window_tracker
                    elif next_ts > data_te:
                        dt = next_ts - data_te
                        if dt < 2*self.advance_sec:
                            pass
                        else:
                            RuntimeError('Suspect a run-away next_starttime incremetation')
                    # If next starttime is before the data starttime, check how large the gap is
                    elif next_ts < data_ts:
                        dt = data_ts - next_ts
                        # If the gap is shorter than the buffer length, proceed assuming that
                        # there may be a backfill
                        if dt < buff_length:
                            pass
                        # If the gap is longer than the buffer length, assume that there was a
                        # data outage and advance the next starttime enough to catch up
                        else:
                            nadv = dt//self.advance_sec
                            self.window_tracker[k0][k1] += nadv*self.advance_sec
                            next_ts = self.window_tracker[k0][k1]
                    # Run window validation/generation sub-routine
                    iwind = self.branch2iwind(
                        branch,
                        next_ts,
                        pad=pad,
                        extra_sec=extra_sec,
                        wgt_taper_len=wgt_taper_len,
                        wgt_taper_type=wgt_taper_type
                        )
                    # If this produces a window, append to queu and increment nnew, and continue iterating
                    if iwind:
                        self.queue.append(iwind)
                        # Increment number of new windows produced
                        nnew += 1
                        # Increment window index in window_tracker
                        self.window_tracker[k0][k1] += self.advance_sec
                    # If a window was not produced, continue iterating across branches
                    else:
                        continue
        return nnew

    def branch2iwind(
        self,
        branch,
        next_window_starttime,
        pad=True,
        extra_sec=1.0,
        wgt_taper_len=0.0,
        wgt_taper_type="cosine",
    ):
        """
        Using a specified candidate window starttime and windowing information
        attributes in this WindWyrm, determine if and input branch from a
        RtInstStream object has enough data to generate a viable window

        :param branch: [dict] of [wyrm.structures.rtbufftrace.RtBuffTrace] objects
                            with keys corresponding to RtBuffTrace datas' component
                            code (e.g., for BHN -> branch = {'N': RtBuffTrace()})
        :param next_window_starttime: [UTCDateTime] start time of the candidate window
        :param pad: [bool] should data windowed from RtBuffTrace objects be padded
                           (i.e., allow for generating masked data?)
                           see obspy.core.trace.Trace.trim() for more information
        :param extra_sec: [None] or [float] extra padding to place around windowed
                            data. Must be a positive value or None. None results in
                            extra_sec = 0.
                            NOTE: Extra samples encompassed by the extra_sec padding
                            on each end of windows are only included after a candidate
                            window has been deemed to have sufficient data. They do not
                            factor into the determination of if a candidate window
                            is valid
        :param wgt_taper_len: [str] or [float-like] amount of seconds on each end
                            of a candidate window to downweight using a specified
                            taper function when assessing the fraction of the window
                            that contains valid data.
                            Supported string arguments:
                                'blinding' - uses the blinding defined by the ML model
                                associated with this WindWyrm to set the taper length
                            float-like inputs must be g.e. 0 and finite
        :param wgt_taper_type: [str] name of taper to apply to data weighting mask when
                                determining the fraction of data that are valid in a
                                candidate window
                            Supported string arguments:
                                'cosine':   apply a cosine taper of length
                                            wgt_taper_len to each end of a
                                            data weighting mask
                                    aliases: 'cos', 'tukey'
                                'step':     set weights of samples in wgt_taper_len of each
                                            end of a candidate window to 0, otherwise weights
                                            are 1 for unmasked values and 0 for masked values
                                    aliases: 'h', 'heaviside'
        :: OUTPUT ::
        :return iwind: [wyrm.core.window.InstrumentWindow] or [None]
                        If a candidate window is valid, this method returns a populated
                        InstWindow object, otherwise, it returns None

        """
        # branch basic compatability check
        if not isinstance(branch, dict):
            raise TypeError("branch must be type dict")
        # next_window_starttime basic compatability check
        if not isinstance(next_window_starttime, UTCDateTime):
            raise TypeError("next_window_starttimest be type obspy.UTCDateTime")
        # pad basic compatability check
        if not isinstance(pad, bool):
            raise TypeError("pad must be type bool")
        # extra_sec compatability checks
        if extra_sec is None:
            extra_sec = 0
        else:
            extra_sec = wuc.bounded_floatlike(
                extra_sec, name="extra_sec", minimum=0.0, maximum=self.window_sec
            )
        # wgt_taper_len compatability checks
        if isinstance(wgt_taper_len, str):
            if wgt_taper_len.lower() == "blinding":
                wgt_taper_len = self.blinding_sec
            else:
                raise SyntaxError(f'str input for wgt_taper_len {wgt_taper_len} not supported. Supported: "blinding"')
        else:
            wgt_taper_len = wuc.bounded_floatlike(
                wgt_taper_len,
                name="wgt_taper_len",
                minimum=0.0,
                maximum=self.window_sec,
            )
        # wgt_taper_type compatability checks
        if not isinstance(wgt_taper_type, str):
            raise TypeError("wgt_taper_type must be type str")
        elif wgt_taper_type.lower() in ["cosine", "cos", "step", "heaviside", "h"]:
            wgt_taper_type = wgt_taper_type.lower()
        else:
            raise ValueError(
                'wgt_taper_type supported values: "cos", "cosine", "step", "heaviside", "h"'
            )

        # Start of processing section #
        # Create a copy of the windowing attributes to pass to InstWindow()
        window_inputs = self.windowing_attr.copy()
        # Add target_starttime
        window_inputs.update({"target_starttime": next_window_starttime})

        # Calculate the expected end of the window
        next_window_endtime = next_window_starttime + self.window_sec
        # Compose kwarg dictionary for TraceBuffer.get_trimmed_valid_fract()
        vfkwargs = {
            "starttime": next_window_starttime,
            "endtime": next_window_endtime,
            "wgt_taper_len": wgt_taper_len,
            "wgt_taper_type": wgt_taper_type,
        }
        # Iterate across component codes in branch
        for _k1 in branch.keys():
            # If _k1 is a Z component code
            if _k1 in self.code_map["Z"]:
                # Pull RtBuffTrace
                zbuff = branch[_k1]
                # Get windowed valid fraction
                valid_fract = zbuff.get_trimmed_valid_fraction(**vfkwargs)
                # Check valid_fraction
                if valid_fract >= self.completeness["Z"]:
                    # If sufficient data, trim a copy of the vertical data buffer
                    _tr = zbuff.to_trace()
                    _tr.trim(
                        starttime=next_window_starttime - extra_sec,
                        endtime=next_window_endtime + extra_sec,
                        pad=pad,
                        fill_value=None,
                    )
                    # Append to input holder
                    window_inputs.update({"Ztr": _tr})

            elif _k1 in self.code_map["N"]:
                hbuff = branch[_k1]
                valid_fract = hbuff.get_trimmed_valid_fraction(**vfkwargs)
                if valid_fract >= self.completeness["N"]:
                    # Convert a copy of the horizontal data buffer to trace
                    _tr = hbuff.to_trace()
                    # Trim data with option for extra_sec
                    _tr.trim(
                        starttime=next_window_starttime - extra_sec,
                        endtime=next_window_endtime + extra_sec,
                        pad=pad,
                        fill_value=None,
                    )
                    window_inputs.update({"Ntr": _tr})

            elif _k1 in self.code_map["E"]:
                hbuff = branch[_k1]
                valid_fract = hbuff.get_trimmed_valid_fraction(**vfkwargs)
                if valid_fract >= self.completeness["E"]:
                    # Convert a copy of the horizontal data buffer to trace
                    _tr = hbuff.to_trace()
                    # Trim data with option for extra_sec
                    _tr.trim(
                        starttime=next_window_starttime - extra_sec,
                        endtime=next_window_endtime + extra_sec,
                        pad=pad,
                        fill_value=None,
                    )
                    window_inputs.update({"Etr": _tr})

        if "Ztr" in window_inputs.keys():
            iwind = wcd.InstrumentWindow(**window_inputs)
            
        else:
            iwind = None
        return iwind

    # ############### #
    # DISPLAY METHODS #
    # ############### #

    def __repr__(self):
        rstr = super().__str__()
        rstr += f" | Windows Queued: {len(self.queue)}\n"
        for _c in self.target_order:
            rstr += f"  {_c}: map:{self.code_map[_c]} thresh: {self.completeness[_c]}\n"
        rstr += "ð -- Windowing Parameters -- ð"
        for _k, _v in self.windowing_attr.items():
            if isinstance(_v, float):
                rstr += f"\n  {_k}: {_v:.3f}"
            else:
                rstr += f"\n  {_k}: {_v}"
        return rstr

    def __str__(self):
        rstr = 'wyrm.core.window.wyrm.WindowWyrm('
        rstr += f'code_map={self.code_map}, completeness={self.completeness}, '
        rstr += f'missing_component_rule={self.mcr}, model_name={self.model_name}, '
        rstr += f'target_samprate={self.target_samprate}, target_npts={self.target_npts}, '
        rstr += f'target_overlap={self.target_overlap}, target_blinding={self.target_blinding}, ]'
        rstr += f'max_pulse_size={self.max_pulse_size}, debug={self.debug})'
        return rstr


    # ################################ #
    # SeisBench MODEL INGESTION METHOD #
    # ################################ #
    def update_params_from_seisbench_model(self, model):
        """
        Update windowing parameters for an initialized WindowWyrm using the 
        attributies carried by child-classes of the seisbench.models.WaveformModel
        base class (e.g., seisbench.models.EQTransformer).

        :: INPUT ::
        :param model: [seisbench.models.WaveformModel] ML model architecture object
        
        :: OUTPUT ::
        :return self: [wyrm.core.window.wyrm.WindowWyrm] enable cascading
        """

        if not isinstance(model, sbm.WaveformModel):
            raise TypeError('model must be type seisbench.models.WaveformModel')
        # Do checks on attributes that aren't populated in sbm.WaveformModel
        if isinstance(model.in_samples, int):
            self.in_samples = model.in_samples
        else:
            raise AttributeError('model does not have an "in_samples" attribute populated. Likely trying to pass a WaveformModel object, rather than a child-class object')
        
        if isinstance(model.sampling_rate, (int, float)):
            self.target_samprate = model.sampling_rate
        else:
            raise AttributeError('model does not have an "sampling_rate" attribute populated. Likely trying to pass a WaveformModel object, rather than a child-class object')
        
        # Scrape information re. model name and channel order
        self.model_name = model.name
        self.target_order = model.component_order

        # Scrape information from annotate_args
        aa = model._annotate_args
        self.target_overlap = aa['overlap'][1]
        blinding = aa['blinding'][1]
        # Handle difference of 2-tuple for blinding in SeisBench vs a single int in Wyrm
        if np.mean(blinding) == blinding[0] == blinding[1]:
            self.target_blinding = blinding[0]
        else:
            raise ValueError('Non-identical blinding values not supported.')
        
        return self

    
###################################################################################
# PROCWYRM CLASS DEFINITION #######################################################
###################################################################################

class ProcWyrm(Wyrm):
    
    def __init__(
        self,
        class_type,
        class_method_list=['.__repr__()'],
        max_pulse_size=10000,
        debug=False):

        super().__init__(max_pulse_size=max_pulse_size, debug=debug)

        try:
            type(class_type)
            self.class_type = class_type
        except NameError:
            raise NameError(f'{class_type} is not a defined class')

        if isinstance(class_method_list, str):
            class_method_list = [class_method_list]
        elif isinstance(class_method_list, (tuple, list)):
            if not all(isinstance(_e, str) for _e in class_method_list):
                raise TypeError('not all etries in class_method_list are type str')
        else:
            raise TypeError('class_method_list must be type str or a list-like of str')
    
        # Syntax checks for all class_method_list elements
        all_bool_list = [
            self._check_class_method_str(_e) for _e in class_method_list
        ]
        if all(all_bool_list):
            self.class_method_list = class_method_list
        # Otherwise
        else:
            # Iterate across checks
            for _i, _e in enumerate(class_method_list):
                # Highlight specific bad eval_string entries
                if not all_bool_list[_i]:
                    print(
                        f'class_method_list entry "{_e}" does not pass formatting checks'
                    )
            # Then raise SyntaxError with advice on formatting
            raise SyntaxError(
                'class_method_list entries must have basic form ".classmethod()"'
            )      
        self.queue = wcd.deque([])

    def _check_class_method_str(self, string):
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
        bool_set.append(string.split("(")[0][1:] in dir(self.class_type))
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
        if isinstance(obj, self.class_type):
            return True
        else:
            return False

    def _run_class_method_list(self, obj):
        """
        Run class_method_list elements in sequence on target object
        :: INPUT ::
        :param obj: [object] candidate object

        :: OUTPUT ::
        :return obj: [object] candidate object with self-altering
                              class methods applied
        """
        for _e in self.class_method_list:
            obj = eval(f"obj{_e}")
        return obj

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

        if not isinstance(x, wcd.deque):
            raise TypeError(f'input "{x}" is not a deque')
        # Get initial length of input deque
        xlen = len(x)
        # Iterate across deque contents (includes empty deque)
        for _i in range(self.max_pulse_size):
            # If deque has 0-length, early stop
            if len(x) == 0:
                break
            # If iteration counter reaches initial deque length, stop early
            elif _i + 1 > xlen:
                break
            # Otherwise
            else:
                # Pop off right-most element
                _x = x.popleft()
                # Check if it matches class_type
                if isinstance(_x, self.class_type):
                    # If it matches, run class_methods
                    _y = self._run_class_method_list(_x)
                    # Append processed item to this wyrm's deque
                    self.queue.append(_y)
                    # del _x # TODO: Double check if these pop/append
                # If the class does not match, appendleft the popped element back onto the input
                else:
                    x.append(_x)

        # Provide an alias of this wyrm's queue
        y = self.queue
        return y



###################################################################################
# WaveformModel WYRM CLASS DEFINITION #############################################
###################################################################################
    
class WaveformModelWyrm(Wyrm):

    def __init__(
            self,
            model,
            weight_names,
            devicetype='cpu',
            max_samples=15000,
            stacking_method='max',
            max_pulse_size=1000,
            debug=False
    ):
        """
        Initialize a WaveformModelWyrm object

        :: INPUS ::
        :param model: [seisbench.models.WaveformModel] model architecture defining
                        object based on the SeisBench WaveformModel class
        :param weight_names: [list-like] of [str]
                        one or more model weights that can be loaded for `model`
                        using the `model.from_pretrained()` method
        :param devicetype: [str] device type to run predictions on
                        see torch.device()
        :param max_samples: [int] maximum number of samples for each PredictionBuffer
                        at the terminations of this MachineWyrm's buffer attribute
                        see wyrm.core.buffer.prediction.PredictionBuffer(max_samples)
        :param stacking_method: [str] stacking method to pass to terminating PredictionBuffer
                        objects. See wyrm.core.buffer.prediction.PredictionBuffer(stacking_method)
        :param max_pulse_size: [int] used as a maximum batch size here for windowed data 
                        accumulation prior to prediction.
        :param debug: [bool] run in debug mode?
        """
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)

        # model compatability checks
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError('model must be a seisbench.models.WaveformModel object')
        elif model.name == 'WaveformModel':
            raise TypeError('model must be a child-class of the seisbench.models.WaveformModel class')
        else:
            self.model = model
        
        # Map model labels to model codes
        self.label_codes = ''
        for _l in self.model.labels:
            self.label_codes += _l[0]

        # Model weight_names compatability checks
        pretrained_list = model.list_pretrained()
        if isinstance(weight_names, str):
            weight_names = [weight_names]
        elif isinstance(weight_names, (list, tuple)):
            if not all(isinstance(_n, str) for _n in weight_names):
                raise TypeError('not all listed weight_names are type str')
        else:
            for _n in weight_names:
                if _n not in pretrained_list:
                    raise ValueError(f'weight_name {_n} is not a valid pretrained model weight_name for {model}')
        self.weight_names = weight_names

        # device compatability checks
        if not isinstance(devicetype, str):
            raise TypeError('devicetype must be type str')
        else:
            try:
                device = torch.device(devicetype)
            except RuntimeError:
                raise RuntimeError(f'devicetype {devicetype} is an invalid device string for PyTorch')
            try:
                self.model.to(device)
            except RuntimeError:
                raise RuntimeError(f'device type {devicetype} is unavailable on this installation')
            self.device = device

        # max_samples compatability checks
        self.max_samples = wuc.bounded_floatlike(
            max_samples,
            name='max_samples',
            minimum=2*self.model.in_samples,
            maximum=None,
            inclusive=False
        )

        # stacking_method compatability checks
        if not isinstance(stacking_method, str):
            raise TypeError('stacking_method must be type str')
        elif stacking_method.lower() in ['max','maximum']:
            self.stacking_method='max'
        elif stacking_method.lower() in ['avg','mean','average']:
            self.stacking_method='avg'
        else:
            raise ValueError(f'stacking_method {stacking_method} not supported. Must be "max", "avg" or select aliases')
        # Initialize TieredBuffer terminating in PredBuff objects
        self.buffer = wcd.BufferTree(
            buff_class=wcd.PredictionBuffer,
            max_samples=self.max_samples,
            stacking_method=self.stacking_method
            )
    
    def pulse(self, x, **options):
        """
        Conduct a pulsed data accumulation from an input deque of PredictionWindow objects
        convert them into a torch.Tensor, and conduct predictions on each input data window
        with each weight specified in self.weight_Names

        :: INPUTS ::
        :param x: [deque] of [wyrm.core.window.predicition.PredictionWindow] objects
        :param options: [kwargs] key word arguments to pass to self.buffer.append()
                        see wyrm.core.data.PredictionBuffer.append()
                            -> include_blinding=False (default)
        :: OUTPUT ::
        :retury y: [wyrm.core.data.BufferTree] budding in [wyrm.core.data.PredicitonBuffer] objects
                    with tier keys:
                        0: Site code(e.g,. 'UW.GNW.--')
                        1: Band / Instrument code (e.g., 'BH')
                        2: Model Weight (e.g., 'pnw')
        """
        if not isinstance(x, wcd.deque):
            raise TypeError('input x must be type deque')

        qlen = len(x)
        batch_data = []
        batch_meta = []
        for _i in range(self.max_pulse_size):
            # Early stopping clauses
            # if the input queue is empty or every queue element has been assessed
            if len(x) == 0:
                break
            elif _i == qlen:
                break
            else:
                _x = x.pop()
                if not isinstance(_x, wcd.PredictionWindow):
                    x.appendleft(_x)
                else:
                    _tensor, _meta = _x.split_for_ml()
                    batch_data.append(_tensor)
                    batch_meta.append(_meta)
                    # # Clean up iterated copies
                    del _tensor, _meta, _x

        batch_data = torch.concat(batch_data)

        for wname in self.weight_names:
            # Load model weights
            self.model = self.model.from_pretrained(wname)
            # Run Prediction
            batch_pred = self.run_prediction(batch_data)
            # Convert to numpy
            batch_pred_npy = self.preds_torch2npy(len(batch_meta), batch_pred)
            del batch_pred #, batch_data <- No, Bad Nate, Don't delete the source data here!

            # Split into individual windows and reconstitute PredictionWindows
            for _i, meta in enumerate(batch_meta):
                # Form keys (this is also included in buffer.append_pwind)
                tk0 = '.'.join(meta['id'].split('.')[:3])
                tk1 = meta['id'].split('.')[-1][:-1]
                tk2 = wname
                meta.update({'labels': self.model.labels,
                             'weight_name': wname})
                # Reconstitute PredictionWindows
                pwind = wcd.PredictionWindow(data=batch_pred_npy[_i,:,:], **meta)
                # Append results as copy, allowing safe deletion of intermediate data objects
                # self.buffer.append(pwind.copy(), TK0=tk0, TK1=tk1, TK2=tk2, **options)
                self.buffer.append_pwind(pwind.copy(), **options)
                # Cleanup prediction windows and associated metadata between iterations
                del pwind, meta, tk0, tk1, tk2
            # Cleanup prediction arrays between pulses
            del batch_pred_npy
        # Cleanup at end of pulse
        del batch_data, batch_meta
        
        y = self.buffer

        return y
    
    def run_prediction(self, batch_data):
        """
        Run a ML prediction on an input batch of windowed data using self.model on
        self.device. Provides checks that self.model and batch_data are appropriately
        formatted and staged on the same device

        :: INPUT ::
        :param batch_data: [numpy.ndarray] or [torch.Tensor] data array with scaling
                        appropriate to the input layer of self.model
        :: OUTPUT ::
        :return batch_preds: [torch.Tensor] or [tuple] thereof - prediction outputs
                        from self.model
        """
        if not isinstance(batch_data, (torch.Tensor, np.ndarray)):
            raise TypeError('batch_data must be type torch.Tensor or numpy.ndarray')
        elif isinstance(batch_data, np.ndarray):
            batch_data = torch.Tensor(batch_data)
        
        if self.model.device.type != self.device.type:
            self.model.to(self.device)
        
        if batch_data.device.type != self.device.type:
            batch_preds = self.model(batch_data.to(self.device))
        else:
            batch_preds = self.model(batch_data)
        
        return batch_preds
    
    def preds_torch2npy(self, nwind, preds):
        """
        Conduct minimal postprocessing to convert raw prediction outputs (preds)
        from a WaveformModel into a numpy.ndarray with the following axes
            0 - windows
            1 - labels
            2 - predicted values
        
        :: INPUTS ::
        :param nwind: [int] number of windows in preds
        :param preds: [torch.Tensor] or [tuple] thereof 
                    raw output from a WaveformModel object
        
        :: OUTPUT ::
        :return preds_npy: [numpy.ndarray] (nwind, nlabel, nvalues) array
                    containing predicted values from preds
        """
        out_shape = (nwind, len(self.model.labels), self.model.in_samples)
        # If output is a list-like of torch.Tensors (e.g., EQTransformer raw output)
        if isinstance(preds, (tuple, list)):
            if all(isinstance(_t, torch.Tensor) for _t in preds):
                preds = torch.concat(preds)
            else:
                raise TypeError('not all elements of preds is type torch.Tensor')
        # If reshaping is necessary
        if preds.shape != out_shape:
            preds = preds.reshape(out_shape)
        
        # Extract to numpy
        if preds.device.type != 'cpu':
            preds_npy = preds.detach().cpu().numpy()
        else:
            preds_npy = preds.detach().numpy()
        return preds_npy