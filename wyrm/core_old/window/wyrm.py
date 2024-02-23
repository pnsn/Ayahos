from wyrm.core._base import Wyrm
from wyrm.core.buffer.structure import TieredBuffer
from wyrm.core.buffer.trace import TraceBuffer
from wyrm.core.window.instrument import InstrumentWindow
import wyrm.util.compatability as wcc
import seisbench.models as sbm
from obspy import UTCDateTime
from collections import deque
from copy import deepcopy
import numpy as np


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
        target_sr=100.0,
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
        :param target_sr: [float]
                        Target sampling_rate to pass to InsturmentWindow init
        :param target_npts: [int]
                        Target temporal samples to pass to InstrumentWindow init
        :param target_overlap: [int]
                        Target overlap between sequential windows. Used alongside
                        target_sr and target_npts to determine window advances
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
        self.target_sr = wcc.bounded_floatlike(
            target_sr,
            name='target_sr',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # Target window number of samples compat check
        self.target_npts = wcc.bounded_intlike(
            target_npts,
            name='target_npts',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # Target window number of sample overlap compat check
        self.target_overlap = wcc.bounded_intlike(
            target_overlap,
            name='target_overlap',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # Target (symmetrical) blinding number of samples compat check
        self.target_blinding = wcc.bounded_intlike(
            target_blinding,
            name='target_blinding',
            minimum=0,
            maximum=self.target_npts/2
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
        self.window_sec = self.target_npts/self.target_sr
        self.advance_sec = (self.target_npts - self.target_overlap)/self.target_sr
        self.blind_sec = self.target_blinding/self.target_sr

        # Create dict for holding instrument window starttime values
        self.default_starttime = None
        self.window_tracker = {}

        # Create queue for output collection of windows
        self.queue = deque([])

        # Update input and output types for TubeWyrm & compatability references
        self._update_io_types(itype=(TieredBuffer, TraceBuffer), otype=(deque, InstrumentWindow))

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
        # Check that x is compatable 
        self._matches_itype(x, raise_error=True)
        self._matches_itype(x.buff_class, raise_error=True)

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
        :param input: [wyrm.core.buffer.structure.TieredBuffer] terminating in 
                        [wyrm.core.buffer.trace.TraceBuffer] objects
        """
        # Run type checks for input
        self._matches_itype(input, raise_error=True)
        self._matches_itype(input.buff_class, raise_error=True)
        if not isinstance(pad, bool):
            raise TypeError('pad must be type bool')
        if isinstance(wgt_taper_len, str):
            if wgt_taper_len.lower() != 'blinding':
                raise TypeError('The only supported str-type wgt_taper_len input is "blinding"')
        elif not isinstance(wgt_taper_len, (int,float)):
            raise TypeError('wgt_taper_len must be type int, float, or str "blinding"')
        if not isinstance(wgt_taper_type, str):
            raise TypeError('wgt_taper_type must be type str')
        
        # Start number of new windows counter
        nnew = 0
        # Iterate across branches
        for k0 in input.keys():
            branch = input[k0]
            # Initialize new tracker branch if this branch key is not in window_tracker
            if k0 not in self.window_tracker.keys():
                self.window_tracker.update({k0: self.default_starttime})
            # Get next starttime
            next_ts = self.window_tracker[k0]
            # If next starttime is the default value, scrape starttime from branch's Z-component
            if next_ts == self.default_starttime:
                for _c in self.code_map['Z']:
                    if _c in branch.keys():
                        if len(branch[_c]) > 0:
                            first_ts = branch[_c].stats.starttime
                            self.window_tracker.update({k0: first_ts})
                            next_ts = first_ts
                            break
            # If a valid next starttime is present (or was scraped), proceed
            if isinstance(next_ts, UTCDateTime):
                data_ts = None
                data_te = None
                # Iterate across entries in branch
                for k1 in branch.keys():
                    # If a branch entry has data
                    if len(branch[k1]) > 0:
                        # Grab the starttime, endtime and buffer length
                        data_ts = branch[k1].stats.starttime
                        data_te = branch[k1].stats.endtime
                        buff_length = branch[k1].max_length
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
                        self.window_tracker[k0] += nadv*self.advance_sec
                        next_ts = self.window_tracker[k0]
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
                    self.queue.appendleft(iwind)
                    nnew += 1
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
            extra_sec = wcc.bounded_floatlike(
                extra_sec, name="extra_sec", minimum=0.0, maximum=self._window_sec
            )
        # wgt_taper_len compatability checks
        if isinstance(wgt_taper_len, str):
            if wgt_taper_len.lower() == "blinding":
                wgt_taper_len = self.blinding_sec
            else:
                raise SyntaxError(f'str input for wgt_taper_len {wgt_taper_len} not supported. Supported: "blinding"')
        else:
            wgt_taper_len = wcc.bounded_floatlike(
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
        # Compose kwarg dictionary for RtBuffTrace.get_trimmed_valid_fract()
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
            iwind = InstrumentWindow(**window_inputs)
            
        else:
            iwind = None
        return iwind

    # ############### #
    # DISPLAY METHODS #
    # ############### #

    def __repr__(self):
        rstr = super().__str__()
        rstr += f" | Windows Queued: {len(self.queue)}\n"
        for _c in self.windowing_attr["target_order"]:
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
        rstr += f'target_sr={self.target_sr}, target_npts={self.target_npts}, '
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
            self.target_sr = model.sampling_rate
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