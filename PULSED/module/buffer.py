"""
:module: camper.module.buffer
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This contains the definition for the BufferModule class that hosts a :class:`~camper.data.stream.Stream`
    object containing :class:`~camper.data.trace.Buffer` objects keyed to the Buffers' id attribute

Classes
-------
:class:`~camper.data.stream.Stream`
:class:`~camper.data.stream.Window`
"""
import logging, sys
from numpy import isfinite
from PULSED.module._base import _BaseMod
from PULSED.data.mltrace import MLTrace
from PULSED.data.mltracebuff import MLTraceBuff
from PULSED.data.mlstream import MLStream
from PULSED.util.pyew import is_wave_msg, wave2mltrace

Logger = logging.getLogger(__name__)

# @add_class_name_to_docstring
class BufferMod(_BaseMod):
    """
    Class for buffering/stacking MLTrace objects into a MLStream of MLTraceBuffer objects with
    self-contained settings and sanity checks for changing the MLTraceBuffer.__add__ method options.
    """
    def __init__(
            self,
            buffer_key='id',
            max_length=300.,
            restrict_past_append=True,
            blinding=None,
            method=1,
            max_pulse_size=10000,
            meta_memory=3600,
            report_period=False,
            max_output_size=1e5,
            **add_kwargs):
        """Initialize a BufferMod object

        :param buffer_key: MLTrace attribute to use for the MLStream keys, defaults to 'id'
        :type buffer_key: str, optional
        :param max_length: Maximum MLTraceBuffer length in seconds, defaults to 300.
        :type max_length: positive float-like, optional
        :param restrict_past_append: Enforce MLTraceBuffer past append safeguards?, defaults to True
        :type restrict_past_append: bool, optional
        :param blinding: Apply blinding before appending MLTrace objects to MLTraceBuffer objects, defaults to None
                also see ewflow.core.trace.mltrace.MLTrace.apply_blinding
                Supported 
                    None - do not blind
                    (int, int) - left and right blinding sample counts (must be non-negative), e.g., (500, 500)
        :type blinding: None or 2-tuple of int, optional
        :param method: method to use for appending data to MLTrace-type objects in this BufferMod's buffer attribute, defaults to 1
                Supported values:
                    ObsPy-like
                        0, 'dis','discard' - discard overlapping dat
                        1, 'int','interpolate' - interploate between edge-points of overlapping data
                    SeisBench-like
                        2, 'max', 'maximum' - conduct stacking with "max" behavior
                        3, 'avg', 'average' - condcut stacking with "avg" behavior

                also see ewflow.core.trace.mltrace.MLTrace.__add__
                         ewflow.core.trace.mltracebuffer.MLTraceBuffer.append
        :type method: int or str, optional
        :param max_pulse_size: maximum number of items to pull from source deque (x) for a single .pulse(x) call for this BufferMod, defaults to 10000
        :type max_pulse_size: int, optional
        :param **add_kwargs: key word argument collector to pass to MLTraceBuffer's initialization **kwargs that in turn pass to __add__
                NOTE: add_kwargs with matching keys to pre-specified values will be ignored to prevent multiple call errors
        :type **add_kwargs: kwargs
        :raises TypeError: _description_
        :raises TypeError: _description_
        :raises ValueError: _description_
        """
        # Inherit from Wyrm
        super().__init__(
            max_pulse_size=max_pulse_size,
            meta_memory=meta_memory,
            report_period=report_period,
            max_output_size=max_output_size)

        # Initialize output of type ewflow.data.mlstream.MLStream
        self.output = MLStream(key_attr=buffer_key)
        self.buffer_key = buffer_key
        # Create holder attribute for MLTraceBuffer initialization Kwargs
        self.mltb_kwargs = {}
        if isinstance(max_length, (float, int)):
            if isfinite(max_length):
                if 0 < max_length:
                    if max_length > 1e7:
                        Logger.warning(f'max_length of {max_length} > 1e7 seconds. May be memory prohibitive')
                    self.mltb_kwargs.update({'max_length': max_length})
                else:
                    raise ValueError('max_length must be non-negative')
            else:
                raise ValueError('max_length must be finite')
        else:
            raise TypeError('max_length must be type float or int')

        # Do sanity checks on restrict_past_append
        if not isinstance(restrict_past_append, bool):
            raise TypeError('restrict_past_append must be type bool')
        else:
            self.mltb_kwargs.update({'restrict_past_append': restrict_past_append})

        # Do sanity checks on blinding
        if not isinstance(blinding, (type(None), bool, tuple)):
            raise TypeError
        else:
            self.mltb_kwargs.update({'blinding': blinding})

        if method in [0,'dis','discard',
                          1,'int','interpolate',
                          2,'max','maximum',
                          3,'avg','average']:
            self.mltb_kwargs.update({'method': method})
        else:
            raise ValueError(f'method {method} not supported. See ewflow.data.mltrace.MLTrace.__add__()')

        for _k, _v in add_kwargs:
            if _k not in self.mltb_kwargs:
                self.mltb_kwargs.update({_k: _v})                      

    
    #################################
    # PULSE POLYMORPHIC SUBROUTINES #
    #################################


    def _unit_input_from_input(self, input):
        """
        POLYMORPHIC
        Last updated with :class: `~ewflow.module.buffer.BufferMod`

        Claim the left-most object in `input` using popleft(), ensure it is a
        :class: `~ewflow.core.mltrace.MLTrace` object, or a collection thereof
        and convert single MLTrace objects into a 1-element list.

        :param input: collection of MLTrace-like objects
        :type input: collections.deque of ewflow.core.trace.mltrace.MLTrace or list-like thereof
        :return unit_input: iterable list of MLTrace-like objects
        :rtype unit_input: list of ewflow.core.trace.mltrace.MLTrace
        """        
        unit_input = input.popleft()
        if isinstance(unit_input, dict):
            if is_wave_msg(unit_input):
                unit_input = [wave2mltrace(unit_input)]
        if isinstance(unit_input, MLTrace):
            unit_input = [unit_input]
        # if listlike, convert to list
        elif all(isinstance(x, MLTrace) for x in unit_input):
            unit_input = [x for x in unit_input]
        else:
            Logger.error('input is not type MLTrace or list-like thereof')
            raise TypeError('input is not type MLTrace or a list-like thereof')
        return unit_input

    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last updated with :class: `~ewflow.module.buffer.BufferMod`

        Iterate across MLTraces in `unit_input` and either generate new
        MLTraceBuffers keyed by the MLTrace.id or append the MLTrace
        to an existing MLTraceBuffer with the same MLTrace.id.

        This method conducts output "capture" by generating new
        buffers or appending to existing buffers

        :param unit_input: iterable set of MLTrace unit_inputects
        :type unit_input: MLStream, list-like
        :return unit_output: standard output of _unit_process
        :rtype unit_output: None
        """ 
        nproc = 0      
        for mlt in unit_input:
            _key = getattr(mlt, self.buffer_key)
            # Add new buffer if needed
            if _key not in self.output.traces.keys():
                if len(self.output) < self.max_output_size:
                    # Initialize an MLTraceBuffer Object with pre-specified append kwargs
                    mltb = MLTraceBuff(**self.mltb_kwargs)
                    mltb.append(mlt)
                    self.output.extend(mltb)
                else:
                    Logger.critical('More traces than allowed by max_output_size - refusing to add new buffers')
                    sys.exit(1)
            # Append to buffer if id's match
            else:
                try:
                    self.output[_key].append(mlt)
                except TypeError:
                    breakpoint()
            nproc += 1
        unit_output = nproc
        
        return unit_output
    
    def _capture_unit_output(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class: `~ewflow.module.buffer.BufferMod`

        Placeholder/termination

        :param unit_output: _description_
        :type unit_output: _type_
        """
        if not isinstance(unit_output,int):
            Logger.warning('Passing non-int object to BufferMod._capture_unit_output') 
        else:
            pass
        return None

    def _should_next_iteration_run(self, unit_output):
        """
        POLYMORPHIC
        Last updated with :class: `~ewflow.module.buffer.BufferMod`

        Signal early stopping (status = False) if unit_input = 0
        i.e., no new trace segments buffered

        :param unit_output: number of trace segments buffered by the last call of _unit_process()
        :type unit_output: int
        :return status: should the next iteration be run?
        :rtype: bool
        """        
        if unit_output > 0:
            status = True
        else:
            status = False
        return status