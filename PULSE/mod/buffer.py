"""
:module: PULSE.mod.buffer
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This contains the definition for the BufferModule class that hosts a :class:`~PULSE.data.dictstream.DictStream`
    object containing :class:`~PULSE.data.mltracebuff.MLTraceBuff` objects keyed to the Buffers' id attribute
"""
import logging, sys, os
# from collections import deque
# from numpy import isfinite
from PULSE.mod.base import BaseMod
from PULSE.data.mltrace import MLTrace
from PULSE.data.mltracebuff import MLTraceBuff
from PULSE.data.dictstream import DictStream
# from PULSE.util.pyew import is_wave_msg, wave2mltrace

Logger = logging.getLogger(__name__)

# @add_class_name_to_docstring
class BufferMod(BaseMod):
    """
    Module class for buffering/stacking MLTrace objects into a :class:`~PULSE.data.dictstream.DictStream` containing
    sets of :class:`~PULSE.data.mltracebuff.MLTraceBuff` objects with options to format the method by which identically
    keyed traces are joined (via :meth:`~PULSE.data.mltrace.MLTrace.__add__`).

    .. rubric:: For Raw Waveform Buffering
    We recommend starting with the pre-set values provided.

    .. rubric:: For SeisBench WaveformModel Prediction Buffering (Stacking)
    We recommend the following adjustments to the preset arguments:
        - pre_blinding -- set to similar values as your model's **_annotate_args['blinding']** value
        - add_method -- set to 2 ('max') or 3 ('avg') to match the corresponding value in your model's
            **_annotate_args['stacking'] value.
        - bufflen -- set to 3+ times the prediction window length (model.pred_sample/model.sampling_rate)
    E.g., For EQTransformer you might consider the following adjustments:
        - pre_blinding = (500,500)
        - add_method = 3
        - bufflen = 180.

    :param bufflen: maximum length of each :class:`~PULSE.data.mltracebuff.MLTraceBuff` generated in this this
        **BufferMod.output** in seconds, defaults to 300.
        Also see :class:`~PULSE.data.mltracebuff.MLTraceBuff`.
    :type bufflen: float-like, optional.
    :param restricted_appends: should the `restrict_past_append` rules of :class:`~PULSE.data.mltracebuff.MLTraceBuff`
        be enforced for any and all MLTraceBuff objects generated in **BufferMod.output**? Defaults to True.
    :type restricted_appends: bool, optional.
        Also see :class:`~PULSE.data.mltracebuff.MLTraceBuff`
    :param pre_blinding: specifieds if, and for how many samples, blinding should be applied to intput mltrace-like
        objects prior to being appended to (or initializing) MLTraceBuff objects in **BufferMod.output**. Defaults to False.
        Supported Formats:
            - int -- number of samples blinded (i.e., fold set to 0) on each ingested MLTrace object.
            - (int, int) -- number of samples blinded on the left and right ends of each ingested MLTrace object.
            - False -- no blinding is applied
    :type pre_blinding: bool, int, or 2-tuple of int, optional.
    :param add_method: value passed to :meth:`~PULSE.data.mltrace.MLTrace.__add__` that underlies methods joining
        MLTrace objects with matching **mltrace.id** attributes. Defaults to 1.
    :type add_method: int or str, optional.
        See :meth:`~PULSE.data.mltrace.MLTrace.__add__` for more information on other supported values
    :param max_pulse_size: maximum number of mltrace-like objects to detach from a :class:`~collections.deque` input
        to :meth:`~PULSE.mod.buffer.BufferMod.pulse` for each call of :meth:`~pulse`, defaults to 10000.
    :type max_pulse_size: int, optional.
    :param name_suffix: suffix string or integer to append to the end of this module's __name__, defaults to None.
    :type name_suffix: int, str, None, optional.
        also see :class:`~PULSE.mod.base.BaseMod`
    :param options: key-word argument collector for :meth:`~PULSE.data.mltrace.MLTrace.__add__` calls underlying
        MLTrace object merges/appends.
    :type options: kwargs, optional 
    """
    def __init__(
            self,
            bufflen=300.,
            restricted_appends=True,
            pre_blinding=False,
            add_method=1,
            max_pulse_size=10000,
            name_suffix=None,
            **options):
        """Initialize a :class:`~PULSE.mod.buffer.BufferMod` object

        :param bufflen: maximum length of each :class:`~PULSE.data.mltracebuff.MLTraceBuff` generated in this this
            **BufferMod.output** in seconds, defaults to 300.
            Also see :class:`~PULSE.data.mltracebuff.MLTraceBuff`.
        :type bufflen: float-like, optional.
        :param restricted_appends: should the `restrict_past_append` rules of :class:`~PULSE.data.mltracebuff.MLTraceBuff`
            be enforced for any and all MLTraceBuff objects generated in **BufferMod.output**? Defaults to True.
        :type restricted_appends: bool, optional.
            Also see :class:`~PULSE.data.mltracebuff.MLTraceBuff`
        :param pre_blinding: specifieds if, and for how many samples, blinding should be applied to intput mltrace-like
            objects prior to being appended to (or initializing) MLTraceBuff objects in **BufferMod.output**. Defaults to False.
            Supported Formats:
                - int -- number of samples blinded (i.e., fold set to 0) on each ingested MLTrace object.
                - (int, int) -- number of samples blinded on the left and right ends of each ingested MLTrace object.
                - False -- no blinding is applied
        :type pre_blinding: bool, int, or 2-tuple of int, optional.
        :param add_method: value passed to :meth:`~PULSE.data.mltrace.MLTrace.__add__` that underlies methods joining
            MLTrace objects with matching **mltrace.id** attributes. Defaults to 1.
        :type add_method: int or str, optional.
            See :meth:`~PULSE.data.mltrace.MLTrace.__add__` for more information on other supported values
        :param max_pulse_size: maximum number of mltrace-like objects to detach from a :class:`~collections.deque` input
            to :meth:`~PULSE.mod.buffer.BufferMod.pulse` for each call of :meth:`~pulse`, defaults to 10000.
        :type max_pulse_size: int, optional.
        :param name_suffix: suffix string or integer to append to the end of this module's __name__, defaults to None.
        :type name_suffix: int, str, None, optional.
            also see :class:`~PULSE.mod.base.BaseMod`
        :param options: key-word argument collector for :meth:`~PULSE.data.mltrace.MLTrace.__add__` calls underlying
            MLTrace object merges/appends.
        :type options: kwargs, optional 

        """
        # Inherit from BaseMod
        super().__init__(max_pulse_size=max_pulse_size, maxlen=None, name_suffix=name_suffix)

        # Initialize self.output as PULSE.data.dictstream.DictStream
        self.output = DictStream(key_attr='id')
        
        # Initialize demerits counter for __init__ (triggers system exit at end of __init__ if non-zero)
        demerits = 0

        # Create kwarg holder for initializing MLTraceBuff objects
        self._kwargs = options
        if isinstance(bufflen, (float, int)):
            if 1e7 > bufflen > 0:
                self._kwargs.update({'max_length': float(bufflen)})
            else:
                self.Logger.critical('bufflen must be a value in (0, 1e7).')
                demerits += 1
        else:
            self.Logger.critical('bufflen must be type int or float.')
            demerits += 1
        
        if isinstance(restricted_appends, bool):
            self._kwargs.update({'restrict_past_append': restricted_appends})
        else:
            self.Logger.critical('restricted appends must be type bool.')
            demerits += 1

        if isinstance(pre_blinding, (int, tuple, bool)):
            if isinstance(pre_blinding, tuple):
                if len(pre_blinding) != 2:
                    self.Logger.critical('tuple inputs for pre_blinding must be 2-tuples')
                    demerits += 1
                else:
                    pass
            self._kwargs.update({'blinding': pre_blinding})
        else:
            self.Logger.critical(f'pre_blinding input of type {type(pre_blinding)} not supported.')
            demerits += 1
        
        if add_method in [0,1,2,3,'dis','int','max','avg']:
            self._kwargs.update({'method': add_method})
        else:
            self.Logger.critical(f'add_method {add_method} not supported. See PULSE.data.mltrace.MLTrace.__add__')
            demerits += 1

        if demerits != 0:
            self.Logger.critical(f'BufferMod.__init__ raised {demerits} errors. Exiting on EX_DATAERR ({os.EX_DATAERR})')
            sys.exit(os.EX_DATAERR)
        else:
            self.Logger.debug(f'{self.__name__()} initalized successfully')
    
    #################################
    # PULSE POLYMORPHIC SUBROUTINES #
    #################################
    def pulse(self, input):
        """TEMPLATE METHOD

        Alias to the :meth:`~PULSE.mod.base.BaseMod.pulse` method to provide documentation
        for its behaviors associated with the :class:`~PULSE.mod.buffer.BufferMod` class.

        For each iteration in this method, MLTrace-like objects are popped off of the **input**,
        converted into MLTrace objects (if necessary) and added to the :class:`~PULSE.data.dictstream.DictStream`
        object **output**. If the **mltrace.id** attribute is not a key in **output** a new :class:`~PULSE.data.mltracebuff.MLTraceBuff`
        object is generated in **output** with that **id** as a new key. If the **id** is already present in **output**
        then the new MLTrace object(s) are appended to the existing entries.

        :param input: collection of MLTrace-like objects
        :type input: collections.deque
        :return: 
         - **output** (*PULSE.data.dictstream.DictStream**) - a DictStream object with **id** keys associated to MLTraceBuff objects

        """        
        output = super().pulse(input)
        return output

    def measure_input(self, input):
        """POLYMORPHIC METHOD

        :class:`~PULSE.mod.buffer.BufferMod` uses the :meth:`~PULSE.mod.base.BaseMod.measure_input` from
        :class:`~PULSE.mod.base.BaseMod` without alterations.

        :param input: collection of objects
        :type input: collections.deque
        :return:
         - **input_size** (*int*) -- length of input
        :rtype: _type_
        """        
        return super().measure_input(input)
    
    def measure_output(self):
        """POLYMORPHIC METHOD

        :class:`~PULSE.mod.buffer.BufferMod` uses the :meth:`~PULSE.mod.base.BaseMod.measure_output` from
        :class:`~PULSE.mod.base.BaseMod` without alterations.
        
        Return the length of **BufferMod.output**

        :return: 
         - **output_size** (*int*) -- length of **output**
        :rtype: _type_
        """        
        return super().measure_output()
    
        
    def get_unit_input(self, input):
        """POLYMORPHIC METHOD

        :class:`~PULSE.mod.buffer.BufferMod` uses the :meth:`~PULSE.mod.base.BaseMod.get_unit_input` from
        :class:`~PULSE.mod.base.BaseMod` without alterations.

        :param input: collection of :class:`~obspy.core.trace.Trace`-like objects
        :type input: collections.deque
        :return: 
         - **unit_output** (*obspy.core.trace.Trace*-like or *NoneType*) -- Trace-like object popleft'd from **input** or None
        """        
        return super().get_unit_input(input)

    def run_unit_process(self, unit_input):
        """POLYMORPHIC METHOD

        Last updated with :class:`~PULSE.mod.buffer.BufferMod`

        Converts **unit_input** into a :class:`~PULSE.data.mltrace.MLTrace` object using
        **unit_input** as an input argument for the __init__ method of the :class:`~PULSE.data.mltrace.MLTrace` class 
        if **unit_input** is not already an MLTrace.

        :param unit_input: Trace-like object
        :type unit_input: obspy.core.trace.Trace, PULSE.data.mltrace.MLTrace 
        :return:
         - **unit_output** (*PULSE.data.mltrace.MLTrace*) - MLTrace representation of the **unit_input**
        """
        # If input is already an MLTrace
        if isinstance(unit_input, MLTrace):
            unit_output = unit_input
        # If unit_input is not an MLTrace, try to make it one
        else:
            try:
                unit_output = MLTrace(unit_input)
            except:
                self.Logger.critical(f'unit_input of type {type(unit_input)} could not be converted into a MLTrace. Exiting on EX_DATAERR ({os.EX_DATAERR})')
                sys.exit(os.EX_DATAERR)
        return unit_output

    def store_unit_output(self, unit_output):
        """POLYMORPHIC METHOD

        Last updated with :class:`~PULSE.mod.buffer.BufferMod`

        Appends an MLTrace object to the **BufferMod.output** attribute, using the MLTrace's **id** as the key-value.
        
        If the **id** is not a key in **output**, a new :class:`~PULSE.data.mltracebuff.MLTraceBuff`
        object is initalized using the **unit_output** provided. 
        
        If the **id** is already a key in **output**
        the **unit_output** is appended to the existing MLTraceBuff entry.

        New MLTraceBuff objects are added to **output** using :meth:`~PULSE.data.dictstream.DictStream.extend`

        No trigger for early stopping (early1) provided in this method.

        :param unit_output: MLTrace object to append to the :class:`~PULSE.data.dictstream.DictStream` **output** object
        :type unit_output: PULSE.data.mltrace.MLTrace
        :return:
         - **None** (*NoneType*) -- returns None
        """        
        # Get ID
        if isinstance(unit_output, MLTrace):
            _id = unit_output.id
            # If the ID is not present in output keys (create new buffer)
            if _id not in self.output.keys():
                self.Logger.debug(f'generating new buffer entry for {_id}')
                _mltb = MLTraceBuff(**self._kwargs)
                _mltb.append(unit_output)
                self.output.extend(_mltb)
            # If the ID is present in the output keys (append to exisitng)
            else:
                self.output[_id].append(unit_output)
        else:
            self.Logger.critical(f'unit_output is type {type(unit_output)}. Must be type MLTrace. Exiting on EX_DATAERR ({os.EX_DATAERR})')
            sys.exit(os.EX_DATAERR)

        return None



    # def _unit_input_from_input(self, input):
    #     """
    #     POLYMORPHIC METHOD
    #     Last updated with :class:`~PULSE.mod.buffer.BufferMod`

    #     Claim the left-most object in `input` using popleft(), ensure it is a
    #     :class:`~PULSE.data.mltrace.MLTrace` object, or a collection thereof
    #     and convert single MLTrace objects into a 1-element list.

    #     :param input: collection of MLTrace-like objects
    #     :type input: collections.deque of PULSE.data.mltrace.MLTrace objects or list-like thereof
    #     :return unit_input: iterable list of MLTrace-like objects
    #     :rtype unit_input: list of PULSE.data.mltrace.MLTrace
    #     """        
    #     unit_input = input.popleft()
    #     if isinstance(unit_input, dict):
    #         if is_wave_msg(unit_input):
    #             unit_input = [wave2mltrace(unit_input)]
    #     if isinstance(unit_input, MLTrace):
    #         unit_input = [unit_input]
    #     # if listlike, convert to list
    #     elif all(isinstance(x, MLTrace) for x in unit_input):
    #         unit_input = [x for x in unit_input]
    #     else:
    #         self.Logger.error('input is not type MLTrace or list-like thereof')
    #         raise TypeError('input is not type MLTrace or a list-like thereof')
    #     return unit_input

    # def _unit_process(self, unit_input):
    #     """
    #     POLYMORPHIC METHOD
    #     Last updated with :class:`~PULSE.mod.buffer.BufferMod`

    #     Iterate across MLTraces in `unit_input` and either generate new
    #     MLTraceBuffers keyed by the MLTrace.id or append the MLTrace
    #     to an existing MLTraceBuffer with the same MLTrace.id.

    #     This method conducts output "capture" by generating new
    #     buffers or appending to existing buffers

    #     :param unit_input: iterable set of MLTrace unit_inputects
    #     :type unit_input: DictStream, list-like
    #     :return unit_output: standard output of _unit_process
    #     :rtype unit_output: None
    #     """ 
    #     nproc = 0      
    #     for mlt in unit_input:
    #         _key = getattr(mlt, self.buffer_key)
    #         # Add new buffer if needed
    #         if _key not in self.output.traces.keys():
    #             if len(self.output) < self.max_output_size:
    #                 # Initialize an MLTraceBuffer Object with pre-specified append kwargs
    #                 mltb = MLTraceBuff(**self.mltb_kwargs)
    #                 mltb.append(mlt)
    #                 self.output.extend(mltb)
    #             else:
    #                 self.Logger.critical('More traces than allowed by max_output_size - refusing to add new buffers')
    #                 sys.exit(1)
    #         # Append to buffer if id's match
    #         else:
    #             try:
    #                 self.output[_key].append(mlt)
    #             except TypeError:
    #                 breakpoint()
    #         nproc += 1
    #     unit_output = nproc
        
    #     return unit_output
    
    # def _capture_unit_output(self, unit_output):
    #     """
    #     POLYMORPHIC METHOD
    #     Last updated with :class:`~PULSE.mod.buffer.BufferMod`

    #     Placeholder/termination in case unit_output is not type int.

    #     :param unit_output: _description_
    #     :type unit_output: _type_
    #     """
    #     if not isinstance(unit_output,int):
    #         self.Logger.warning('Passing non-int object to BufferMod._capture_unit_output') 
    #     else:
    #         pass
    #     return None

    # def _should_next_iteration_run(self, unit_output):
    #     """
    #     POLYMORPHIC METHOD
    #     Last updated with :class:`~PULSE.mod.buffer.BufferMod`

    #     Signal early stopping (status = False) if unit_input = 0
    #     i.e., no new trace segments buffered

    #     :param unit_output: number of trace segments buffered by the last call of _unit_process()
    #     :type unit_output: int
    #     :return status: should the next iteration be run?
    #     :rtype: bool
    #     """        
    #     if unit_output > 0:
    #         status = True
    #     else:
    #         status = False
    #     return status