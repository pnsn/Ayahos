"""
:module: PULSE.seq.ring2ring
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose: This sequence provides a set of worked examples of PyEarthworm-style RING2RING
    workflows using PULSE (i.e., reading and writing messages from/to Earthworm memory rings).

    .. rubric:: General Structure
    squence = {import: TransactMod, process: SequenceMod, export:TransactMod}

    All classes defined in this module are child-classes of the :class:`~PULSE.mod.sequence.SequenceMod` class
    and inherit it's capabilities
"""
import os, sys
from PULSE.mod.base import BaseMod
from PULSE.mod.sequencer import SequenceMod
from PULSE.mod.pyew import PyEWMod, EWTransactMod


class EW_R2R_Seq(SequenceMod):

    def __init__(self,
        pyew_module,
        import_method='get_wave',
        import_conn=None,
        import_msg_type=19,
        export_method='put_wave',
        export_conn=None,
        export_msg_type=19,
        process_seq=[],
        meta_max_age=60,
        max_pulse_size=1,
        maxlen=None,
        name_suffix=None
        ):
        """Create a PyEarthworm-style RING2RING workflow in the form of a PULSE :class:`~PULSE.seq.pyew_r2r.EW_R2R_Mod` module object
        that can house an arbitrary processing workflow.

        :param pyew_module: initialized PyEWMod object with definied connections to a running Earthworm instance
        :type pyew_module: PULSE.mod.pyew.PyEWMod
        :param import_method: EWTransactMod method for getting messages from the import_conn, defaults to 'get_wave'
            also see :class:`~PULSE.mod.pyew.EWTransactMod`
        :type import_method: str, optional
        :param import_conn: Import connection name matching an entry in **pyew_module.conn**, defaults to None.
            Note: input of None uses the `DEFAULT` connection
            also see :class:`~PULSE.mod.pyew.EWTransactMod`
        :type import_conn: str, optional
        :param import_msg_type: Import message type (consult your installation's earthworm_global.d file), defaults to 19
            19 is the code for TYPE_TRACEBUFF2 messages
            also see :meth:`~PULSE.util.earthworm.translate_message_codes`
        :type import_msg_type: int, optional
        :param export_method: EWTransactMod method for putting messages on the export_conn, defaults to 'put_wave'
            also see :class:`~PULSE.mod.pyew.EWTransactMod`
        :type export_method: str, optional
        :param export_conn: Export connection name matching an entry in **pyew_module.conn**, defaults to None
        :type export_conn: str, optional
        :param export_msg_type: Export message type (consult your installation's earthworm_global.d file), defaults to 19
        :type export_msg_type: int, optional
        :param process_seq: processing sequence to insert between the import and export modules, defaults to None
            Must be compatable with the **sequence** input for :class:`~PULSE.mod.sequence.SequenceMod`.
        :type process_seq: NoneType, PULSE.mod.base.BaseMod-like or list/dict's thereof, optional
        :param meta_max_age: Maximum relative age of metadata, in seconds, to preserve in the **metadata** attribute of this class, defaults to 60.
            also see :class:`~PULSE.mod.sequence.SequenceMod`
        :type meta_max_age: float-like, optional
        :param max_pulse_size: maximum number of iterations for the :meth:`~PULSE.mod.sequence.SequenceMod.pulse` method this class inherits, defaults to 1.
            also see :class:`~PULSE.mod.sequence.SequenceMod`
        :type max_pulse_size: int, optional
        :param maxlen: maximum length of this module's **output** attribute, defaults to None
            also see :class:`~PULSE.mod.sequence.SequenceMod`
        :type maxlen: int, optional
        :param name_suffix: suffix to attatch to the __name__ of this object, defaults to None.
        :type name_suffix: NoneType, int, str, optional
        """
        # Initialize Ring2RingMod as an empty sequence
        super().__init__(sequence=None,
                         meta_max_age=meta_max_age,
                         max_pulse_size=max_pulse_size,
                         maxlen=maxlen,
                         name_suffix=name_suffix)

        # Compatability check on pyew_module
        if not isinstance(pyew_module, PyEWMod):
            self.Logger.critical(f'pyew_module is not type PULSE.mod.pyew.PyEWModule. Exiting on EX_DATAERR ({os.EX_DATAERR})')
            sys.exit(os.EX_DATAERR)

        # Initialize import module
        inmod = EWTransactMod(pyew_module,
                              conn_name=import_conn,
                              pulse_method=import_method,
                              msg_type=import_msg_type)
        # Attach the import module to the sequence
        self.update({import_method: inmod})

        # TODO: Shift this check into :meth:`~PULSE.mod.sequence.SequenceMod.update`
        # Compatability check for seq
        if isinstance(process_seq, BaseMod):
            self.update({process_seq.__name__():process_seq})
        elif isinstance(process_seq, list):
            if all(isinstance(_e, BaseMod) for _e in process_seq):
                for _m in process_seq:
                    self.update({_m.__name__(): _m})
            else:
                self.Logger.critical(f'Not all elements in list-type "process_seq" are type BaseMod. Exiting on EX_DATAERR ({os.EX_DATAERR})')
                sys.exit(os.EX_DATAERR)
        elif isinstance(process_seq, dict):
            if all(isinstance(_e, BaseMod) for _e in process_seq.values()):
                self.update(process_seq)
            else:
                self.Logger.critical(f'Not all values in dict-type "process_seq" are type BaseMod. Exiting on EX_DATAERR ({os.EX_DATAERR})')
                sys.exit(os.EX_DATAERR)
        else:
            self.Logger.critical(f'process_seq type {type(process_seq)} not supported. Must be BaseMod or dict/list thereof. Exiting on EX_DATAERR ({os.EX_DATAERR})')
            sys.exit(os.EX_DATAERR)

        # Initialize export module
        outmod = EWTransactMod(pyew_module,
                              conn_name=export_conn,
                              pulse_method=export_method,
                              msg_type=export_msg_type)

        self.update({export_method: outmod})
        