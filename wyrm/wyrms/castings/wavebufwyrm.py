"""
:modul
"""


from wyrm.wyrms.wyrm import Wyrm
from wyrm.structures.castings.sncl_dict import RtBuff_Dict
from wyrm.message.trace import TraceMsg
import fnmatch
import PyEW


class WaveBuffWyrm(Wyrm):

    def __init__(
        self,
        module,
        conn_id,
        sub_str="*.*.*.*",
        buff_sec=300,
        max_pulse_size=12000
    ):
        """
        Initialize a WaveBuffWyrm object
        :: INPUTS ::
        :param module: [PyEW.EWModule] initialized PyEarthworm EWModule
                        with a connection to a WAVE ring
        :param conn_id: [int] identifier integer for the connection to the
                        Earthworm WAVE ring for provided module
                        Tip: Cross ref with the associated heartwyrm.conn_info
                            dataframe
        :param sub_str: [str] S.N.C.L formatted search string compatable with
                        fnmatch.filter() that will subset 
        """
        Wyrm.__init__(self)
        # Run compatability checks on module
        if isinstance(module, PyEW.EWModule):
            self.module = module
        else:
            raise TypeError("module must be a PyEW.EWModule object!")

        # Compatability check for conn_id
        if not isinstance(conn_id, int):
            raise TypeError("conn_id must be int")
        elif conn_id < 0:
            raise ValueError("conn_id must be 0+")
        else:
            self._conn_id = conn_id

        # Compatability checks for sub_str
        if not isinstance(sub_str, str):
            raise TypeError("sub_str must be type str")
        elif len(sub_str.split(".")) != 4:
            raise SyntaxError('sub_str must be a 4-element, "."-delimited string')
        else:
            self._sub_str = sub_str

        # Compatability / Sanity checks for buff_sec
        if isinstance(buff_sec, float):
            self.buff_sec = buff_sec
        elif isinstance(buff_sec, int):
            self.buff_sec = float(buff_sec)
        else:
            raise TypeError("buff_sec must be type int or float")

        if self.buff_sec < 0:
            raise ValueError("buff_sec must be positive")
        elif self.buff_sec < 30:
            UserWarning("buff_sec < 30. will likely be too short for ML windowing!")
        else:
            pass

        # Compatability checks for max_pulse_size
        if isinstance(max_pulse_size, int):
            self.pulse_size = max_pulse_size
        elif isinstance(max_pulse_size, float):
            self.pulse_size = int(max_pulse_size)
        else:
            raise TypeError("pulse_size must be type int")

        if self._pulse_size < 0:
            raise ValueError("max_pulse_size must be positive")
        elif self._pulse_size < 100:
            UserWarning(
                "max_pulse_size < 100 is rather low. Depending on pulse_rate from HeartWyrm, data may be missed"
            )
        else:
            pass

        # Initialize RtTrace Buffer Dictionary
        self.queue = RtBuff_Dict(
            main_key="rtbuff", buff_sec=self.buff_sec, extra_files=None
        )

    def __repr__(self, extended=False):
        rstr = f"MOD: {self.module} | CONN: {self._conn_id} | FLOW: EW -> PY\n"
        rstr += f"FILT: {self._sub_str} | BSEC: {self._buff_sec} sec\n"
        rstr += f"{self.queues.__repr__(extended=extended)}"
        return rstr

    def pulse(self, x=None):
        if x is not None:
            print(f'x={x} does nothing for WaveBuffWyrm. Proceeding')
        # Itrate for _max_pulse_size (but allow break)
        for _ in range(self._pulse_size):
            _wave = self.module.get_wave(self._conn_id)
            # If an empty message is returned, end iteration
            if _wave == {}:
                break
            # Otherwise
            else:
                # Convert into TraceMsg
                _msg = TraceMsg(_wave)
                # If using a sub_str, check that it matches
                if fnmatch.filter([_msg.sncl], self.sub_str) == [_msg.sncl]:
                    self.queue.append(_msg)
                # Otherwise return unaltered _wave to WAVE RING
                # TODO: Make sure this doesn't duplicate the message...
                else:
                    self.module.put_wave(_wave, self._conn_id)
        # Provide access to the queue as the pulse output
        y = self.queue
        return y
