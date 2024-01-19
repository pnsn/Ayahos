"""
:module: wyrm.structures.rtinststream
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains the class definition for Realtime Instrument Stream
    objects "RtInstStream" that provide a shallow hierarchical dictionary
    structure based on individual seismometer codes and component codes
    for collections of Realtime Buffer Traces (RtBuffTrace).

    The added structure provides appreciable search acceleration compared
    to a comparable ObsPy stream representation of a Trace collection for
    forming data tensor inputs common to machine learning algorithms
    e.g., EQTransformer (Mousavi et al., 2019), PhaseNet (Zhu & Beroza, 2018)
    and for reconstituting their outputs into continuous streams of modeled
    feature probabilities.
"""
from obspy import Trace, Stream
from wyrm.structures.rtbufftrace import RtBuffTrace
import wyrm.util.input_compatability_checks as icc
from copy import deepcopy
import numpy as np
import fnmatch
from tqdm import tqdm


class RtInstStream(dict):
    """
    Realtime Instrument Stream
    This object houses sets of wyrm.structures.rtbufftrace.RtBuffTrace
    objects organized in a 2-layered python dictionary structure to
    provide accelerated trace quering via hash-table mechanics that
    underly dictionary objects.

    This structure groups data by instrument components (i.e. data streams
    coming from a single seismic sensor), with a top-level key set composed
    of Network, Station, Location, and SEED Band+Instrument codes (N, S, L, I)
    matching either the NSLC or SNCL ID format used in ObsPy and Earthworm,
    respectively, becoming NSLC -> NSLI or SNCL -> SNIL. Within each of these
    keys the SEED component code for each channel is used as a secondary key

    Example Structure with one 6-component, one 3-component, and one 1-component
    {'UW.TOUCH..EN':
        {'Z': RtBuffTrace(UW.TOUCH..ENZ),
         'N': RtBuffTrace(UW.TOUCH..ENN),
         'E': RtBuffTrace(UW.TOUCH..ENE)},
     'UW.TOUCH..HH':
        {'Z': RtBuffTrace(UW.TOUCH..HHZ),
         'N': RtBuffTrace(UW.TOUCH..HHN),
         'E': RtBuffTrace(UW.TOUCH..HHE)},
     'UW.SLA.00.HN':
        {'1': RtBuffTrace(UW.SLA.00.HN1),
         '2': RtBuffTrace(UW.SLA.00.HN2),
         'Z': RtBuffTrace(UW.SLA.00.HNZ)},
     'UW.GPW.01.EH':
        {'Z': RtBuffTrace(UW.GPW.01.EHZ)}
    }
    """

    def __init__(self, max_length=180, id_fmt="NSLC"):
        # Inherit dict attributes
        super().__init__(self)
        # Compatability check for max_length
        self.max_length = icc.bounded_intlike(
            max_length, name="max_length", minimum=0, maximum=None, inclusive=False
        )
        # Compatability check for id_fmt
        if not isinstance(id_fmt, str):
            raise TypeError("id_fmt must be type str")

        elif id_fmt.upper() == "NSLC":
            self.id_fmt = "NSLC"
            self.k1_fmt = "{net}.{sta}.{loc}.{band}{inst}"
        elif id_fmt.upper() == "SNCL":
            self.id_fmt = "SNCL"
            self.k1_fmt = "{sta}.{net}.{band}{inst}.{loc}"
        else:
            raise ValueError("id_fmt must be NSLC or SNCL (case insensitive)")

    def copy(self):
        return deepcopy(self)

    def _add_branch(self, k1, k2, verbose=False):
        """
        Add a branch to this RtInstStream if it does
        not already exist and populate with an empty
        RtBuffTrace with max_length = self.max_length.

        :: INPUTS ::
        :param k1: [str] level 1 key formatted as
                    N.S.L.BI (e.g., UW.GNW.--.BH)
                    or
                    S.N.BI.L (e.g., GNW.UW.BH.--)
                    must conform with self.id_fmt
        :param k2: [str] level 2 key containing the
                    component code. Normally 'Z', 'N', 'E',
                    '1', '2', or '3'.
        :param verbose: [bool] - display already exists message
                    in the even that the branch is already present?
        """
        if not isinstance(k1, str):
            raise TypeError("k1 must be type str")
        elif len(k1.split(".")) != 4:
            raise SyntaxError('k1 must be a 4 element, "."-delimited string')
        if not isinstance(k2, str):
            raise TypeError("k2 must be type str")
        if k1 not in self.keys():
            self.update({k1: {k2: RtBuffTrace(max_length=self.max_length)}})
        elif k2 not in self[k1].keys():
            self[k1].update({k2: RtBuffTrace(max_length=self.max_length)})
        else:
            if not verbose:
                print(f"self[{k1}][{k2}] already exists")
        return self

    def append(self, input):
        """
        Append a waveform data containing object to this
        RtInstStream.
        :: INPUT ::
        :param input: [wave-dict] - see _append_wave()
                      [obspy.Trace] - see _append_trace()
                      [list-like of obspy.Trace] - see _append_traces()
                      [list-like of PyEW wave-dict] - see _append_waves()
        """
        if isinstance(input, Trace):
            self._append_trace(input)
        elif icc.isPyEWwave(input):
            self._append_wave(input)
        elif all([isinstance(tr, Trace) for tr in input]):
            self._append_traces(input)
        elif all([icc.isPyEWwave(_x) for _x in input]):
            self._append_waves(input)
        else:
            raise TypeError(
                f"input {input} does not conform to obspy.Trace\
                              or PyEW wave-dict types, or lists thereof"
            )
        return self

    def _append_trace(self, trace):
        """
        Primary append subroutine for appending a trace object to
        an existing RtBuffTrace object contained in this RtInstStream.

        If the specific channel does not exist in the branch structure
        of this RtInstStream, a new branch is created via self._add_branch()

        :: INPUT ::
        :param trace: [obspy.core.trace.Trace] trace object to append
        """
        if not isinstance(trace, Trace):
            raise TypeError("input trace must be type obspy.Trace")
        k1 = self.k1_fmt.format(
            net=trace.stats.network,
            sta=trace.stats.station,
            loc=trace.stats.location,
            band=trace.stats.channel[0],
            inst=trace.stats.channel[1],
        )
        k2 = trace.stats.channel[2:]

        if k1 not in self.keys():
            self._add_branch(k1, k2)
        elif k2 not in self[k1].keys():
            self._add_branch(k1, k2)

        self[k1][k2].append(trace)
        return self

    def _append_traces(self, traces):
        """
        Wrapper around _append_trace() that accepts list-like collections
        of obspy.Trace as an input (i.e., iterable sets of obspy.Trace objects).

        :: INPUT ::
        :param trace: collection of traces to append.
                     recommended formats:
                        [obspy.core.stream.Stream]
                        [list]
                        [deque]
        """
        if not icc.isiterable(traces):
            raise TypeError("traces must be a list-like object")
        if not all([isinstance(tr, Trace) for tr in traces]):
            raise TypeError("contents of traces is not all traces")
        if len(traces) > 0:
            for trace in traces:
                self._append_trace(trace)
        return self

    def _append_wave(self, wave):
        """
        Append a PyEarthworm formatted `wave` dictionary [wave-dict]
        to this RtInstStream

        :: INPUT ::
        :param wave: [dict] wave message formatted dictionary generated
                    by PyEW.EWModule.get_wave().
        """
        if not icc.isPyEWwave(wave):
            raise TypeError(
                "input wave does not conform to the PyEarthworm `wave` message format"
            )
        header = {
            "station": wave["station"],
            "network": wave["network"],
            "channel": wave["channel"],
            "location": wave["location"],
            "sampling_rate": wave["samprate"],
            "starttime": wave["startt"],
        }
        data = np.array(wave["data"], dtype=wave["dtype"])
        trace = Trace(data=data, header=header)
        self._append_trace(trace)
        return self

    def _append_waves(self, waves):
        """
        Append a list-like collection of PyEarthworm formatted wave-dict
        objects to this RtInstStream

        :: INPUT ::
        :param waves: [list-like] of [wave-dict] objects
        """
        if not icc.isiterable(waves):
            raise TypeError("waves must be a list-like object")
        if not all([icc.isPyEWwave(wave) for wave in waves]):
            raise TypeError("contents of waves is not all wave-dict objects")
        if len(waves) > 0:
            for wave in waves:
                self._append_wave(wave)
        return self

    def select(self, k1, k2="*", as_copy=True, alt_format=None):
        if not isinstance(k1, str):
            raise TypeError("k1 must be type str")
        if not isinstance(k2, str):
            raise TypeError("k2 must be type str")
        if not isinstance(alt_format, (str, type(None))):
            raise TypeError(
                'alt_format must be type str or None. Supported values: "stream" and "list". All others default to RtInstStream output'
            )
        # Handle alternative formatting
        if alt_format.lower() == "stream":
            out = Stream()
        elif alt_format.lower() == "list":
            out = []
        else:
            out = RtInstStream(max_length=self.max_length, id_fmt=self.id_fmt)
        for _k1 in fnmatch.filter(self.keys(), k1):
            for _k2 in fnmatch.filter(self[k1].keys(), k2):
                if as_copy:
                    rtbt = self[k1][k2].copy()
                else:
                    rtbt = self[k1][k2]
                out.append(rtbt)

        return out

    def _repr_line(self, key):
        _l2 = self[key]
        # Display key
        rstr = f"{key:<15}"
        if len(_l2.keys()) > 0:
            # tsmax = UTCDateTime(0)
            # tsmin = UTCDateTime("2500-01-01")
            sr = [_l2[_k2].stats.sampling_rate for _k2 in _l2.keys()]
            if min(sr) == max(sr):
                rstr += f" | {sr[0]:.1f} Hz, "
            else:
                rstr += f" | {min(sr):.1f} - {max(sr):.1f} Hz, "
        for _k in _l2.keys():
            ntot = len(_l2[_k])
            # if _l2[_k].stats.starttime >
            if np.ma.is_masked(_l2[_k].data):
                numa = sum(_l2[_k].data.mask)
                rstr += f" [{_k}] {numa:d} ({ntot:d}) npts"
            else:
                rstr += f" [{_k}] {ntot:d} npts"
        rstr += "\n"
        return rstr

    def __str__(self, extended=False):
        """
        Debugging representation string
        """
        ntr = 0
        for _k1 in self.keys():
            for _k2 in self[_k1].keys():
                if isinstance(self[_k1][_k2], RtBuffTrace):
                    ntr += 1
        rstr = (
            f"RtInstStream with {len(self)} instrument(s) comprising {ntr} trace(s)\n"
        )
        for _i, _k in enumerate(self.keys()):
            if _i < 1:
                rstr += self._repr_line(_k)
            elif _i == 1 and not extended:
                rstr += f"...\n({len(self) - 2} other instruments)\n...\n"
            elif _i == len(self) - 1 and not extended:
                rstr += self._repr_line(_k)
            elif extended:
                rstr += self._repr_line(_k)
            else:
                pass
        if not extended:
            rstr += '[Use "print(RtInstStream.__str__(extended=True))" to print all sub-dictionary summaries]'
        return rstr

    def __repr__(self, extended=False):
        rstr = self.__str__(extended=extended)
        return rstr

    