"""
:module: wyrm.structures.rtstream
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

from obspy import Trace, Stream, Inventory
# from obspy.realtime.rttrace import RtTrace
from wyrm.structures.rtbufftrace import RtBuffTrace
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

    Use of RtBuffTrace objects to buffer data for a specified max_length
    eliminates the occurrence of multiple identical SNCL/NSLC keys resulting
    from sequential data packets that occurs in obspy.core.stream.Stream objects

    This structure accelerates data querying by roughly an order of magnitude
    compared to the stream.select() method on large in-memory collections of
    traces when trying to query slices of the collection.

    for some large, heterogeneous stream:
    e.g., 
    from obspy import read
    from wyrm.message.rtstream import RtStream

    st = read('../../example/uw61957912/bulk.mseed')
    rtst = RtSt(traces=st)

    st
        2610 Trace(s) in Stream:
        CC.ARAT..BHE | 2023-10-19T03:32:21.740000Z - 2023-10-19T03:34:51.720000Z | 50.0 Hz, 7500 samples
        ...
        (2608 other traces)
        ...
        UW.YPT..HHZ | 2023-10-19T03:32:44.250000Z - 2023-10-19T03:35:14.240000Z | 100.0 Hz, 15000 samples

        [Use "print(Stream.__str__(extended=True))" to print all Traces]

    rtst
        RtInstStream with 920 instrument(s) comprising 2610 traces
        CC.ARAT..BH     | 50.0 Hz,  [E] 7500 npts [N] 7500 npts [Z] 7500 npts
        ...
        (918 other instruments)
        ...
        UW.YPT..HH      | 100.0 Hz,  [E] 15000 npts [N] 15000 npts [Z] 15000 npts

        
    
    # Filtering for a single station
        %timeit st.select(station='TOUCH')
        #    1.29 ms ± 13.9 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        %timeit rtst.select(station='TOUCH')
        #    649 µs ± 19.1 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        %timeit rtst._fnmatch_rtst('*.TOUCH.*')
        #    579 µs ± 1.25 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        %timeit Stream(rtst._fnmatch('*.TOUCH.*'))
        #    131 µs ± 2.05 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

        
    # Filtering for every instrument subset:
        si_tup = []
        for _tr in st:
            si = (_tr.stats.station, _tr.stats.channel[:-1])
            if si not in si_tup:
                si_tup.append(si)

        tick = time()
        for s,i in si_tup:
            st.select(station=s, channel=i+'?')
        tock = time(); print(tock - tick)

        # 1.2123889923095703

        tick = time()
        for s,i in si_tup:
            rtst.select(station=s, instband=i)
        tock = time(); print(tock - tick)

        # 0.18765783309936523

        tick = time()
        for s,i in sn_tup:
            rtst._fnmatch(f'*.{s}.*.{i}')
        tock = time(); print(tock - tick)
        
        # 0.1803269386291504


    # Slicing collection across station and some, but not all, components
        %timeit st.select(station='TOUCH', channel='??[ZN]')
        #    1.3 ms ± 22.1 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        %timeit rtst.select(station='TOUCH', instband='??', component='[ZN]')
        #    481 µs ± 19.1 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        %timeit rtst._fnmatch_rtst('*.TOUCH.*', fnkey2='[ZN]')
        #    474 µs ± 36.4 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        %timeit Stream(rtst._fnmatch('*.TOUCH.*', fnkey2='[ZN]'))
        #    132 µs ± 3.54 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

        
    # For doing large-scale slicing (e.g., by instrument type), obspy.stream.select() 
    # tends to outperform the RtStream.select(), but not RtStream._fnmatch()

        %timeit st.select(channel='?N?')
        #    1.43 ms ± 8.5 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        %timeit rtst.select(instband='?N')
        #    125 ms ± 2.38 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        
        %timeit rtst._fnmatch_rtst('*.*.*.?N')
        #    124 ms ± 746 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

        %timeit Stream(rtst._fnmatch('*.*.*.?N'))
        #    230 µs ± 3.37 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        
    # Comparison of detailed/lazy wildcard formatting - not much difference
        %timeit rtst._fnmatch_rtst('*.*.*.?N')
            124 ms ± 746 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

        %timeit rtst._fnmatch_rtst('*.*.*.*N')
        #    122 ms ± 1.65 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

        %timeit rtst._fnmatch_rtst('*N')
        #    122 ms ± 1.4 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

    TL;DR: with mindful composition, the RtInstStream._fnmatch() method provides
           the fastest slicing compared to the *.select() wrapper methods in many cases
           for iterations across large sets of individual instruments, most of the class
           methods provided by RtInstStream result in an appreciable speed-up compared
           to obspy.core.stream.Stream.

    NOTE: Benchmarking conducted on an Apple MacBook Pro 2023 with M2 MAX chipset and 64 Gb RAM
    """
    def __init__(self, max_length=150, id_fmt='nsli', traces=None, inv=None):
        # Initialize dictionary inheritance
        super().__init__(self)
        
        # Compatability check for max_length:
        if isinstance(max_length, float):
            self.max_length = max_length
        elif isinstance(max_length, int):
            self.max_length = float(max_length)
        else:
            raise TypeError('max_length must be type int or float')
        
        if not np.isfinite(self.max_length):
            raise ValueError('max_length must be finite')
        else:
            pass

        if self.max_length < 0:
            raise ValueError('max_length must be positive')
        else:
            pass

        # Compatability checks for id_fmt
        if not isinstance(id_fmt, str):
            raise TypeError('id_fmt must be type str (options: "nsli" or "snil")')
        elif id_fmt.lower() == 'nsli':
            self.id_fmt = id_fmt.lower()
            self.id_fstr = '{net}.{sta}.{loc}.{inst}'
        elif id_fmt.lower() == 'snil':
            self.id_fmt = id_fmt.lower()
            self.id_fstr = '{sta}.{net}.{inst}.{loc}'
        else:
            raise ValueError('id_fmt must be "nsli" or "snil"')

        # (Optional) structure initialization from inventory
        if inv is None:
            pass
        elif isinstance(inv, Inventory):
            self.populate_from_inv(inv)
        else:
            raise TypeError('inv must be type Inventory or None')

        # (Optional) Data ingestion at initialization
        if traces is None:
            pass
        elif isinstance(traces, Trace):
            self.append(traces)
        elif isinstance(traces, (list, Stream)):
            if all(isinstance(tr, Trace) for tr in traces):
                for tr in traces:
                    self.append(tr)
            else:
                raise TypeError('All elements of input list or Stream "traces" must be type Trace')
        else:
            raise TypeError('traces must be type list, Trace, or Stream. For list all elements must be type Trace')

    def _repr_line(self, key):
        _l2 = self[key]
        # Display key
        rstr = f'{key:<15}'
        if len(_l2.keys()) > 0:
            # tsmax = UTCDateTime(0)
            # tsmin = UTCDateTime("2500-01-01")
            sr = [_l2[_k2].stats.sampling_rate for _k2 in _l2.keys()]
            if min(sr) == max(sr):
                rstr += f' | {sr[0]:.1f} Hz, '
            else:
                rstr += f' | {min(sr):.1f} - {max(sr):.1f} Hz, '
        for _k in _l2.keys():
            ntot = len(_l2[_k])
            # if _l2[_k].stats.starttime > 
            if np.ma.is_masked(_l2[_k].data):
                numa = sum(_l2[_k].data.mask)
                rstr += f' [{_k}] {numa:d} ({ntot:d}) npts'
            else:
                rstr += f' [{_k}] {ntot:d} npts'
        rstr += '\n'
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
        rstr = f'RtInstStream with {len(self)} instrument(s) comprising {ntr} trace(s)\n'
        for _i, _k in enumerate(self.keys()):
            if _i < 1:
                rstr += self._repr_line(_k)
            elif _i == 1 and not extended:
                rstr += f'...\n({len(self) - 2} other instruments)\n...\n'
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

    def append(self, input):
        """
        Wrapper for self.append_trace that can accept
        Trace-type and Stream-type objects as 'input'

        :: INPUT ::
        :param input: [obspy.Trace], [obspy.Stream] or child-class
                    data packet(s) to append to rtstream buffers.
        """
        if isinstance(input, Trace):
            self.append_trace(input)
        elif isinstance(input, Stream):
            if all(isinstance(_tr, Trace) for _tr in input):
                for _tr in tqdm(input, disable=len(input) < 100):
                    self.append_trace(_tr)
            else:
                raise TypeError('not all elements of Stream-type input are Trace-type')
        else:
            raise TypeError('input must be type Stream or type Trace')
        
    def append_trace(self, trace):
        """
        Append data from Trace-type object into the RtInstStream
        :: INPUT ::
        :param trace: [obspy.core.trace.Trace]
                    [obspy.realtime.rttrace.RtTrace] - Trace child
                    [wyrm.message.trace.TraceMsg] - Trace child
                    Input Trace-type object

        TODO: need to provide some work-around for data overlaps. Obspy RtTrace
                currently breaks if there's a masked array with overlap into new data.
                --> this means that the sync function 
                --> this functionality should be shifted to WindWyrm
                    when querying 
        """
        # Compatability check on trace
        if not isinstance(trace, Trace):
            raise TypeError('trace must be a Trace-type object')
        else:
            pass
        
        # If trace.id is not the empty-trace ID string, proceed
        if trace.id != '...':
            # Get datastream code components
            n,s,l,c = trace.id.split('.')
            # Split SEED channel name into Inst & Comp
            i = c[:-1]
            cc = c[-1]
            key1 = self.id_fstr.format(sta=s, net=n, loc=l, inst=i)
            key2 = cc

            # Run checks if self[key1][key2] exists

            # If key1 exists in RtInstStream...
            if key1 in self.keys():
                # If key2 exists in RtInstStream[key1]...
                if key2 in self[key1].keys():
                    # if RtInstStream[key1][key2] is RtBuffTrace, pass
                    if isinstance(self[key1][key2], RtBuffTrace):
                        pass
                    # otherwise, populate with empty RtBuffTrace
                    else:
                        self[key1][key2] = RtBuffTrace(max_length=self.max_length)
                # If key2 does not exist in RtInstStream[key1], populate
                else:
                    self[key1].update({key2: RtBuffTrace(max_length=self.max_length)})
            # If entirely new branch, populate branch
            else:
                self.update({key1: {key2: RtBuffTrace(max_length=self.max_length)}})
            # Ensure that in_trace is an obspy.core.Trace object
            in_tr = Trace(data=trace.data, header=trace.stats)

            # Finally, append data!
            self[key1][key2].append(in_tr)
        # END
                
    def populate_from_inv(self, inv):
        """
        Populate an empty RtInstStream structure from input obspy Inventory object

        :: INPUT ::
        :param inv: [obspy.core.inventory.Inventory]
                    Inventory object that has information down to the channel level.
                    i.e.,
                        inv
                            networks
                                stations
                                    channels
        """

        if not isinstance(inv, Inventory):
            raise TypeError('inv must be type Inventory (obspy.core.inventory.Inventory)')
        else:
            pass
        for net in inv.networks:
            _n = net.code
            for sta in net.stations:
                _s = sta.code
                for cha in sta.channels:
                    _l = cha.location_code
                    _c = cha.code
                    _cc = _c[-1]
                    _i = _c[:-1]
                    key1 = self.id_fstr.format(sta=_s, net=_n, loc=_l, inst=_i)
                    key2 = _cc
                    if key1 in self.keys():
                        if key2 in self[key1].keys():
                            if not isinstance(self[key1][key2], RtBuffTrace):
                                self[key1][key2] = RtBuffTrace(max_length=self.max_length)
                            else:
                                pass
                        else:
                            self[key1].update({key2: RtBuffTrace(max_length=self.max_length)})
                    else:
                        self.update({key1: {key2: RtBuffTrace(max_length=self.max_length)}})
        # END

    def copy(self):
        """
        Create a deep-copy of this RtInstStream object
        :: OUTPUT ::
        :return rtistream: [wyrm.structure.rtstream.RtInstStream]
        """
        rtistream = deepcopy(self)
        return rtistream


    def _fnmatch(self, fnkey1, fnkey2=None):
        """
        Return a list of aliased RtTrace objects from this
        source RtInstStream object using key filtering via
        fnmatch.filter() for the 1st level (instrument) keys
        and provide optional filtering for 2nd level (component)
        keys

        :: INPUTS ::
        :param fnkey1: [str] fnmatch.filter() compliant string
                        for NSLI/SNIL codes
                        e.g., 'UW.TOUCH..*' - all UW.TOUCH rttraces
                            'UW.*.?N' - all UW net accelerometer rttraces
        :param fnkey2: [str] or [None] fnmatch.filter() compliant
                        string for SEED component code characters
                        e.g., 'Z', '[ZNE]', '[Z3N1E2]'
        
        :: OUTPUT ::
        :return out_list: [list] list of aliased RtTrace objects sliced from
                        source RtInstStream
        
        NOTE: See root documentation for RtInstStream for comparison of 
              slicing speeds for this method and comparable *.select() methods
              of RtInstStream and Stream.
        
        TL;DR - this can be the fastest method in many cases
        """
        if not isinstance(fnkey1, str):
            raise TypeError('fnkey1 must be type str')
        else:
            pass

        out_list = []
        for x in fnmatch.filter(self.keys(), fnkey1):
            _out = self[x]
            if fnkey2 is not None:
                for y in fnmatch.filter(_out.keys(), fnkey2):
                    out_list.append(_out[y])
            else:
                out_list += list(_out.values())
        
        return out_list

    def _fnmatch_rtst(self, fnkey1, fnkey2=None):
        """
        Alternative version of the _fnmatch() method that returns a
        RtInstStream object, rather than a list of RtBuffTrace objects.

        See documentation on wyrm.structure.rtstream.RtInstStream._fnmatch()
        """
        if not isinstance(fnkey1, str):
            raise TypeError('fnkey1 must be type str')
        else:
            pass

        out_rtst = RtInstStream()
        for x in fnmatch.filter(self.keys(), fnkey1):
            _out = self[x]
            if fnkey2 is not None:
                for y in fnmatch.filter(_out.keys(), fnkey2):
                    out_rtst.append(_out[y])
            else:
                for _tr in _out.values():
                    out_rtst.append(_tr)
        
        return out_rtst

    def select(self, network=None, station=None, location=None, instband=None, component=None):
        """
        Provide a class method approximating the syntax of obspy.stream.select() with the NSLI / SNIL + C syntax
        built on top of the RtInstStream._fnmatch() method. All arguments accept wildcards and None values
        default to '*' values for the composed search via _fnmatch()

        :: INPUTS ::
        :param network: [str] or [None] network code(s)
        :param station: [str] or [None] station code(s)
        :param location: [str] or [None] location code(s)
        :param instband: [str] or [None] Band and Instrument codes from SEED channel naming conventions
        :param component: [str] or [None] Component code(s) from SEED channel naming conventions

        :: OUTPUT ::
        :return out_st: [obspy.core.stream.Stream] Subset stream composed of aliased RtTrace objects
        """

        if network is None:
            net = '*'
        elif isinstance(network, str):
            net = network
        else:
            raise TypeError('network must be type str or None')
        
        if station is None:
            sta = '*'
        elif isinstance(station, str):
            sta = station
        else:
            raise TypeError('station must be type str or None')
        
        if location is None:
            loc = '*'
        elif isinstance(location, str):
            loc = location
        else:
            raise TypeError('location must be type str or None')
        
        if instband is None:
            inst = '*'
        elif isinstance(instband, str):
            inst = instband
        else:
            raise TypeError('instband must be type str or None')
        
        if component is None:
            comp = '*'
        elif isinstance(component, str):
            comp = component
        else:
            raise TypeError('component must be type str or None')
        

        fnkey1 = self.id_fstr.format(net=net, sta=sta, loc=loc, inst=inst)
        fnkey2 = comp
        out_st = Stream(self._fnmatch(fnkey1, fnkey2=fnkey2))

        return out_st


    # ### Castings ###
    # def trim(self, starttime=None, endtime=None, iscopy=True, fnkey1='*', fnkey2='*', otype='trace'):
    #     """
    #     Conduct trim on RtTraces contained within this RtInstStream
    #     Default is to produce trimmed copies of data (non-in-place trim)
    #     but an option (iscopy) is provided to do in-place trimming
    #     """


    # def sync_instrument_buffer_windowing(self, refcomp='[Z3]', fill_value=None):
    #     """
    #     Using specified reference component code(s), use the rttrace.trim() method
    #     to trim/pad data within a given NSLI / SNIL keyed sub-dictionary

    #     :: INPUTS ::
    #     :param refcomp: [str] or [None]
        
    #                     [str] fnmatch compatable string that selects a single component
    #                     for every instrument sub-dictionary present in this RtInstStream.
    #                     Default '[Z3]' is highly recommended as these are codes for vertical
    #                     components

    #                     [None] pad all rttraces present to the latest endtime value available
    #                     for traces within the sub-dictionary (max_te) and back-pad data out to
    #                     max_te - self.max_length.
                        
    #                     NOTE: max endtime was chosen for this because it allows for late-arriving
    #                     packets to fill in the leading edge of rttraces

    #     :param fill_value: see obspy.core.trace.Trace.trim() for full details. Gist is that
    #                     [None] provides masked array padding for gaps or samples outside the
    #                     bounds of a given rttrace, which is the desired result for most 
    #                     processing using this class
    #     """
    #     if not isinstance(refcomp, (str, type(None)):
    #         raise TypeError('refcomp must be type str or None-type')
    #     else:
    #         pass

    #     for _k1 in self.keys():
    #         _i1 = self[_k1]
    #         if refcomp is not None:
    #             _rk = fnmatch.filter(_i1.keys(), refcomp)
    #             if len(_rk) != 1:
    #                 raise SyntaxError(f'refcomp {refcomp} produced {len(_rk)} options for {_k1}')
    #             else:
    #                 ts = self[_k1][_rk].stats.starttime
    #                 te = self[_k1][_rk].stats.endtime
    #             for _k2 in _i1.keys():
    #                 _rttr = _i1[_k2]
    #                 _rttr.trim(starttime=ts, endtime=te, pad=True, fill_value=fill_value)
    #         else:
    #             # Find maximum starttime
    #             max_ts = UTCDateTime(0)
    #             for _k2 in _i1.keys():
    #                 if _i1[_k2].stats.starttime > max_ts:
    #                     max_ts = _i1[_k2].stats.starttime
    #             # Apply trimming/padding
    #             for _k2 in _i1.keys():
    #                 _rttr = _i1[_k2]
    #                 _rttr.trim(starttime=max_te - self.max_length,
    #                            endtime=max_te,
    #                            pad=True,
    #                            fill_value=fill_value)
                


    # def get_time_bounds(self, fnkey1='*', fnkey2='*')



