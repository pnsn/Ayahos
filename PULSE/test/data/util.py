from pathlib import Path
from obspy import read, read_inventory, read_events, Trace
from obspy.core.trace import Stats
import numpy as np

def load_townsend_example():
    """Load a subset of waveforms, event metadata, and station metadata
    from a diverse subset of instrument types and source-receiver 
    distances observing the M4.3 earthquake near Port Townsend, WA
    on 9. October 2023.

    Contains 27 traces from 11 instruments at 6 stations
    
    Station UW.HANS  - 6 component BB + SMA, near source region
    Station UW.NATEM - 6 component BB + SMA, near source region
    Station UO.BUCK  - 6 component BB + SMA, long source-receiver offset
    Station UW.MCW   - 4 component SP + SMA, near source region
    Station UW.PUPY  - 4 component SP + SMA, intermediate source-recevier offset 
    Station UW.PAT2  - 1 component SP, long source-receiver offset

    BB - Broadband Instrument
    SP - Short Period Instrument (vertical component only)
    SMA - Strong Motion Accelerometer Instrument

    :return: 
     - **st** (*obspy.core.stream.Stream*) -- waveform data
     - **inv** (*obspy.core.inventory.Inventory*) -- station metadata to response level
     - **cat** (*obspy.core.event.Catalog*) -- event metadata
    """    
    data_path = Path(__file__).parent.parent
    data = data_path / 'files' / 'uw61965081_PortTownsend_M4.3.mseed'
    stas = data_path / 'files' / 'uw61965081_Example_Stations.xml'
    even = data_path / 'files' / 'uw61965081_Origin_Quake.xml'
    # data = os.path.join('PULSE','test','files','uw61965081_PortTownsend_M4.3.mseed')
    st = read(data)
    inv = read_inventory(stas)
    cat = read_events(even)
    return st, inv, cat

def load_seattle_example():
    """Load waveforms from the M2.1 earthquake in North Seattle, WA
    (uw61956601) on 2. September 2023.

        evid: uw61956601
          t0: 2023-09-02T18:09:50.50000
         lat: 47.687833
         lon: -122.284
         dep: 29.310

    Contains 395 traces from 133 instruments at 109 stations
    located within 60 km of the epicenter for channels ?[HN]?
    and locations '0?' and ''. Traces start 5 seconds after
    the origin time and conclude 25 seconds after the origin
    time.

    :return: 
     - **st** (*obspy.core.stream.Stream*) -- waveform data
     - **inv** (*obspy.core.inventory.Inventory*) -- station metadata to response level
     - **cat** (*obspy.core.event.Catalog*) -- event metadata
    """    
    data_path = Path(__file__).parent.parent
    data = data_path / 'files' / 'uw61956601_Seattle_M2.1.mseed'
    stas = data_path / 'files' / 'uw61956601_Stations.txt'
    even = data_path / 'files' / 'uw61956601_Origin_Quake.xml'
    # data = os.path.join('PULSE','test','files','uw61965081_PortTownsend_M4.3.mseed')
    st = read(data)
    inv = read_inventory(stas)
    cat = read_events(even)
    return st, inv, cat

def load_logo_vector():
    data = [0,0,0,2,4,6,4,2,0,-2,-4,-6,-4,-2,0,-1,-2,-1,0,-1,-2,-1,0,0,1,0,0,-1,0,0]
    return np.array(data, dtype=np.float64)

def load_logo_trace():
    tr = Trace(data=load_logo_vector(),
               header={'network':'UW',
                       'station':'PULSE',
                       'location':'LO',
                       'channel':'GO1'})
    return tr.copy()

def make_gappy_trace(tr,fill_value=-999, frac1=0.25, frac2=0.5):
    ii = int(round(frac1*tr.stats.npts))
    jj = int(round(frac2*tr.stats.npts))
    mtr = tr.copy()
    mtr.data = np.ma.MaskedArray(data=mtr.data,
                                 mask=[False]*tr.stats.npts,
                                 fill_value=fill_value)
    mtr.data.mask[ii:jj] = True
    return mtr


def assert_common_trace(tracelike0, tracelike1):
    # Check types
    assert isinstance(tracelike0, Trace)
    assert isinstance(tracelike1, Trace)
    # Check data
    np.testing.assert_array_equal(tracelike0.data, tracelike1.data)
    # Check dtype
    assert tracelike0.data.dtype == tracelike1.data.dtype
    # Check stats
    for _k in Stats.defaults.keys():
        assert tracelike0.stats[_k] == tracelike1.stats[_k]


def assert_common_trace_set(traceset0, traceset1):
    # Test that both have __iter__
    assert hasattr(traceset0, '__iter__')
    assert hasattr(traceset1, '__iter__')
