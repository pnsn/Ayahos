"""
:module: wyrm.util.generate_example_msgs.py
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This script generates a set of dictionaries formatted in the PyEarthworm
    `wave` and `pick` message formats using queries to the FSDN webservice and
    ComCat. These serve as test data for handling development of python-side
    utilities in wyrm. 
    
"""
import os
import sys
from tqdm import tqdm
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Stream
from libcomcat.search import get_event_by_id
from libcomcat.dataframes import get_phase_dataframe

ROOT = os.path.join("..", "..")
sys.path.append(ROOT)
from wyrm.util import PyEW_translate as pet

# Set EVIDs
EVID = ["uw61972566"]#, "uw61736757"]
# Set padding
pads = [-60, 60]
# Connect to FDSN client
client = Client("IRIS")


def get_example_event_data(EVID=EVID, pads=pads, client=client):
    events = []
    for _evid in EVID:
        # Get event from ComCat
        event = get_event_by_id(_evid)
        # Get phases from Comcat
        phase = get_phase_dataframe(event)
        SN_list = []
        # Create holder for waveform messages
        wave_msg_list = []
        # Iterate across phases
        for _i in tqdm(range(len(phase))):
            # Pull phase Series
            _series = phase.iloc[_i, :]
            # Pull SNCL
            _n, _s, _c, _l = _series.Channel.split(".")
            if (_s, _n) not in SN_list:
                # Format pick time
                tp = UTCDateTime(_series["Arrival Time"].timestamp())
                # Get waveform(s)
                try:
                    st = client.get_waveforms(
                        station=_s,
                        network=_n,
                        channel=_c[:2]+'?',
                        location=_l,
                        starttime=tp + pads[0],
                        endtime=tp + pads[1],
                    )
                except:
                    st = Stream()
                SN_list.append((_s, _n))
                # Iterate across trace(s) in Stream
                for _tr in st:
                    # Convert trace into wave message
                    wave_msg = pet.trace_to_pyew_tracebuff2(_tr)
                    wave_msg_list.append(wave_msg)

        # Wrap up each event
        events.append({"EV": event, "PHZ": phase, "Wmsg": wave_msg_list})
    return events
