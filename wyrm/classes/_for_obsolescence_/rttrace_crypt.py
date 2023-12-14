"""
:module: wyrm.classes.rttrace_crypt
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This module contains an extension of the Obspy Real Time Trace
    (obspy.realtime.rttrace.RTTrace) class, providing the additional
    functionality of adding a "crypt" for a specified length of data

"""


class RTTraceTorch(RTTrace):

    def __init__(self, main_length=60, crypt_length=60):
    

"""
Ehhhhh, this can probably be done without coding a child class
of RtTrace, can probably just use an attribute in _Hydra and 
(grand) child classes that keeps track of the last window indices
for striding predictions (which are going to be necessary anyway
for reconstructing windows into streams) and then have the maximum
length of each data stream


"""