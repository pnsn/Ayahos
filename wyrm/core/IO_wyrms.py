"""
:module: wyrm.core.unit_wyrms
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This module contains class definitions stemming from the Wyrm BaseClass
    that serve as segments of a Python-side processing line for "pulsed"
    data flow 

:attribution:
    This module builds on the PyEarthworm (C) 2018 F. Hernandez interface
    between an Earthworm Message Transport system and Python distributed
    under an AGPL-3.0 license.

"""
import PyEW
from collections import deque
from wyrm.core.base_wyrms import Wyrm, RingWyrm
import pandas as pd
import seisbench.models as sbm
from obspy import UTCDateTime, Stream, Trace
from obspy.realtime import RtTrace
from wyrm.core.pyew_msg import WaveMsg

##################
### RING WYRMS ###
##################

class Wave2PyWyrm(RingWyrm):
    """
    Child class of RingWyrm, adding a pulse(x) method
    to complete the Wyrm base definition. Also adds a maximum
    iteration attribute to limit the number of wave messages
    that can be pulled in a single Wave2PyWyrm.pulse(x) call.
    """
    def __init__(
            self,
            module,
            conn_index,
            stations='*',
            networks='*',
            channels='*',
            locations='*',
            max_iter=int(1e5),
            max_staleness=300):
        """
        Initialize a Wave2PyWyrm object
        :: INPUTS ::
        :param module: [PyEW.EWModule] pre-initialized module
        :param conn_index: [int] index of pre-established module
                        connection to an EW WAVE RING
        :param stations: [str] or [list] station code string(s)
        :param networks: [str] or [list] network code string(s)
        :param channels: [str] or [list] channel code string(s)
        :param locations: [str] or [list] location code string(s)
        :param max_iter: [int] (default 10000) maximum number of
                        messages to receive for a single pulse(x)
                        command
        :param max_staleness: [int] (default 300) maximum number of
                        pulses an given SNCL keyed entry in queue_dict
                        can go without receiving new data.

        :: ATTRIBUTES ::
        -- Inherited from RingWyrm --
        :attrib module: [PyEW.EWModule] connected EWModule
        :attrib conn_index: [int] connection index

        -- New Attributes --
        :attrib queue_dict: [dict] SNCL keyed dict with tuple values:
                        
                        {'S.N.C.L': (deque, staleness_index)}
                        e.g. 
                        {'GNW.UW.BHN.--': (deque([WaveMsg]),0)}

                        with the dequeue adding newest entries on 
                        the left end of the deque. Subsequent
                        interactions with this attribute should
                        pop() entries to pull the oldest members

                        staleness_index starts at 0 and increases
                        for each pulse where a given SNCL code
                        does not appear. This serves as a mechanism
                        for clearing out outdated data that may arise
                        from a station outage and help prevent ingesting
                        large data gaps in subsequent

        """
        # Use RingWyrm baseclass initialization
        super().__init__(module, conn_index)
        # self.module
        # self.
        # Run compatability checks & formatting on stations
        if isinstance(stations, (str, int)):
            self.stations = str(stations)
        elif isinstance(stations, (list, deque)):
            if all(isinstance(_sta, (str, int)) for _sta in stations):
                self.stations = [str(_sta) for _sta in stations]
        else:
            raise TypeError
        
        # Run compatability checks & formatting on networks
        if isinstance(networks, (str, int)):
            self.networks = str(networks)
        elif isinstance(networks, (list, deque)):
            if all(isinstance(_net, (str, int)) for _net in networks):
                self.networks = [str(_net) for _net in networks]
        else:
            raise TypeError
        
        # Run compatability checks & formatting on channels
        if isinstance(channels, (str, int)):
            self.channels = str(channels)
        elif isinstance(channels, (list, deque)):
            if all(isinstance(_cha, (str, int)) for _cha in channels):
                self.channels = [str(_cha) for _cha in channels]
        else:
            raise TypeError
        
        # Run compatability checks & formatting on locations
        if isinstance(locations, (str, int)):
            self.locations = str(locations)
        elif isinstance(locations, (list, deque)):
            if all(isinstance(_loc, (str, int)) for _loc in locations):
                self.locations = [str(_loc) for _loc in locations]
        else:
            raise TypeError
        
        # Compatability check for max_iter
        try:
            self._max_iter = int(max_iter)
        except TypeError:
            print('max_iter must be int-like!')
            raise TypeError
        
        # Compatability check for max_staleness
        try:
            self._max_staleness = int(max_staleness)
        except TypeError:
            print('max_staleness must be positive and int-like')
            raise TypeError
        

    def __repr__(self):
        """
        Command line representation of class object
        """
        rstr = super().__repr__()
        rstr += f'Max Iter: {self._max_iter}\n'
        rstr += f'Max Stale: {self._max_staleness}\n'
        rstr += f'STA Filt: {self.stations}\n'
        rstr += f'NET Filt: {self.networks}\n'
        rstr += f'CHA Filt: {self.channels}\n'
        rstr += f'LOC Filt: {self.locations}\n'
        rstr += 
        return rstr
    
    def _get_wave_from_ring(self):
        """
        Fetch the next available message from the WAVE ring
        connection and check if it is on the approved list
        defined by SNCL information

        :: OUTPUT ::
        :return msg: [wyrm.core.pyew_msg.WaveMsg] 
                            - if a wave message is received and passes filtering
                     [False] 
                            - if an empty wave message is received
                     [True]
                            - if a wave message is received and fails filtering

        """
        wave = self.module.get_wave(self.conn_index)
        # If wave message is empty, return False
        if wave == {}:
            msg = False
        # If the wave has information, check if it's on the list
        else:
            sta_bool, net_bool, cha_bool, loc_bool = False, False, False, False
            # Do sniff-test that 
            if all(_k in wave.keys() for _k in ['station','network','channel','locaiton'])
                # Station Check
                if isinstance(self.stations, list):
                    if wave['station'] in self.stations:
                        sta_bool = True
                elif isinstance(self.stations, str):
                    fnsta = fnmatch.filter([wave['station']], self.stations)
                    if len(fnsta) == 1:
                        sta_bool = True

                # Network Check
                if isinstance(self.networks, list):
                    if wave['network'] in self.networks:
                        net_bool = True
                elif isinstance(self.networks, str):
                    fnnet = fnmatch.filter([wave['network']], self.networks)
                    if len(fnnet) == 1:
                        net_bool = True

                # Channel Check
                if isinstance(self.channels, list):
                    if wave['channel'] in self.channels:
                        cha_bool = True
                elif isinstance(self.channels, str):
                    fncha = fnmatch.filter([wave['channel']], self.channels)
                    if len(fncha) == 1:
                        cha_bool = True

                # Location Check
                if isinstance(self.locations, list):
                    if wave['location'] in self.locations:
                        loc_bool = True
                elif isinstance(self.locations, str):
                    fnloc = fnmatch.filter([wave['location']], self.locations)
                    if len(fnloc) == 1:
                        loc_bool = True
            else:
                print('Received something that does not look like a wave message')
                raise TypeError
            
            # If all fields match filtering criteria
            if all([sta_bool, net_bool, cha_bool, loc_bool]):
                # Convert message from type dict to type WaveMsg
                msg = WaveMsg(wave)
            # If filtering criteria fails, but there was a valid message
            else:
                msg = True            
            return msg

    def _add_msg_to_queue_dict(self, wavemsg):
        """
        Add a WaveMsg to the queue_dict, creating a new {code: WaveMsg}
        entry if the WaveMsg SNCL code is not in the queue_dict keys,
        and left-appending the WaveMsg to an active deque keyed to
        code if it already exists. 
        """

        # If the SNCL code of the WaveMsg is new, create a new dict entry
        if wavemsg.code not in self.queue_dict.keys():
            new_member = {wavemsg.code: (deque([wavemsg]), 0)}
            self.queue_dict.update(new_member)
            # And return True state
            return wavemsg.code
        # If the SNCL code of the WaveMsg is already known
        elif wavemsg.code in self.queue_dict.keys():
            # Left append the message to the existing queue
            new_message = deque([wavemsg])
            self.queue_dict[wavemsg.code][0].appendleft(new_message)
            # reset "staleness index"
            self.queue_dict[wavemsg.code][1] = 0
            return wavemsg.code
        # If something unexpected happens, raise KeyError
        else:
            print('Something went wrong with matching keys')
            raise KeyError

    def _update_staleness(self,updated_codes=[]):
        """
        Increase the "staleness index" of each SNCL keyed
        entry in self.queue_dict by 1 that is not present
        in the provided updated_codes list
        """
        # For each _k(ey) in queue_dict
        for _k in self.queue_dict.keys():
            # If the key is not in the updated_codes list
            if _k not in updated_codes:
                # If max staleness is met
                if self.queue_dict[_k][1] >= self._max_staleness:
                    # Determine if there are any data in the queue
                    nele = len(self.queue_dict[_k][0])
                    # If there are data, pop the last entry in the queue
                    if nele > 0:
                        self.queue_dict[_k][0].pop();
                    # If there are no data in the queue, pop the entire SNCL keyed entry
                    else:
                        self.queue_dict.pop(_k);
                # Otherwise increase that entry's staleness index by 1
                else:
                    self.queue_dict[_k][1] += 1
        

    def pulse(self, x=None):
        """
        Iterate for some large number (max_iter) pulling
        wave messages from the specified WAVE RING connection,
        appending to a list if the messages are unique, and stopping
        the iteration at the first instance of an empty dictionary
            i.e., wave = {}. 
        Not to be confused with a dataless wave message 
            i.e. wave['data'] =  np.array([]))
        
        :: INPUT ::
        :param x: [NoneType] or [int] Connection index
                [NoneType] is default, uses self.conn_index for the connection index
                [int] superceds self.conn_index.
        
        :: OUTPUT ::
        :return y: [list] list of 
        """
        # Provide option for passing an alternative index
        if x is not None and isinstance(x, int):
            idx = x
        else:
            idx = self.conn_index
        # Create holder for codes of updated SNCL queues
        updated_codes = []
        for _ in range(self._max_iter):
            wave = self._get_wave_from_ring()
            # If the wave message is indeed a WaveMsg
            if isinstance(wave, WaveMsg):
                # Append valid message to queue_dict
                refreshed_code = self._add_msg_to_queue_dict(wave)
                # If code was not already in 
                if refreshed_code  not in updated_codes:
                    updated_codes.append(refreshed_code)
            
            # If the wave message is carrying a continue/break Bool code
            elif isinstance(wave, bool):
                if wave:
                    pass
                # If wave is False, this signals an empty message. 
                elif not wave:
                    break
            else:
                raise RuntimeError
        
        # Increase staleness index and clear out overly stale data
        self._update_staleness(updated_codes)

        # Pass self.queue_dict as output
        y = self.queue_dict
        return y


class Py2WaveWyrm(RingWyrm):
    """
    Child class of RingWyrm, adding a pulse(x) method
    to complete the Wyrm base definition. Also adds a maximum
    message buffer (_max_msg_buff) attribute to limit the number 
    of previously sent messages buffered in this object for the purpose
    of preventing repeat message transmission.
    """

    def __init__(self, module, conn_index, max_msg_buff=1000):
        """
        Initialize a Py2WaveWyrm object
        :: INPUTS ::
        :param module: [PyEW.EWModule] Connected module
        :param conn_index: [int] Connection index for target WAVE RING
        :param max_msg_buff: [int-like] Maximum number of messages
                        that are buffered in object memory before purging
        """
        super().__init__(module, conn_index)
        self._max_msg_buff = max_msg_buff

    def __repr__(self):
        rstr = super().__repr__()
        rstr += f'Max Buffer: {self._max_msg_buff}\n'
        return rstr

    def pulse(self, x):
        """
        Submit data from PyEW_WaveMsg objects contained in `x`
        to the specified WAVE RING as tracebuff2 messages
        :: INPUTS ::
        :param x: [list] of PyEW_WaveMsg objects

        :: OUTPUT ::
        :param y: [int] count of successfully sent messages

        """
        send_count = 0
        # If working with a single message, ensure it's a list
        if isinstance(x, PyEW_WaveMsg):
            x = [x]
        # Iterate across list of input message(s)
        for _msg in x:
            # Ensure the message is not a duplicate of a recent one
            if isinstance(_msg, PyEW_WaveMsg) and _msg not in self.past_msg_queue:
                # SEND MESSAGE TO RING
                try:
                    self.module.put_wave(_msg.to_tracebuff2(), self.conn_index)
                    send_count += 1
                # TODO: Clean this dev choice up...
                except:
                    continue
                # Append message to to dequeue on left
                self.past_msg_queue.appendleft(_msg)
                # If we've surpassed the buffer allotment, pop oldest msg at right
                if len(self.past_msg_queue)+ 1 < self._max_msg_buff:
                    self.past_msg_queue.pop()
        # output number of sent messages
        y = int(send_count)
        return y

class Pick2PyWyrm(RingWyrm):


class Py2PickWyrm(RingWyrm):




##################
### DISK WYRMS ###
##################

class DiskWyrm(Wyrm):
    def __init__(self, mseed_file_list):
        self.to_read = deque(mseed_file_list)
        self.current = deque([])
        self.processed = deque([])

    def _read_mseed_from_disk(self, file):
        st = read(file)
        return st
    
    def


##########################################
### ML PROCESSING (GRAND)CHILD CLASSES ###
##########################################

class StreamMLWyrm(MLWyrm):
    """
    ML Prediction module for models where the input (data) and output (pred) arrays
    consist of windowed time-series with associated metadata
    """
    def __init__(self, model, device, ml_input_shape, window_axis, sample_axis, channel_axis):
        super().__init__(model, device)
        self.ml_input_shape = ml_input_shape
        self.window_axis = window_axis
        self.sample_axis = sample_axis
        self.channel_axis = channel_axis
    
    def __repr__(self):
        rstr = f'Input Dimensions: {self.ml_input_shape}\n'
        rstr += f'Window Axis: {self.window_axis}\nSample Axis: {self.sample_axis}\nChannel Axis: {self.channel_axis}\n'
        rstr += super().__repr__()

        return rstr
    
    def _preprocess_data(self, x, )


#class ELEPWyrm(MLWyrm):
# Placeholder for later development using the ensemble picking approach from ELEP (Yuan et al., 2023)

#class OctoWyrm(Wyrm):
# Placeholder for later development of a Wyrm that wraps PyOcto (Münchmeyer et al., in review)

