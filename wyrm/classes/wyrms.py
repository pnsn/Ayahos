"""
:module: wyrm.classes.wyrms
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This module contains class definitions stemming from the Wyrm BaseClass
    that serve as segments of a Python-side processing line for "pulsed"
    data flow that conforms to both the "heartbeat" paradigm of Earthworm
    and the "striding window" paradigm for continuous ML model prediction
    data flows.

    As such, Wyrm, and each child class have a polymorphic form of the
    class-method `pulse(self, x)` that executes a standard (series) of
    class-methods for that child class. This provides an utility of chaining
    together compatable *Wyrm objects to successively process data during a
    triggered "pulse"

:attribution:
    This module builds on the PyEarthworm (C) 2018 F. Hernandez interface
    between an Earthworm Message Transport system and Python distributed
    under an AGPL-3.0 license.

"""
import torch
import numpy as np
import pandas as pd
import seisbench.models as sbm
from obspy import UTCDateTime, Stream, Trace
from obspy.realtime import RtTrace
from wyrm.classes.pyew_msg import *
####################
### BASE CLASSES ###
####################

class Wyrm:
    """
    Base class for all *Wyrm classes in this module that are defined
    by having the y = *wyrm.pulse(x) class method.

    The Wyrm base class produces an empty
    """
    def __init__(self):
        return None

    def __repr__(self):
        msg = "~~wyrm~~\nBaseClass\n...I got no legs...\n"
        return msg

    def pulse(self, x):
        return None


class MLWyrm(Wyrm):
    """
    BaseChildClass for generalized handling of PyTorch prediction work
    and minimal handling of 
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model_to_device = False

    def __repr__(self):
        rstr = f'Device: {self.device}\n'
        rstr += f'M2D?: {self.model_to_device}\n'
        rstr += f'Model Component Order: {self.model.component_order}\n'
        rstr += f'Model Prediction Classes: {self.model.classes}\n'
        rstr += f'Model '
        rstr += f'Model Citation\n{self.model.citation}'
        return rstr

    def _send_model_to_device(self):
        self.model.to(self.device)
        self.model_to_device = True

    def _send_data_to_device(self, x):
        # If the data presented are not already in a torch.Tensor format
        # but appear that they can be, convert.
        if not isinstance(x, torch.Tensor) and isinstance(x, np.ndarray):
            x = torch.Tensor(x)
            x.to(self.device)
            return x
        # If the data are already a torch.Tensor, pass input to output
        elif isinstance(x, torch.Tensor):
            x.to(self.device)
            return x
        # For all other cases, raise TypeError
        else:
            raise TypeError

    def _run_prediction(self, x):
        y = self.model(x)
        return y
    
    def pulse(self, x):
        """
        Run a prediction on input data tensor.

        :: INPUT ::
        :param x: [torch.Tensor] or [numpy.ndarray] 
                pre-processed data with appropriate dimensionality for specified model.
                e.g., 
                for PhaseNet: x.shape = (nwind, chans, time)
                                        (nwind, 3, 3000)
                for EQTransformer: x.shape = (time, chans, nwind)
                                             (6000, 3, nwind)
        :: OUTPUT ::
        :return y: [torch.Tensor] predicted values for specified model
            e.g.
            PhaseNet: y = (nwind, [P(tP(t)), P(tS(t)), P(Noise(t))], t)
            EQTransformer: y = (t, [P(Detection(t)), P(tP(t)), P(tS(t))], nwind)
        
        """
        # Sanity check that model submitted to device
        if not self.model_to_device:
            self._send_model_to_device()
        # Send data to device if needed
        if x.device != self.device:
            self._send_data_to_device(x)
        # Run prediction
        y = self._run_prediction(x)
        # Return raw output of prediction
        return y

###############################################
### EARTHWORM <--> PYTHON INTERFACE CLASSES ###
###############################################

class WaveRingWyrm(Wyrm):
    """
    This class provides a general interface wrapping around EWModule
    connections to the WAVE RING of an Earthworm instance.

    Pulse takes an unused input (x) and returns an unordered list of
    PyEW_WaveMsg objects
    """
    def __init__(self, WAVE_RING_ID, MODULE_ID, INST_ID, HB_PERIOD, last_CR_index, debug=False):
        # Establish connection to wave ring
        self.conn = PyEW.EWModule(WAVE_RING_ID, MODULE_ID, INST_ID, HB_PERIOD, debug)
        self.conn.add_ring(WAVE_RING_ID)
        self.index = last_CR_index + 1
        self.prior_message_buffer = []
        self.max_message_iterations = int(1e6)

    def __repr__(self):
        rstr = f'WaveRingWyrm Connection: {self.conn}'
        rstr += f'Connection Index: {self.index}'
        return rstr

    def pulse(self, x):
        new_messages = []
        for _ in range(self.max_message_iterations):
            wave = self.conn.get_wave(self.index)
            # Check that wave is not empty
            if wave != {}:
                # If not empty, do sanity checks that it's a wave message and convert to PyEW_WaveMsg object
                # Check that wave is not in new_messages
                if wave not in new_messages:
                    # If wave did not slip through from previous messages
                    if wave not in self.prior_message_buffer:
                        # Append to new messages
                        new_messages.append(wave)
            # if the WAVE ring runs out of new messages, truncate for loop
            elif wave == {}:
                break
        self.prior_message_buffer = new_messages
        y = new_messages
        return y
        




        waves = self.conn.get_waves(self.index)
        if isinstance(waves, dict):
            waves = [waves]
        elif isinstance(waves, list) and len(waves) == 0:
            y = []
        else:
            raise TypeError
        if isinstance(waves,list):    
            y = [PyEW_WaveMsg(_wave) for _waves in waves]
            return y        
    

class PickRingWyrm(Wyrm):
    """
    This class provides a general interface wrapping around EWModule
    connections to the PICK RING of an Earthworm instance.

    Pulse takes an unused input (x) and returns an unordered list of
    PyEW_PickMsg objects
    """
    def __init__(self, PICK_RING_ID, MODULE_ID, INST_ID, HB_PERIOD, last_CR_index, debug=False):
        # Establish connection to PICK ring
        self.conn = EWModule(PICK_RING_ID, MODULE_ID, INST_ID, HB_PERIOD, debug)
        self.conn.add_ring(PICK_RING_ID)
        self.index = last_CR_index + 1
        
    def __repr__(self):
        rstr = f'PickRingWyrm Connection: {self.conn}'
        rstr += f'Connection Index: {self.index}'
        return rstr

    def pulse(self, x):
        waves = self.conn.get_msg(self.index)
        if isinstance(waves, dict):
            waves = [waves]
        elif isinstance(waves, list) and len(waves) == 0:
            y = []
        else:
            raise TypeError
        if isinstance(waves,list):    
            y = [PyEW_WaveMsg(_wave) for _waves in waves]
            return y   


############################
### ORGANIZATION CLASSES ###
############################
class EarWyrm(Wyrm):
    """
    This Wyrm listens for a specific Station/Network/Channel/Location
    (SNCL) combination in offered of PyEarthworm `wave` objects. 
    Matching messages are aggregated and converted into an ObsPy Trace
    by the EarWyrm.pulse(x) method 
    """
    def __init__(self, station, network, channel, location):
        self.station = station
        self.network = network
        self.channel = channel
        self.location = location
        self._sncl_dict = dict(
            zip(['station', 'network', 'channel', 'location'],
                [station, network, channel, location]))

    def __repr__(self):
        fstr = '~~EarWyrm~~\nListening For: '
        fstr += f'{".".join(list(self._sncl_dict.values()))}'
        return fstr

    def pulse(self, x):
        """
        This pulse(x) takes an array-like set of PyEarthworm
        `wave` messages and returns a list that has matching
        SNCL values for this particular EarWyrm

        :: INPUT :: 
        :param x: [list] List of PyEarthworm `wave` objects

        :: OUTPUT ::
        :param y: [list] List of PyEarthworm `wave` objects
                  with matching SNCL labels
        """
        # Create a holder stream for trace elements
        waves = []
        # Iterate across presented messages
        for _x in x:
            # Use the `all` iteration operator to match 
            match = all(_x[_k] == _v for _k, _v in self._sncl_dict)
            # If SNCL is a perfect match, proceed
            if match:
                waves.append(_x)
        y = waves
        return y


class BookWyrm(Wyrm):
    """
    This class acts as in indexer and sorter for data arrays
    keyed by SNCL entries
    """

    def __init__(self, msg_class=PyEW_WaveMsg):
        self.msg_type = msg_class
        self.SNCL_dataframe = pd.DataFrame(columns=['Station','Network','Location','Channel'])

    def reset_SNCL_dataframe(self):
        self.SNCL_dataframe = pd.DataFrame(columns=['Station','Network','Location','Channel'])

    def append_to_SNCL_dataframe(self, x):
        for _i, _x in enumerate(x):
            if isinstance(_x, self.msg_type):
                self.SNCL_dataframe = pd.concat([self.SNCL_dataframe, pd.DataFrame(_x.SNCL_dict), index=[_i]],axis=1,ignore_index=False)

    def pulse(self, x):
        """
        :param x: [list] unsorted list of PyEW_Msg objects

        :return 
        """


class TubeWyrm(Wyrm):
    """
    Contain a linear sequence of Wyrm objects that must have 
    compatable hand-offs in data/data-type for inputs and 
    outputs of their .pulse() class methods. A TubeWyrm provides 
    a pulse(x) method that accepts an arbitrary input (x) that 
    must comply with self.wyrm_can[0].pulse(x) and will return
    the output of the sequence of wyrm.pulse(x)'s

    :: ATTRIBUTES ::
    :attrib index: [int-like] Index number for this WyrmCan
    :attrib wyrm_list: [list] List of Wyrm* objects that have
                compatable sequential input/output data from
                their versions of the .pulse() class method
    :attrib cfg: [dict] Dictionary to hold configuration information
                NOTE: Presently not used

    :: CLASS METHODS ::
    :method __init__:
    :method pulse:
    """
    def __init__(self, index, wyrm_list, cfg=None):
        """
        Initialize a TubeWyrm object with the following
        input parameters
        :: INPUTS ::
        :param index: [int-like] Index number for this WyrmCan TODO: OBSOLITE
        :param wyrm_list: [list] List of Wyrm* objects that have
                    compatable sequential input/output data from
                    their versions of the .pulse() class method
        :param cfg: [dict] Dictionary to hold configuration information
                    NOTE: Presently not used

        :: OUTPUT ::
        None
        """
        # Type handling for index
        if isinstance(index, (int, np.int32, np.int64)):
            self.index = index
        elif isinstance(index, (float, np.float32, np.float64)):
            self.index = int(index)
        else:
            raise TypeError
        
        # Type checking for all members of Wyrm
        if isinstance(wyrm_list, list):
            # Check that all items in wyrm_list are a child of Wyrm
            match = all(isinstance(_wyrm, Wyrm) for _wyrm in wyrm_list)
            if match:
                self.wyrm_list = wyrm_list
            # Raise TypeError otherwise
            else:
                raise TypeError
        # Handle the case where the WyrmCan is given a single Wyrm object
        elif isinstance(wyrm_list, Wyrm):
            self.wyrm_list = [wyrm_list]
        # Otherwise, raise TypeError
        else:
            raise TypeError

        self.cfg = cfg
        return None

    def __repr__(self):
        fmsg = 'TubeWyrm with\n'
        fmsg += f'Index: {self.index}\n'
        fmsg += 'WyrmList:\n'
        for _wyrm in self.wyrm_list:
            fmsg += f'  {type(_wyrm)}\n'
        fmsg += f'cfg: {self.cfg}'
        return fmsg

    def pulse(self, x):
        """
        Run .pulse() for each member of self.wyrm_list in sequence,
        passing the output of the i_th *wyrm.pulse(x) = y to the input
        to the input for the i+1_th *wyrm.pulse(x = y).

        :: INPUT ::
        :param x: Input for *wyrm.pulse(x) for the first member of
                  self.wyrm_list
        :: OUTPUT ::
        :return y: [dict] output of the last *wyrm.pulse(x) in 
                  self.wyrm_list with an associated key self.index
                  from this WyrmCan object
        """
        for _wyrm in self.wyrm_list:
            x = _wyrm.pulse(x)
        y = {self.index: x}
        return y



###############################
### DATA PROCESSING CLASSES ###
###############################

class BuffWyrm(Wyrm):
    """
    This Wyrm hosts ordered ObsPy RealTime Trace (RtTrace) objects that are
    populated from incoming PyEarthworm `wave` objects, serves as a waveform 
    data buffer between pulses and provides access to RtTrace processing steps 
    via the `eval` on buffered data if they meet certain time bounds
    """
    
    def __init__(self, rtstream=None, rttrace_processing=['.'])
        if rtstream is None
            self.rtstream = Stream()

        elif isinstance(rtstrea, Stream):
            match = all(isinstance(_tr, RtTrace) for _tr in rtstream)
            if match:
                self.rtstream = rtstream
        else:
            raise TypeError

    def pulse(self, x):
        # Handle x = empty list
        if x == []:
            match = True
        # Handle instance of a slngle PyEW_Wave object as input for x
        if isinstance(x, PyEW_Wave):
            x = [x]
            match = True
        # Check entries in list of probable PyEW_Wave objects
        elif isinstance(x, (list, np.array)):
            match = all(isinstance(_x, PyEW_Wave) for _x in x)
        # Raise error in all other cases
        else:
            match = False
            raise TypeError

        if match:
            for _x in x:
                # Convert PyEW_Wave into Obspy Trace
                _tr = pew2otr(_x)
                # Append new trace to real-time trace buffer

        # Clean outdated data out of the buffer
        
        time_match = all(_tr.stats.endtime >= self.next_window_end for _tr in self.)
            


class WindowWyrm(Wyrm):

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

