"""
:module: wyrm.classes.hydra
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
              ____
             /  ' \ 
         ____|  /vv-<_
      _/__'   \ \/  ' \ 
     / XX_>,  | /  /vv-<
    / /____/      /
    \____________/
    
:purpose:
    This module contains the definition of the _Hydra class and child
    classes used to coordinate multi-channel connections to EarthWorm 
    rings via PyEarthworm, provide pre-/post-processing utilities 
    for passing data arrays to/from PyTorch models, and streaming 
    operation of PyTorch predictions on pulsed data.
    
    The following classes are provided:

    _Hydra
        - Abstraction of Ring2Buff & ring2obspy (see attributions) that 
          holds connections for up to 3 channels and has a main waveform 
          buffer and a "tail" buffer that preserves trailing data segments
          of each heartbeat to facilitate data striding, that becomes the "head"
          in the next heartbeat. This functionally builds on the 
          obspy.streaming.rttrace.RtTrace
            Hence Hydra: a snake eating it's own tail.
        - This base class strictly pulls data from the WAVE ring into ObsPy compliant
          formats. Functionalities for passing information back to other 
        - This class and offers options
          for manipulating 
    
    _TorchHydra
        - Child class of _Hydra - TODO: Probably should just fold this into _Hydra
          and cut out the multi-generational class definitions...

    StreamHydra
        - (Grand)Child class of _Hydra that provides functionalities for sending ML predicitons
    
    PickHydra
        - Child class of _Hydra that adds  

:attribution:
    This class and child classes are largely based on the Ring2Ring 
    and Ring2Buff class examples and the ring2obspy example from the 
    PyEarthorm package by F.J. Hernandez Ramirez distributed under an 
    AGPL-3.0 license. 
    
    We copy and adapt large segments of these examples to minimize
    dependencies to the core class from PyEarthworm `PyEW.EWModule`
    
    If you find the core functionalities of this module useful, please
    cite the oritinal PyEarthworm project consistent with their AGPL-3.0
    license.

    If you use this module directly, please cite both the original PyEarthworm
    project and this repository.

    For original source code, see:
    https://github.com/Boritech-Solutions/PyEarthworm
        PyEarthworm/examples/ring2ring/EWMod.py
        PyEarthworm/examples/ring2buff/EWMod.py
        PyEarthworm/examples/ring2obspy/ewobspy.py
"""
import os
import sys
import time
import logging
import obspy
import torch
import seisbench.models as sbm
from PyEW import EWModule
from threading import Thread
ROOT = os.path.join('..','..')
sys.path.append(ROOT)
import wyrm.util.PyEW_translate as pet
from wyrm.classes.rttrace_crypt import RTTraceTorch

logger = logging.getLogger(__name__)

class _Hydra:
    """
    This base class provides a 1-3 channel connection to a WAVE ring and waveform buffering
    utilities, but it DOES NOT provide methods data processing or writing buffered/processed
    data to another ring. 

    As such, the class itself is somewhat useless, however, it provides the template for child
    classes (see below) that do offer these functionalities

    :: INITIALIZATION INPUTS ::
    :param HBsec: [float] Heart Beat rate in seconds 
                    NOTE: for striding ML analyses, this is the stride length
    :param BuffSec: [float] Buffer maximum length in seconds
                    NOTE: for ML analyses, this should be VERY close (within 1-2 samples)
                        to the window length for the ML input vector plus the Heart Beat rate
                            e.g., PhaseNet --> ~30 sec (@ 100 Hz) + HBsec
                            e.g., EQTransformer --> ~60 sec (@ 100 Hz) + HBsec
                    TODO: Need to deal with the fact that RtTrace erases the oldest end
                    of data as soon as the new data are appended. Need to 
    :param RING_ID: [int] Ring Identifier for this process
    :param MOD_ID: [int] Module Identifier for this process

    :param RING_IN: [int] Input Ring Identifier for the WAVE ring 
                    NOTE: Default 1000
    :param RING_OUT: [int] Output Ring Identifier (not used in base class)
    :param Z_INST_IND: [int] Instrument Identifier for the Z component of target seismometer
                    NOTE: Default is None
    :param N_INST_IND: [int] Instrument Identifier for the North-ish component of target seismometer
                    NOTE: Default is None
    :param E_INST_IND: [int] Instrument Identifier for the East-ish component of target seismometer
                    NOTE: Default is None
    :param debug: [bool] Should EWModule(s) be run in debug mode?

    :: PRIVATE ATTRIBUTES ::
    :attrib _station: [str] or [None]
    :attrib _network: [str] or [None]
    :attrib _location: [str] or [None]

    :attrib _Zchan: [str] or [None]
    :attrib _Ztrace: [obspy.realtime.RtTrace] or [None]
    :attrib _Zmodule: [PyEW.EWModule]

    :attrib _Nchan: [str] or [None]
    :attrib _Ntrace: [obspy.realtime.RtTrace] or [None] 
    :attrib _Nmodule: [PyEW.EWModule]

    :attrib _Echan: [str] or [None]
    :attrib _Etrace: [obspy.realtime.RtTrace] or [None]    
    :attrib _Emodule: [PyEW.EWModule]

    :attrib _minutes:

    """
    
    def __init__(self, HBsec, BuffSec, RING_ID, MOD_ID, RING_IN=1000, RING_OUT=None, 
                 Z_INST_ID=None, N_INST_ID=None, E_INST_ID=None, debug=False):
        # SNCL - initially blank. Becomes populated with runtime
        self._station = None
        self._network = None
        self._Zchan = None
        self._Ztrace = obspy.realtime.RtTrace(max_length=BuffSec)
        self._Nchan = None
        self._Ntrace = obspy.realtime.RtTrace(max_length=BuffSec)
        self._Echan = None
        self._Etrace = obspy.realtime.RtTrace(max_length=BuffSec)
        self._location = None
        # Create a thread for `self`
        self._thread = Thread(target=self.run)
        # Start EW Module for Vertical channel INST_ID
        self._Zmodule = EWModule(RING_ID, MOD_ID, Z_INST_ID, HBsec, debug)
        # Add input ring as Ring 0 and output ring as Ring 1
        self._Zmodule.add_ring(RING_IN)

        # If a North channel INST_ID is provided, start EW Module
        if N_INST_ID != Z_INST_ID and N_INST_ID is not None:
            self._Nmodule = EWModule(RING_ID, MOD_ID, N_INST_ID, HBsec, debug)
            self._Nmodule.add_ring(RING_IN)
        else:
            self._Nmodule = False
        
        # If a North channel INST_ID is provided, start EW Module
        if E_INST_ID != Z_INST_ID and E_INST_ID is not None:
            self._Emodule = EWModule(RING_ID, MOD_ID, E_INST_ID, HBsec, debug)
            self._Emodule.add_ring(RING_IN)
        else:
            self._Emodule = False

        # Buffer
        self._minutes = BuffSec/60.
        self.wave_buffer = {}

        self.runs = True
        self.debug = debug

    def ingest_waves(self):
        # Get Z-channel wave from 
        try:
            Zwave = self._Zmodule.get_wave(0)
        except:
            Zwave = {}
        if Zwave != {}:
            # Convert into a trace
            trZ = pet.pyew_tracebuff2_to_trace(Zwave)
            # Add current Z trace to rttrace
            self._Ztrace.append(trZ)

        # For initial step where data are received
        if self.Zchan is None and Zwave != {}:
            self.station = Zwave['station']
            self.network = Zwave['network']
            self.Zchan = Zwave['channel']
            self.location = Zwave['location']
        # If there is a mismatch in the channel kill process gracefully
        elif self.Zchan != Zwave['channel']:
            self.runs = False
        
        if isinstance(self._Nmodule, EWModule):
            try: 
                Nwave = self._Nmodule.get_wave(0)
            except:
                Nwave = pet.empty_pyew_tracebuff2()
            if Nwave != {}:
                trN = pet.pyew_tracebuff2_to_trace(Nwave)
                self._Ntrace.append(trN)
    
        if isinstance(self._Emodule, EWModule):
            try: 
                Ewave = self._Emodule.get_wave(0)
            except:
                Ewave = pet.empty_pyew_tracebuff2()
            if Ewave != {}:
                trE = pet.pyew_tracebuff2_to_trace(Ewave)
                self._Etrace.append(trE)


class _TorchHydra(_Hydra):
    """
    Child class of _Hydra that provides striding data handling utilities, 
    pre-processing class methods, an injection point for a PyTorch model,
    and a `predict` class method that runs the embedded PyTorch model
    and captures its raw output
    
    Like it's parent _Hydra, _TorchHydra isn't quite functional for streaming data
    as the style of ML prediction outputs is too ambiguous at this stage
     
    As such, the `run`, `start`, and `stop` class methods are not provided. 
    """
    def __init__(self, torch_model, device, nworkers, rt_process_list=[])
        self.ml_model = torch_model
        self.ml_device = device
        self.nworkers = nworkers
        self._last_wind_t0 = None
        self._next_wind_t0 = None
        self.processing_list = rt_process_list
        self.pp_stream = obspy.Stream()
    
    def form_stream(self, order='ZNE'):
        """
        Use 
        """
        stream = Stream()
        if order == 'ZNE':
            stream += self._Ztrace.copy()
            stream += self._Ntrace.copy()
            stream += self._Etrace.copy()
        elif order == 'ENZ':
            stream += self._Etrace.copy()
            stream += self._Ntrace.copy()
            stream += self._Ztrace.copy()
        self.pp_stream = stream
    
    def apply_preprocessing(self):


    def run_prediction(self):
    
    

    


class PickHydra(_TorchHydra):
    """
    Grandchild class of _Hydra that provides a full pipeline for ML prediction
    that connects to the WAVE ring for inputs and the PICK ring for outputs,
    providing 
    """

    def __init__(self, OUT_RING):


class StreamHydra(_TorchHydra):
    """
    Grandchild class of _Hydra that provides a full pipeline for ML prediction
    that connects to WAVE-like rings at either end, taking inputs of raw waveform data
    and providing outputs of continuous prediction values.

    This class is less likely to be used in operations as the data traffic it would produce
    is much larger and has a less clear end result compared to PickHydra, which conforms to
    current WAVE --> PICK processing evolution
    """
    def __init__(self, OUT_RING):

    

# NOTE: ALL OF THESE RUN METHODS GO WITH StreamHydra or PickHydra
    # def start(self):
    #     self._thread.start()

    # def stop(self):
    #     self.runs = False

    # def run(self):
    #     """
    #     Primary process
    #     """
    #     # Main loop
    #     while self.runs:
    #         if self.Zmodule.mod_sta() is False:
    #             break
    #         elif self.Nmodule:
    #             if self.Nmodule.mod_sta() is False:
    #                 break
    #         elif self.Emodule:
    #             if self.Emodule.mod_sta() is False:
    #                 break

    #         time.sleep(0.001)
    #         self.ingest_waves()
    #         ### INSERT PROCESSING CHUNK HERE ###
    #         self.run_ml_pulse()
    #         self.output_waves()
    
    #     self.Zmodule.goodbye()
    #     if self.Nmodule:
    #         self.Nmodule.goodbye()
    #     if self.Emodule:
    #         self.Emodule.goodbye()
    #     raise BufferError(f"{self.station}.{self.network} module exit")
    
    #     logger.info("Exiting")

    
            


def generate_dummy_iconfig():
    iconfig = {'station': 'GNW',
               'network': 'UW',
               'components': 'ENZ,ENN,ENE',
               'native_SR': 100,
               'wave_ring_ID': 1000,
               'pick_ring_ID': 1005,
               'Z_inst_ID': 1001,
               'N_inst_ID': 1002,
               'E_inst_ID': 1003,
               }





class Uroboros:
    """


    
    """
    def __init__(self, iconfig):
        # Instrument attributes
        self.station = iconfig['station']
        self.network = iconfig['network']
        self.components = iconfig['components']
        self.native_sr = iconfig['native_SR']
        # PyEarthworm EWModule attributes
        self.wave_ring_ID = iconfig['wave_ring_ID']
        # self.pick_ring_ID = iconfig['pick_ring_ID']
        self.zinst_id = iconfig['Z_inst_ID']
        self.ninst_id = iconfig['N_inst_ID']
        self.einst_id = iconfig['E_inst_ID']
        self.heartbeat = iconfig['heartbeat']
        # Buffer attributes
        self.raw_buffer = Stream()
        self.res_buffer = Stream()




class TorchUroboros(Uroboros):
    def __init__(self, iconfig):
        super().__init__(iconfig)
        self.model = sbm.PhaseNet
        self.torch_model = self.model.from_pretrained(iconfig['PhaseNet pretrained'])


class EQTransformerUroboros(Uroboros)
    def __init__(self, iconfig):
        super().__init__(iconfig)
        self.model = sbm.PhaseNet