#!/usr/bin/env python
"""
Ayahos_Pick_Mod.py is an example module that:
    1) reads tracebuff2 messages from the WAVE RING of an Earthworm instance
    2) conducts on-the-fly windowing and waveform data preprocessing
    3) predicts phase arrival times using the EQTransformer model architecture and PNW pretrained weights
        (Mousavi et al., 2019; Ni et al., 2023)
    4) Triggers with a set threshold (0.21) and uses maximum probabilities of triggers as onset times
        (after Ni et al., 2023)
    5) Converts trigger/pick data to TYPE_PICK2K messages and submits them to the PICK RING

This module is based on examples 4 and 5 in the PyEarthworm Workshop 

"""
import configparser, argparse, logging, os, sys, time
from ayahos import Ayahos
from ayahos.wyrms import *

# Setup argument parsing for an input cfg file
parser = argparse.ArgumentParser(description="This is an example wave ring to pick ring module with Ayahos")
parser.add_argument('-f', action='store', dest='ConfFile', default='Ayahos_Pick_Mod.cfg', type=str)

# Read arguments
inargs = parser.parse_args()

# Parse cfg file
Config = configparser.ConfigParser()
Config.read(inargs.ConfFile)

# # Setup Logging
# log_path = os.environ['EW_LOG']
# log_name = inargs.ConfFile.split('.')[0]+'.log'

# formatter = logging.Formatter('%(ascitime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)
Logger = logging.getLogger('Ayahos_Pick_Mod')


