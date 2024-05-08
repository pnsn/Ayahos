#!/bin/bash

# Set the path to your local Earthworm installation here
EW_INSTALL_HOME=/usr/local/earthworm
EW_VERSION=memphis
EW_OS_FILE=$EW_HOME/$EW_VERSION/params/ew_macosx.bash

conda create -n wyrm_v0 python=3.8 cython numpy pip
conda activate wyrm_v0



# YAML_FILE='wyrm_v0_apple.yml'

# # Conduct export so PyEarthworm can find Earthworm

# conda create -n wyrm_v0
# conda activate wyrm_v0
# conda install python=3.7 pip
# pip install cython numpy
# export $EW_HOME
# pip install git+https://github.com/Boritech-Solutions/PyEarthworm
# pip install obspy ipython
# conda install -c conda-forge seisbench



