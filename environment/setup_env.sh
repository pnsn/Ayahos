#!/bin/bash

# Set the path to your local Earthworm installation here
EARTHWORM_LOCAL=/usr/local/earthworm
YAML_FILE='wyrm_v0_apple.yml'

# Conduct export so PyEarthworm can find Earthworm

conda create -n wyrm_v0
conda activate wyrm_v0
conda install python=3.7 pip
pip install cython numpy
export EW_HOME=$EARTHWORM_LOCAL
pip install git+https://github.com/Boritech-Solutions/PyEarthworm
pip install obspy ipython
conda install -c conda-forge seisbench



