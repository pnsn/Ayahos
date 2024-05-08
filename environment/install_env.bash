#!/bin/bash

# This uses the paths associated with the Memphis tankplayer example
conda create --name wyrm_v0 python=3.8 cython numpy
eval "$(conda shell.bash hook)"
source ~/miniconda3/bin/activate wyrm_v0

source /usr/local/earthworm/memphis/params/ew_macosx.bash