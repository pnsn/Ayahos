#!/bin/bash
#
# Shell Script to set up the Earthworm/Ayahos environment needed to run
# this example of Ayahos
# 
# This script is based on the script of the same name by Francisco Hernandez
# as part of the PyEarthworm Workshop github repository and is used in compliance
# with its licensing and our AGPL-3.0 licensing of Ayahos

# Add This Ayahos Module To Path
CurDir="$( cd "$( dirname "$0" )" && pwd )"
cd $CurDir
export PATH=$PATH:$CurDir/bin
export $CurDir/params/ew_macosx.bash
conda activate Ayahos

