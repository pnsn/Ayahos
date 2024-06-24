#!/bin/bash
# :auth: Nathan T. Stevens
# :email: ntsteven (at) uw.edu
# :org: Pacific Northwest Seismic Network
# :license: AGPL-3.0
# :purpose: This script creates a conda environment and installs EWFlow and all its dependencies
ENVNAME='PULSE'
ACTIVATE="$CONDA_PREFIX/bin/activate"
PXD='~~^v~~~'

# Check if environment exists
if conda info --envs | grep -q "$ENVNAME"; then
    echo "$PXD Conda env '$ENVNAME' already exists"
# If it doesn't exist, create environment and install dependencies
else
    conda create --name $ENVNAME python=3.9 -y
    echo "$PXD Environment '$ENVNAME' created"
    source "$ACTIVATE" "$ENVNAME"
fi

#  Check if environment is active
if [[ "$CONDA_DEFAULT_ENV" == "$ENVNAME" ]]; then
    echo "$PXD Environment '$ENVNAME' already active"    
else
    echo "$PXD Activating '$ENVNAME'"
    source "$ACTIVATE" "$ENVNAME"
fi

# Install PULSE and dependencies using pip backend
if [[ "$CONDA_DEFAULT_ENV" == "$ENVNAME" ]]; then
    echo "$PXD Installing '$ENVNAME' and dependencies from local copy"
    python -m pip install .
    echo "$PXD Sourcing Mac OSX Earthworm Environment"
    source live_example/params/ew_macosx.bash
    echo "$PXD Installing PyEarthworm from GitHub"
    pip install git+https://github.com/Boritech-Solutions/PyEarthworm
    echo "$PXD Installation Complete!"
fi