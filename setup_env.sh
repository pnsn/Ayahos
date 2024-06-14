#!/bin/bash
# :auth: Nathan T. Stevens
# :email: ntsteven (at) uw.edu
# :org: Pacific Northwest Seismic Network
# :license: AGPL-3.0
# :purpose: This script creates a conda environment and installs EWFlow and all its dependencies
ENVNAME='PULSED'
ACTIVATE="$CONDA_PREFIX/bin/activate"

# Check if environment exists
if conda info --envs | grep -q "$ENVNAME"; then
    echo "Conda env '$ENVNAME' already exists"
# If it doesn't exist, create environment and install dependencies
else
    conda create --name $ENVNAME python=3.9 -y
    echo "~~~~~~~ Environment '$ENVNAME' created ~~~~~~~"
    source "$ACTIVATE" "$ENVNAME"
fi

#  Check if environment is active
if [[ "$CONDA_DEFAULT_ENV" == "$ENVNAME" ]]; then
    echo "~~~~~~~ Environment '$ENVNAME' already active ~~~~~~~"    
else
    echo "~~~~~~~ Activating '$ENVNAME' ~~~~~~"
    source "$ACTIVATE" "$ENVNAME"
fi

# Install Ayahos and dependencies using pip backend
if [[ "$CONDA_DEFAULT_ENV" == "$ENVNAME" ]]; then
    echo "~~~~~~~ Installing EWFlow and dependencies from local copy ~~~~~~~"
    python -m pip install .
fi