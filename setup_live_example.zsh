#!/bin/zsh
ENVNAME='PULSE'
ACTIVATE="$CONDA_PREFIX/bin/activate"
PXD='~~|^v~~~'
# Check if environment exists
if conda info --envs | grep -q "$ENVNAME"; then
    echo "$PXD Conda env '$ENVNAME' already exists $PXD"
# If it doesn't exist, create environment and install dependencies
else
    echo "$PXD '$ENVNAME' does not exist $PXD"
fi

#  Check if environment is active
if [[ "$CONDA_DEFAULT_ENV" == "$ENVNAME" ]]; then
    echo "$PXD Environment '$ENVNAME' already active $PXD"    
else
    echo "$PXD Activating '$ENVNAME' $PXD"
    source "$ACTIVATE" "$ENVNAME"
fi
# source live_example/params/ew_macosx.bash
# startstop