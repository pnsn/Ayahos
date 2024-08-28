#!/bin/bash/
cd ..
python -m pip install .
cd docs
make clean
make html