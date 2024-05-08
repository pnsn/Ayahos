#!/bin/bash
ENV_ROOT=~/miniconda3/envs/wyrm_v0
cp PyEW.cpython-312-darwin.so $ENV_ROOT/lib/python3.12/site-packages/
cp -r PyEarthworm-1.41.dist-info $ENV_ROOT/lib/python3.12/site-packages/