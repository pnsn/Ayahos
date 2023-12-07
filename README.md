# wyrm
### An integration of Python and Earthworm for incorporating pythonic seismology tools into near-real-time monitoring workflows.

# About:  
This code base provides connective utilities between the Python/Earthworm API `PyEarthworm` (or `PyEW`) and the machine learning seismology API `SeisBench`.

# Goals:
Inital focus is providing a set of flexible classes for receiving, parsing, and generating `PyEW`-formatted messages and pre-/post-processing data for waveform-based machine learning algorithms in the `SeisBench` / `PyTorch` machine learning API.  

Additional goal: Integrating the `PyOcto` association algorithm into Earthworm  

#### Developer  
Nate Stevens  
email: ntsteven (at) uw.edu  
org: Pacific Northwest Seismic Network

#### Branching Plan/Development Notice  
The current developmental version of this code will primarily be hosted on the `develop` branch and subsidiary `feature-*` branches in advance of a version 0.0 commit on the `main` branch.  

## Installation for Dev  
Full installation of this software requires the following steps:

1) Install *most* dependencies for the Python-side, e.g.,  
`conda create -f environment/dev_env_apple_silicon.yml`  

2) Follow instructions on installation of EarthWorm and PyEarthworm on the `PyEarthworm` repository: https://github.com/Boritech-Solutions/PyEarthworm


## Key Project Dependencies
`PyEarthworm`: https://github.com/Boritech-Solutions/PyEarthworm  
`SeisBench`: https://github.com/seisbench/seisbench  
`PyOcto`: https://github.com/yetinam/pyocto  
 
## License Note
This repository uses an identical GNU Affero General Public License (APGL-3.0) to match the license from PyEarthworm.