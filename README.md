# Wyrm
### A package joining Python ML workflows to the Earthworm Message Transport System for streaming waveform data operations.
<a title="Richard Dybeck, Public domain, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:U_887,_Skillsta.jpg"><img width="256" alt="U 887, Skillsta" src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/U_887%2C_Skillsta.jpg/256px-U_887%2C_Skillsta.jpg"></a>

https://en.wikipedia.org/wiki/Germanic_dragon
# About:  
This package primarily builds the `PyEarthworm` (or `PyEW`), `ObsPy`, `NumPy`, and `SeisBench`/`PyTorch` APIs to splice machine learning based seismic data analyses written in Python into the Earthworm Message Transport System.  

## Key Project Dependencies
`NumPy`: https://numpy.org  
`ObsPy`: https://docs.obspy.org  
`SeisBench`: https://github.com/seisbench/seisbench  
`PyEarthworm`: https://github.com/Boritech-Solutions/PyEarthworm; https://github.com/Fran89/PyEarthworm_Workshop  

## License
This repository uses an identical GNU Affero General Public License (AGPL-3.0) to match the license from `PyEarthworm` and conform with the GPL-3.0 license for `SeisBench`

## Installation Instructions  

We recommend installing Wyrm and its dependencies in a `conda` environment. Installation instructions of `PyEarthworm` may be subject to change based on its development team.

### Install with `conda` and `pip`
Create and activate your `conda` environment  
```
conda create --name wyrm  
conda activate wyrm  
```
Source your `Earthworm` OS-specific environment, e.g.,  
```
source /usr/local/earthworm/memphis/params/ew_macosx.bash
```
(University of Memphis EW Tankplayer example for MacOS X)  


Install `Wyrm` from `develop`  
```
pip install git+https://github.com/pnsn/wyrm.git@develop
```  
Install `PyEarthworm` from `main`  
```
pip install git+https://github.com/Boritech-Solutions/PyEarthworm
```  

Tested with Python 3.12, Apple M2 chipset, and Earthworm 7.10  

## Notes on the Initial Package Development
This initial version focuses on body wave detection and labeling tasks using the EarthquakeTransformer (EQT; Mousavi et al., 2018) and PhaseNet (Zhu et al., 2019) model architectures, along with the following pretrained model weights available through `SeisBench` (Woollam et al., 2020).

| Model  | Weight   | Appeal                              | Reference               | DOI |
|:------:| -------- | ----------------------------------- | ----------------------- | ------ |
| EQT    | pnw      | PNSN Data Transfer Learning         | Ni et al. (2023)        | https://doi.org/10.26443/seismica.v2i1.368 |
| EQT/PN | instance | Extensive Training Augmentation     | Michelini et al. (2021) | https://doi.org/10.13127/INSTANCE |
| EQT/PN | stead    | "Go-To" Benchmark Training Dataset  | Mousavi et al. (2019)   | https://doi.org/10.1109/ACCESS.2019.2947848 |
| EQT/PN | iquique  | Subduction Zone Aftershock Sequence | Woollam et al. (2019)   | https://doi.org/10.1785/0220180312 |
| EQT/PN | lendb    | Local Seismicity                    | Magrini et al. (2020)   | https://doi.org/10.1016/j.aiig.2020.04.001; http://doi.org/10.5281/zenodo.3648232 |
| PN     | diting   | Large mag range & event diversity   | Zhao et al. (2022)      | https://doi.org/10.1016/j.eqs.2022.01.022 |  

Abstracted from `SeisBench` documentation: https://seisbench.readthedocs.io/en/stable/pages/benchmark_datasets.html#  

## Developer  
Nathan T. Stevens  
email: ntsteven (at) uw.edu  
org: Pacific Northwest Seismic Network

## Branching Plan/Development Notice  

Current development version: ALPHA 

The current developmental version of this code is hosted on the `develop` branch. Starting with version 0.0.1 the `main` branch will host deployment read code, `develop` will contain code that is in beta (debug only), and subsidiary `feature-*` branches will host new functionalities under development..  

## Why "Wyrm"?  
Evocative elements of key tools for this project - `Earthworm`, `Python`, and `Torch` - that echo descriptions of wyrms in European folklore: a subterranean, fire-breathing serpent.  

## Documentation (Work In Progress)  
Will produce Sphinx documentation to host on ReadTheDocs


### Project Structure Outline  
wyrm  
|__core - source code  
|  |__stream     - obspy.core.stream.Stream child classes  
|  |__trace      - obspy.core.trace.Trace child classes  
|  |__wyrm       - process-focused classes ("wyrms")  
|  
|__modules - fully assembled wyrm modules for Py-EW or Py-DISK processing tasks    
|  
|__submodules - assembled sets of wyrms for routine tasks within a complete wyrm module  
|  
|__util - general helper functions  
|  
|__scripts    - additional worked examples (likely to be obsolited)  

