# wyrm
### An integration of Python and Earthworm for incorporating pythonic seismology tools into near-real-time monitoring workflows.
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

## Installation Instructions (Work In Progress)
Full installation of this software requires the following steps:

Create initial conda environment  
`conda create --name wyrm python=3.12`  
Activate environment  
`conda activate wyrm`  
install Wyrm from the main project directory using a `setuptools` backend to `pip`  
`python -m pip install .`

#### If interfacing with Earthworm via PyEarthworm (most cases) . 
source your earthworm installation environment. e.g.,  
`source /usr/local/earthworm/memphis/params/ew_macosx.bash`  
install PyEarthworm from their `main` branch  
`pip install git+https://github.com/Boritech-Solutions/PyEarthworm`  


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
The current developmental version of this code will primarily be hosted on the `develop` branch and subsidiary `feature-*` branches in advance of a version 0.0 commit on the `main` branch.  

## Why "wyrm"?  
Evocative elements of key tools for this code base - `Earthworm`, `Python`, and `Torch` - parallel descriptions of wyrms and dragons across various parts and periods of European folklore: a subterranean, fire-breathing serpent.  




## References  



## Documentation (Work In Progress)  
Will produce Sphinx documentation to host on ReadTheDocs


