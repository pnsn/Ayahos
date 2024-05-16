#  ʔayahos (Ayahos)
## Connecting Python ML Seismic Analysis Tools to Earthworm  

<a title="Depiction of the ʔayahos, Curtis Collection, Public Domain, via Wikimedia Commons" href="https://en.wikipedia.org/wiki/Ayahos">
    <img width="448" alt="Ayahos3" src="https://upload.wikimedia.org/wikipedia/commons/4/41/Ayahos3.png">
</a>  
<a title="Pacific Northwest Seismic Network Logo" href="https://pnsn.org">
    <img width="128" alt="PNSN_GitHub_Logo" src="https://avatars.githubusercontent.com/u/11384896?s=200&v=4">
<a/>  
<img width="50" alt="CC Public Domain Button" src="https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg">

## About  
ʔayahos (Ayahos) is an open-source python project for connecting machine learning (ML) seismic analysis tools into the [`Earthworm`](http://www.earthwormcentral.org) automated seismic processing system. It builds on popular Python APIs used in the seismological research community for routine data processing ([`ObsPy`](https://docs.obspy.org) and [`NumPy`](https://numpy.org)) and ML enhanced analysis tasks ([`SeisBench`](https://seisbench.readthedocs.io/en/stable/) and [`PyTorch`](https://pytorch.org)). Through this project we seek to provide a familiar, modular python API that can be adapted to meet Earthworm installation operators' needs and enable rapid integration of emerging ML tools from the seismological research community into seismic networks' existing automated operations.  

This project relies upon the open-source [`PyEarthworm`](https://github.com/Boritech-Solutions/PyEarthworm) project for brokering in-memory data transfers between Earthworm and Python.  

**We thank each of these development teams for their dedication to open-source scientific software.**  
### License
This project is distributed under a GNU Affero General Public License (AGPL-3.0) to comform with licensing terms of its key dependencies and inspirations.  
<a title="Affero General Public License" href="https://en.wikipedia.org/wiki/GNU_Affero_General_Public_License">
    <img width="256" alt="AGPLv3 Logo" src="https://upload.wikimedia.org/wikipedia/commons/0/06/AGPLv3_Logo.svg">
</a>  

### Whats in a name?  
**ʔayahos (Ayahos), for where we stand**  
This is a product of the Puget Sound, where oral traditions of Lushootseed speaking peoples associate earthquake activity with the supernatural spirit ʔayahus, who can take the form of a monsterous two headed serpent.
* [PNSN Outreach: Serpent Spirit-power Stories along the Seattle Fault](https://pnsn.org/outreach/native-american-stories/serpent-spirit-power/native-american-serpent-spirit-power-stories)
* [Ludiwin et al. (2005, SRL)](https://doi.org/10.1785/gssrl.76.5.526)

**Wyrm, for shorthand**  
The project was initially named Wyrm as its key dependencies `Earthworm` and `PyTorch` echo descriptions of Wyrms in European folklore: subterranean, fire-breathing serpents. This naming convention persists for building-block submodule classes (i.e., `ayahos.core.wyrms`) as a nod to uniqueness in these oral traditions - there are many Wyrms, but only one ʔayahos. 

<a title="Richard Dybeck, Public domain, via Wikimedia Commons" href="https://en.wikipedia.org/wiki/Germanic_dragon  ">
    <img width="128" alt="U 887, Skillsta" src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/U_887%2C_Skillsta.jpg/256px-U_887%2C_Skillsta.jpg">
</a>  
<img width="50" alt="CC Public Domain Button" src="https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg">  



# Getting Started

Ayahos consists of a collection of single-task-oriented building-block classes `wyrms`, adapted obspy `Trace` and `Stream` classes for handling data, and pre-constructed `modules` based on examples from the [PyEarthworm Workshop](https://github.com/Fran89/PyEarthworm_Workshop).    

## For new users 
We recommend installing `Ayahos` and working through the **Pure-Python Tutorials** first to get familiar with the python-side aspects of the Ayahos API.  
Once you're comfortable with these parts of the project, proceed with installing `Earthworm`, a Test Suite dataset, and `PyEarthworm` and try out the **Earthworm-Integrated Tutorials**

### Installation Instructions  

### Installing `Ayahos`
We recommend creating a `conda` environment with clean installs of `pip` and `git` for the current distribution:  
```
conda create --name ayahos pip git
conda activate ayahos
pip install git+https://github.com/pnsn/Ayahos.git@develop
``` 

#### Pure-Python Tutorials (No Earthworm Required)

| Examples                        | Source Data  |  Notebook    | Reference                    |  
| ------------------------------- | ------------ | ------------ | ---------------------------- |
| Introduction to Ayahos Data Classes | local | PLACEHOLDER | | 
| ObsPy Signal Processing | local        | PLACEHOLDER  |                              |
| PhaseNet on One Station         | local        | PLACEHOLDER  | [Retailleau et al. (2022)](https://doi.org/10.1785/0220210279)   |
| EQTransformer on Many Stations  | local        | PLACEHOLDER  | [Ni et al. (2023)](https://doi.org/10.26443/seismica.v2i1.368) | 
| Ensembling Model Predictions    | local        | PLACEHOLDER  | [Yuan et al. (2023)](https://doi.org/10.1109/TGRS.2023.3320148) | 
| PhaseNet + GaMMA Pick/Associate | local        | PLACEHOLDER  | |


### Installing `Earthworm` and a Test Suite
Follow  
* Directions on Earthworm 7.10 installation can be found [here](https://gitlab.rm.ingv.it/earthworm/earthworm)  
* The Univerity of Memphis Test Suite can be downloaded directly [here] (http://www.earthwormcentral.org/distribution/memphis_test.zip)

**NOTE**: The PNSN is developing an PNW test suite to showcase Ayahos' functionalities. Stay tuned!  

### Installing `PyEarthworm`
#### `pip` install from `main`
**NOTE**: This is an abstraction from the PyEarthworm install instructions, refer to their repository for authoritative installation instructions  

Source your `Earthworm` OS-specific environment (e.g., for the Memphis Test Suite example installed on a Mac)     
```
source /usr/local/earthworm/memphis/params/ew_macosx.bash
```

Install `PyEarthworm` from `main`  
```
pip install git+https://github.com/Boritech-Solutions/PyEarthworm
```  

#### Earthworm Integrated Tutorials  

| Examples                        | Source Data  |  Notebook    | 
| ------------------------------- | ------------ | ------------ |
| PhaseNet RING2DISK Prediction   | Tankplayer   | PLACEHOLDER  | 
| ObsPy Picker RING2RING          | Tankplayer   | PLACEHOLDER  |
| PhaseNet Picker RING2RING       | TankPlayer   | PLACEHOLDER  | 
| Ensemble Picker RING2RING       | TankPlayer   | PLACEHOLDER  |
| GaMMA Association RING2RING     | TankPlayer   | PLACEHOLDER  |
| PhaseNet + GaMMA RING2RING        | TankPlayer   | PLACEHOLDER  |


## Installation In A Nutshell (For Experienced Users)
```
conda create --name Ayahos pip git
```
```
conda activate Ayahos
```
```
pip install git+https://github.com/pnsn/Ayahos@develop
```
```
source </path/to/your/ew_env.bash>
```
```
pip install git+https://github.com/Boritech-Solutions/PyEarthworm
```

## Adding Visualization Tools for `class DictStream` (For Experienced Users)  
Ayahos includes data visualization methods for the `DictStream` class that use elements of the [Pyrocko](https://pyrocko.org) project, namely `snuffler`. To add these tools to the environment described above, install the Pyrocko library following their instructions [here](https://pyrocko.org/docs/current/install/). These functionalities are not required for typical module operation with Ayahos, but users may find them handy.  

```
conda install -c pyrocko pyrocko
```

# Documentation (Work In Progress)  
Sphinx documentation in ReadTheDocs formatting is under construction - stay tuned!  
Resource: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/  


<!-- ### Install with `conda`  
The same as above, but using a *.yaml  
```
wget https://github.com/pnsn/Ayahos/conda_env_create.yaml
``` -->



# Additional Information

## Primary Developer  
Nathan T. Stevens  
email: ntsteven (at) uw.edu  
org: Pacific Northwest Seismic Network

## Project Dependencies & Resources
[`Earthworm`](http://www.earthwormcentral.org)  
[`NumPy`](https://numpy.org)  
[`ObsPy`](https://docs.obspy.org)  
[`PyEarthworm`](https://github.com/Boritech-Solutions/PyEarthworm)  
[`PyEarthworm Workshop`](https://github.com/Fran89/PyEarthworm_Workshop)  
[`Pyrocko`](https://pyrocko.org)  
[`SeisBench`](https://github.com/seisbench/seisbench)  

## Branching Plan/Development Notes

Current development version: ALPHA 

The current developmental version of this code is hosted on the `develop` branch. Starting with version 0.0.1 the `main` branch will host deployment read code, `develop` will contain code that is in beta (debug only), and subsidiary `feature-*` branches will host new functionalities under development..  

Developed with Python 3.1X, Apple M2 chipset, and Earthworm 7.10  


<!-- ## Notes on the Initial Package Development
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
 -->