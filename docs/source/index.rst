.. PULSE documentation master file, created by
   sphinx-quickstart on Mon Aug 26 09:03:14 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PULSE documentation (|release|)
===============================

**PULSE**\: **P**\rocessing **U**\tility for **L**\ive **S**\eismic **E**\vents

Overview
--------

PULSE:
 * is an open-source project for integrating emerging seismic analysis codes written in Python into streaming seismic analysis workflows.
 * applies an `event-driven architecture <https://en.wikipedia.org/wiki/Event-driven_architecture>`_, composing processing programs from series of unit modules that each check for, and trigger processing of, new (meta)data packet(s) as they arise using a polymorphic method :meth:`~PULSE.module._base._BaseModule.pulse` shared by all classes defined in :mod:`~PULSE.module`.
 * builds on the `ObsPy <https://obspy.org>`_ and `Numpy <https://numpy.org>`_ APIs to provide familiar, optimized data classes for live seismic data processing defined in :mod:`~PULSE.data`. 
 * uses `SeisBench <https://seisbench.readthedocs.io/en/stable/#>`_ to provide standardized formats and syntax for importing machine learning model architectures and pre-trained weights.
 * uses `PyEarthworm <https://github.com/Boritech-Solutions/PyEarthworm>`_ to directly interface with the Earthworm Message Transport System native to `Earthworm <https://earthwormcentral.org>`_ and operate PULSE programs as Earthworm-recognizable Modules. Example provided in :mod:`~live_example`.


.. This project was inspired by, and builds upon, the workflow presented in .. ref: Retailleau2022



Installation
------------
1. Install `Earthworm 7.10 <http://www.earthwormcentral.org/documentation4/index.html>`_.
2. Install `miniconda <https://docs.anaconda.com/miniconda/miniconda-install/>`_.
3. Create and activate a `conda` environment:

   .. code-block:: console

      conda create -n PULSE pip git  
      conda activate PULSE  

4. Source your Earthworm environment (required for PyEarthworm installation):

   .. code-block:: console

      source /path/to/your/earthworm/params/ew_your_os_type.bash

5. Install `PyEarthworm` from GitHub:

   .. code-block:: console 

      pip install git+https://github.com/Boritech-Solutions/PyEarthworm

6. Install `PULSE` from GitHub:

   .. code-block:: console

      pip install git+https://github.com/pnsn/PULSE.git@develop

Getting Started
---------------

We provide a live example :mod:`live_example` that operates a

Development
-----------

The current distribution of **PULSE** and provided example(s) were developed with the following hardware, OS, and software:

 * Apple M2 MAX chipset and 32 GB RAM.
 * Mac OS 14.X.
 * Earthworm version 7.10

License
-------
PULSE is distributed under a GNU Affero General Public License Version 3 (AGPL-3.0) as detailed
in the attached `LICENSE <https://github.com/pnsn/PULSE/blob/develop/LICENSE>`_ in the PULSE code repository.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   packages/PULSE.mod.rst
   packages/PULSE.data.rst
   .. packages/PULSE.sequences.rst
   .. packages/PULSE.util.rst

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`