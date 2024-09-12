.. _installation:


Installation and Configuration
==============================

PULSE installation is currently supported using `pip` and requires installation
of `Earthworm 7.10 <http://www.earthwormcentral.org/documentation4/index.html>`_
and `PyEarthworm <http://github.com/Boritech-Solutions/PyEarthworm>`_ prior to
installing PULSE. We strongly recommend installing PULSE in it's own virtual environment,
which we provide an example for using `conda`.

Pre-Requisite Installation Steps
--------------------------------

1. Install `Earthworm 7.10 <http://www.earthwormcentral.org/documentation4/index.html>`_.
2. Install `miniconda <https://docs.anaconda.com/miniconda/miniconda-install/>`_.
3. Create and activate a `conda` environment:

   .. code-block:: console

      conda create -n PULSE pip git  
      conda activate PULSE  

4. Source your Earthworm environment (required for PyEarthworm installation). For example on OS X:

   .. code-block:: console

      source /path/to/your/earthworm/params/ew_macosx.bash


5. Install `PyEarthworm` from GitHub:

   .. code-block:: console 

      pip install git+https://github.com/Boritech-Solutions/PyEarthworm

Installation with `pip`
----------------------

PULSE is currently available from GitHub. To install locally run: ::

      pip install git+https://github.com/pnsn/PULSE.git@develop