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


To learn more about the PULSE API, see the code :ref:`documentation_overview`.

To see how to install PULSE on your local machine, see the :ref:`installation` page.

.. To see a working example of PULSE on live streaming data, see the :ref:`live_example`.

.. Getting Started
.. ---------------

.. We provide a live example :mod:`live_example` that operates a

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
   :maxdepth: 4
   :hidden:

   self
   pages/installation.rst
   pages/live_example.rst
   pages/documentation/overview.rst

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`