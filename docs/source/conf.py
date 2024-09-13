# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path Setup --------------------------------------------------------------
# Modifications to `sphinx-quickstart` based on:
# - seisbench/docs
#   - https://github.com/seisbench/seisbench/blob/e0046f20de1586322075e740ead1a26d503e604a/docs/conf.py
# - obspy/misc/docs
#   - https://github.com/obspy/obspy/tree/master/misc/docs
# - sphinx-rtd-tutorial.readthedocs.io
#   - https://sphinx-rtd-tutorial.readthedocs.io/en/latest/sphinx-config.html
# (Last Accessed 26 AUG 2024)
import os, sys
from datetime import datetime
from importlib.metadata import version as _version

sys.path.insert(0, os.path.abspath("../../PULSE/"))
# sys.path.insert(0, os.path.abspath("../../live_example/"))
# sys.path.append(os.path.abspath("_ext"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PULSE'
author = 'Nathan T. Stevens, Pacific Northwest Seismic Network'
copyright = f'2023-{datetime.now().year}, {author}, AGPL-3.0'

release = _version(project)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
needs_sphinx = '7.4.7' # Initial docs development version

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme"
]

templates_path = ['_templates']
autodoc_member_order = 'bysource'
master_doc = 'index'

exclude_patterns=['_static','_templates','side_storage']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
pygments_style ='sphinx'


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'obspy': ('https://docs.obspy.org/', None),
}