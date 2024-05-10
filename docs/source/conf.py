# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
from datetime import datetime

# The rename is necessary to avoid namespace collision with the version attribute for RTD
# See: https://github.com/sphinx-doc/sphinx/issues/10904
from importlib.metadata import version as _version

sys.path.insert(0, os.path.abspath("../../wyrm/"))

project = 'wyrm'
copyright = f"{datetime.now().year}, Nathan T. Stevens"
author = 'Nathan T. Stevens'
release = _version("wyrm")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
]

templates_path = ['_templates']
exclude_patterns = ['_build','Thumbs.db','.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
