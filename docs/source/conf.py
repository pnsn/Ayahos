# Configuration file for the Sphinx documentation builder.
#
# This file was modeled after the conf.py for SeisBench documentation (accessed May 2024)
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
from datetime import datetime
from importlib.metadata import version as _version
sys.path.insert(0, os.path.abspath("../"))


project = 'Ayahos'
copyright = f'{datetime.now().year}, Nathan T. Stevens, Pacific Northwest Seismic Network'
author = 'Nathan T. Stevens, Pacific Northwest Seismic Network'
release = _version('ayahos')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
    ]

templates_path = ['_templates']
exclude_patterns = ['_build','.DS_Store']

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "_static/Ayahos3.png"
html_favicon = "_static/PNSN_Logo.png"

html_theme_options = {
    "logo_only": True,
    "display_version": False,
}