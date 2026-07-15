# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = 'NebulaPy'
copyright = '2024, Arun Mathew'
author = 'Arun Mathew'

from NebulaPy.version import __version__ as release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc"]

# Mocked so the docs build without the heavy/native stack (ChiantiPy needs its
# atomic database, pypion needs a compiled silo). numpy et al. are NOT mocked:
# modules evaluate constants like numpy.pi at import time, which a mock breaks.
autodoc_mock_imports = ["ChiantiPy", "pypion", "pyneb", "ipyparallel"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
