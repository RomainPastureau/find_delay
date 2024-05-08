import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'src')))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'find_delay'
copyright = '2024, Romain Pastureau'
author = 'Romain Pastureau'
release = '2.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx_tabs.tabs',
    'sphinx_copybutton',
    #'sphinx.ext.autosectionlabel',
    #'sphinx-autodoc-typehints',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    "style_nav_header_background": '#99cc00',
    "collapse_navigation": False,
    "navigation_depth": 4,
    "includehidden": False
}

html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

napoleon_use_param = True
