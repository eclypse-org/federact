# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

import fedclypse

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "fedclypse"
copyright = str(datetime.now().year) + ", Valerio De Caro"
author = "Valerio De Caro"
release = fedclypse.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_book_theme",
]


# templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

coverage_show_missing_items = True
# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# Don't show class signature with the class' name.
autodoc_class_signature = "separated"
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_title = "Version " + release
html_theme_options = {
    "repository_url": "https://github.com/eclypse-org/fedclypse",
    "use_repository_button": True,
    "collapse_navigation": False,
}
