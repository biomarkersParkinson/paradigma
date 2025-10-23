# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "paradigma"
copyright = "2025, Erik Post, Kars Veldkamp, Nienke Timmermans, Diogo Coutinho Soriano, Vedran Kasalica, Peter Kok, and Luc Evers"
author = "Erik Post, Kars Veldkamp, Nienke Timmermans, Diogo Coutinho Soriano, Vedran Kasalica, Peter Kok, and Luc Evers"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

myst_enable_extensions = [
    "substitution",
    "deflist",
]

autoapi_dirs = ["../src"]

# Include the following entities in the API documentation, this explicitly excludes 'imported-members',
# as we don't want to clutter the documentation with all the imported members.
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#confval-autoapi_options
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

# myst-nb execution settings
jupyter_execute_notebooks = "off"
nb_kernel_name = "python3"
nb_execution_timeout = 600  # bump timeout if needed

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_static_path = ["source/_static"]
html_css_files = ["custom.css"]


def setup(app):
    app.add_css_file("custom.css")
