import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "demovuln"
author = "Alex Giménez-Romero"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",
]

# autosummary_generate = True

html_theme = "sphinx_rtd_theme"

autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

nb_execution_mode = "off"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
