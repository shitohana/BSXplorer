import os
import sys

project = 'BSXplorer'
copyright = '2023, shitohana'
author = 'shitohana'
release = '1.0.0a2'
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

import bsxplorer

extensions = [
    # TODO add myst_parser to dependencies
    'myst_parser',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
autodoc_mock_imports = ['polars', 'matplotlib', 'numpy', 'scipy', 'pandas', 'pyarrow', 'pyreadr', 'dynamicTreeCut', 'plotly', 'numba', 'pathlib']
autodoc_default_options = {
    'members': True,
    'exclude-members': '__init__'
}
autosummary_generate = True
autodoc_member_order = 'bysource'
add_module_names = False
html_short_title = 'BSXplorer Documentation'


html_theme = 'sphinx_rtd_theme'
html_show_sourcelink = False

