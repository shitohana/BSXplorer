import os
import sys

project = 'BSXplorer'
copyright = '2024, shitohana'
author = 'shitohana'
release = '1.1.0'
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))


extensions = [
    'myst_parser',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode'
]
myst_enable_extensions = [
    "dollarmath",
    "attrs_inline"
]

templates_path = ['_templates']
autodoc_mock_imports = ['polars', 'matplotlib', 'numpy', 'scipy', 'pandas', 'pyarrow', 'pyreadr', 'dynamicTreeCut', 'plotly', 'numba', 'pathlib', 'progress']
autodoc_default_options = {
    'members': True,
    'exclude-members': '__init__'
}
autosummary_generate = True
autodoc_member_order = 'bysource'
add_module_names = False
html_short_title = 'BSXplorer Documentation'
html_favicon = 'images/favicon.ico'
html_logo = 'images/logo.png'
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
html_theme = 'sphinx_rtd_theme'
html_show_sourcelink = False

