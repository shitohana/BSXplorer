project = 'BismarkPlot'
copyright = '2023, shitohana'
author = 'shitohana'
release = '1.2'
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('../src/bismarkplot'))
sys.path.append(os.path.abspath('.'))


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints'
]

templates_path = ['_templates']
autodoc_mock_imports = ['polars', 'matplotlib', 'numpy', 'scipy', 'pandas', 'pyarrow', 'pyreadr', 'dynamicTreeCut']
autodoc_default_options = {
    'members': True,
    # 'exclude-members': '__init__'
}
autosummary_generate = True
add_module_names = False
html_short_title = 'BismarkPlot Documentation'


html_theme = 'sphinx_book_theme'
