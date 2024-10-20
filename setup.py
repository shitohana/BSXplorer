from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options

Options.docstrings = True
Options.annotate = False

ext = Extension(
    name="bsxplorer.cbsx",
    sources=["src/bsxplorer/cython/cbsx.pyx"],
    include_dirs=["src/bsxplorer/cython/cpp"],
    extra_compile_args=["-std=c++17", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setup(
    ext_modules=cythonize(ext)
)
