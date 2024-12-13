from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import sys
import numpy

numpy_include_dir = numpy.get_include()

Options.docstrings = True
Options.annotate = False

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'

ext = Extension(
    name="bsxplorer.cbsx",
    sources=["src/bsxplorer/cython/cbsx.pyx"],
    include_dirs=["src/bsxplorer/cython/cpp"],
    extra_compile_args=["-std=c++17", openmp_arg],
    extra_link_args=["-lgomp", openmp_arg],
)
ext.include_dirs.append(numpy_include_dir)

setup(
    ext_modules=cythonize(ext)
)
