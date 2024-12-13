from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import sys
import numpy

Options.docstrings = True
Options.annotate = False
numpy_path = numpy.get_include()

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
ext.include_dirs.append(numpy_path)

setup(
    ext_modules=cythonize(ext)
)
