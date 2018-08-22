# -*- coding: utf-8 -*-

"""Compile the Cython libraries of Pyslvs."""

from distutils.core import setup, Extension
import os

from Cython.Distutils import build_ext
import numpy


sources = []
for source in os.listdir("./src"):
    if source.split('.')[-1] == 'pyx':
        sources.append(source)

extra_compile_args = [
    # Compiler optimize.
    '-O3',
    # Disable NumPy warning only on Linux.
    '-Wno-cpp',
    # Windows format warning.
    '-Wno-format',
]


def basename(name: str) -> str:
    """No suffix name."""
    return name.split('.')[0]


# Original src
ext_modules = []
for source in sources:
    ext_modules.append(Extension(
        basename(source),
        sources = ['src/' + source], # path + file name
        
        include_dirs = [numpy.get_include()],
        extra_compile_args = extra_compile_args,
    ))


setup(
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)
