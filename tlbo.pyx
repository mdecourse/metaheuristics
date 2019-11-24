# -*- coding: utf-8 -*-
# cython: language_level=3, embedsignature=True, cdivision=True

"""Teaching Learning Based Optimization

author: Yuan Chang
copyright: Copyright (C) 2016-2019
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from .verify cimport (
    Verification,
    AlgorithmBase,
)

ctypedef unsigned int uint


@cython.final
cdef class TeachingLearning(AlgorithmBase):

    """Algorithm class."""

    cdef uint class_size

    def __cinit__(
        self,
        Verification func,
        dict settings,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        """
        settings = {
            'class_size': int,
            'max_gen': int or 'min_fit': float or 'max_time': float,
            'report': int,
        }
        """
        self.class_size = settings.get('class_size', 50)

    cdef void initialize(self):
        """Initialize function."""
        raise NotImplementedError

    cdef void generation_process(self):
        """The process of each generation."""
        raise NotImplementedError
