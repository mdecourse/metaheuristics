# -*- coding: utf-8 -*-
# cython: language_level=3

"""The callable class of the validation in algorithm.

author: Yuan Chang
copyright: Copyright (C) 2016-2018
license: AGPL
email: pyslvs@gmail.com
"""

from time import time
from numpy import zeros as np_zeros
cimport cython
from libc.stdlib cimport rand, RAND_MAX, srand
srand(int(time()))


@cython.cdivision
cdef inline double rand_v() nogil:
    return <double>rand() / (RAND_MAX * 1.01)


@cython.final
@cython.freelist(100)
cdef class Chromosome:

    """Data structure class."""

    def __cinit__(self, n: int):
        self.n = n if n > 0 else 2
        self.f = 0.0
        self.v = np_zeros(n)

    cdef void assign(self, Chromosome other):
        if other is self:
            return
        self.n = other.n
        self.f = other.f
        self.v = other.v.copy()


cdef class Verification:

    """Verification function class base."""

    cdef ndarray[double, ndim=1] get_upper(self):
        """Return upper bound."""
        raise NotImplementedError

    cdef ndarray[double, ndim=1] get_lower(self):
        """Return lower bound."""
        raise NotImplementedError

    cdef int get_nParm(self):
        """How many parameters do we need."""
        raise NotImplementedError

    cdef double fitness(self, ndarray[double, ndim=1] v):
        """Calculate the fitness.

        Usage:
        f = MyVerification()
        fitness = f(chromosome.v)
        """
        raise NotImplementedError

    cpdef object result(self, ndarray[double, ndim=1] v):
        """Show the result."""
        raise NotImplementedError
