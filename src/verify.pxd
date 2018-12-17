# -*- coding: utf-8 -*-
# cython: language_level=3

"""The callable class of the validation in algorithm.

author: Yuan Chang
copyright: Copyright (C) 2016-2018
license: AGPL
email: pyslvs@gmail.com
"""

from numpy cimport ndarray


cdef enum Limit:
    maxGen
    minFit
    maxTime


cdef double rand_v() nogil


cdef class Chromosome:
    cdef public int n
    cdef public double f
    cdef public ndarray v
    cdef void assign(self, Chromosome obj)


cdef class Verification:
    cdef ndarray[double, ndim=1] get_upper(self)
    cdef ndarray[double, ndim=1] get_lower(self)
    cdef int get_nParm(self)
    cdef double fitness(self, ndarray[double, ndim=1] v)
    cpdef object result(self, ndarray[double, ndim=1] v)
