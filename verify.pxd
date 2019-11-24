# -*- coding: utf-8 -*-
# cython: language_level=3

"""The callable class of the validation in algorithm.
The 'verify' module should be loaded when using sub-class.

author: Yuan Chang
copyright: Copyright (C) 2016-2019
license: AGPL
email: pyslvs@gmail.com
"""

from numpy cimport ndarray

ctypedef unsigned int uint

cdef enum stop_option:
    MAX_GEN
    MIN_FIT
    MAX_TIME
    SLOW_DOWN


cdef double rand_v(double lower = *, double upper = *) nogil
cdef uint rand_i(int upper) nogil


cdef class Chromosome:
    cdef uint n
    cdef double f
    cdef ndarray v
    cdef void assign(self, Chromosome obj)
    @staticmethod
    cdef ndarray[object, ndim=1] new_pop(uint d, uint n)


cdef class Verification:
    cdef ndarray[double, ndim=1] get_upper(self)
    cdef ndarray[double, ndim=1] get_lower(self)
    cdef double fitness(self, ndarray[double, ndim=1] v)
    cpdef object result(self, ndarray[double, ndim=1] v)


cdef class AlgorithmBase:

    cdef uint stop_at_i, gen, rpt
    cdef double stop_at_f, time_start
    cdef stop_option stop_at
    cdef Verification func
    cdef Chromosome last_best
    cdef list fitness_time
    cdef double[:] lb, ub
    cdef object progress_fun, interrupt_fun

    cdef void initialize(self)
    cdef void generation_process(self)
    cdef void report(self)
    cpdef tuple run(self)
