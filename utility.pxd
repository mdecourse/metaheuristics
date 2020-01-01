# -*- coding: utf-8 -*-
# cython: language_level=3

"""The callable class of the validation in algorithm.
The 'utility' module should be loaded when using sub-class.

author: Yuan Chang
copyright: Copyright (C) 2016-2020
license: AGPL
email: pyslvs@gmail.com
"""

ctypedef unsigned int uint

cdef enum stop_option:
    MAX_GEN
    MIN_FIT
    MAX_TIME
    SLOW_DOWN

cdef double rand_v(double lower = *, double upper = *) nogil
cdef uint rand_i(int upper) nogil


cdef class Chromosome:
    cdef double f
    cdef double[:] v
    cdef void assign(self, Chromosome obj)
    @staticmethod
    cdef Chromosome[:] new_pop(uint d, uint n)


cdef class Objective:
    cpdef double[:] get_upper(self)
    cpdef double[:] get_lower(self)
    cdef double fitness(self, double[:] v)
    cpdef object result(self, double[:] v)


cdef class AlgorithmBase:

    cdef public Objective func
    cdef uint dim, stop_at_i, gen, rpt
    cdef double stop_at_f, time_start
    cdef stop_option stop_at
    cdef Chromosome last_best
    cdef list fitness_time
    cdef double[:] lb, ub
    cdef object progress_fun, interrupt_fun

    cdef void initialize(self)
    cdef void generation_process(self)
    cdef void report(self)
    cpdef list history(self)
    cpdef object run(self)
