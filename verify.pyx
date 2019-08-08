# -*- coding: utf-8 -*-
# cython: language_level=3, embedsignature=True, cdivision=True

"""The callable class of the validation in algorithm.
The 'verify' module should be loaded when using sub-class of base classes.

author: Yuan Chang
copyright: Copyright (C) 2016-2019
license: AGPL
email: pyslvs@gmail.com
"""

from time import time
from numpy import zeros
cimport cython
from libc.stdlib cimport rand, srand, RAND_MAX

srand(int(time()))


cdef inline double rand_v(double lower = 0., double upper = 1.) nogil:
    """Random real value between [lower, upper]."""
    return lower + <double>rand() / RAND_MAX * (upper - lower)


cdef inline int rand_i(int upper) nogil:
    """Random integer between [0, upper]."""
    return rand() % upper


@cython.final
@cython.freelist(100)
cdef class Chromosome:

    """Data structure class."""

    def __cinit__(self, n: cython.uint):
        self.n = n if n > 0 else 2
        self.f = 0.
        self.v = zeros(n)

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


cdef class AlgorithmBase:

    """Algorithm base class."""

    def __cinit__(
        self,
        func: Verification,
        settings: dict,
        progress_fun: object = None,
        interrupt_fun: object = None
    ):
        """Generic settings."""
        # object function
        self.func = func

        self.stop_at_i = 0
        self.stop_at_f = 0.
        if 'max_gen' in settings:
            self.stop_at = MAX_GEN
            self.stop_at_i = settings['max_gen']
        elif 'min_fit' in settings:
            self.stop_at = MIN_FIT
            self.stop_at_f = settings['min_fit']
        elif 'max_time' in settings:
            self.stop_at = MAX_TIME
            self.stop_at_f = settings['max_time']
        elif 'slow_down' in settings:
            self.stop_at = SLOW_DOWN
            self.stop_at_f = 1 - settings['slow_down']
        else:
            raise ValueError("please give 'max_gen', 'min_fit' or 'max_time' limit")
        self.rpt = settings.get('report', 0)
        if self.rpt <= 0:
            self.rpt = 10
        self.progress_fun = progress_fun
        self.interrupt_fun = interrupt_fun

        self.lb = self.func.get_lower()
        self.ub = self.func.get_upper()
        if len(self.lb) != len(self.ub):
            raise ValueError("length of upper and lower bounds must be equal")

        # setup benchmark
        self.gen = 0
        self.time_start = 0
        self.fitness_time = []

    cdef void initialize(self):
        """Initialize function."""
        raise NotImplementedError

    cdef void generation_process(self):
        """The process of each generation."""
        raise NotImplementedError

    cdef inline void report(self):
        self.fitness_time.append((self.gen, self.last_best.f, time() - self.time_start))

    @cython.cdivision
    cpdef tuple run(self):
        """Init and run GA for max_gen times."""
        self.time_start = time()
        self.initialize()
        self.report()

        cdef double diff, last_best
        cdef double last_diff = 0
        while True:
            last_best = self.last_best.f
            self.gen += 1
            self.generation_process()
            if self.gen % self.rpt == 0:
                self.report()
            if self.stop_at == MAX_GEN:
                if self.gen >= self.stop_at_i > 0:
                    break
            elif self.stop_at == MIN_FIT:
                if self.last_best.f <= self.stop_at_f:
                    break
            elif self.stop_at == MAX_TIME:
                if time() - self.time_start >= self.stop_at_f > 0:
                    break
            elif self.stop_at == SLOW_DOWN:
                diff = last_best - self.last_best.f
                if last_diff > 0 and diff / last_diff >= self.stop_at_f:
                    break
                last_diff = diff
            # progress
            if self.progress_fun is not None:
                self.progress_fun(self.gen, f"{self.last_best.f:.04f}")
            # interrupt
            if (self.interrupt_fun is not None) and self.interrupt_fun():
                break
        self.report()
        return self.func.result(self.last_best.v), self.fitness_time
