# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True

"""The callable class of the validation in algorithm.
The 'utility' module should be loaded when using sub-class of base classes.

author: Yuan Chang
copyright: Copyright (C) 2016-2020
license: AGPL
email: pyslvs@gmail.com
"""

from time import process_time
from numpy import zeros, array, float64 as np_float
cimport cython
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time


cdef inline double rand_v(double lower = 0., double upper = 1.) nogil:
    """Random real value between lower <= r <= upper."""
    return lower + <double>rand() / RAND_MAX * (upper - lower)


cdef inline uint rand_i(int upper) nogil:
    """A random integer between 0 <= r < upper."""
    return rand() % upper


@cython.final
@cython.freelist(100)
cdef class Chromosome:
    """Data structure class."""

    def __cinit__(self, uint n):
        self.f = 0.
        self.v = zeros(n, dtype=np_float)

    cdef void assign(self, Chromosome other):
        """Assign from an old generation."""
        if other is self:
            return
        self.f = other.f
        self.v[:] = other.v

    @staticmethod
    cdef Chromosome[:] new_pop(uint d, uint n):
        """Create new population."""
        return array([Chromosome.__new__(Chromosome, d) for _ in range(n)])


cdef class Objective:
    """Objective function base class.

    It is used to build the objective function for Metaheuristic Random
    Algorithms.
    """

    cpdef double[:] get_upper(self):
        """Return upper bound."""
        raise NotImplementedError

    cpdef double[:] get_lower(self):
        """Return lower bound."""
        raise NotImplementedError

    cdef double fitness(self, double[:] v):
        raise NotImplementedError

    cpdef object result(self, double[:] v):
        """Return the result from the variable list `v`."""
        raise NotImplementedError


cdef class AlgorithmBase:
    """Algorithm base class.

    It is used to build the Metaheuristic Random Algorithms.
    """

    def __cinit__(
        self,
        Objective func,
        dict settings,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        """Generic settings."""
        srand(time(NULL))
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
        self.dim = len(self.ub)
        if self.dim != len(self.lb):
            raise ValueError("length of upper and lower bounds must be equal")
        self.last_best = Chromosome.__new__(Chromosome, self.dim)
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
        """Report generation, fitness and time."""
        self.fitness_time.append((
            self.gen,
            self.last_best.f,
            process_time() - self.time_start,
        ))

    cpdef list history(self):
        """Return the history of the process.

        The first value is generation (iteration);
        the second value is fitness;
        the third value is time in second.
        """
        return self.fitness_time

    cpdef object run(self):
        """Run and return the result and convergence history.

        The first place of `return` is came from
        calling [`Objective.result()`](#objectiveresult).

        The second place of `return` is a list of generation data,
        which type is `Tuple[int, float, float]]`.
        The first of them is generation,
        the second is fitness, and the last one is time in second.
        """
        self.time_start = process_time()
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
                if process_time() - self.time_start >= self.stop_at_f > 0:
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
        return self.func.result(self.last_best.v)
