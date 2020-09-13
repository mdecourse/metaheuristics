# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False

"""Firefly Algorithm

author: Yuan Chang
copyright: Copyright (C) 2016-2020
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from libc.math cimport exp, sqrt
from .utility cimport rand_v, ObjFunc, Algorithm

ctypedef unsigned int uint


cdef double _distance(double[:] me, double[:] she, uint dim) nogil:
    """Distance of two fireflies."""
    cdef double dist = 0
    cdef uint i
    cdef double diff
    for i in range(dim):
        diff = me[i] - she[i]
        dist += diff * diff
    return sqrt(dist)


@cython.final
cdef class Firefly(Algorithm):
    """The implementation of Firefly Algorithm."""
    cdef double alpha, beta_min, beta0, gamma

    def __cinit__(
        self,
        ObjFunc func,
        dict settings,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        """
        settings = {
            'n': int,
            'alpha': float,
            'beta_min': float,
            'beta0': float,
            'gamma': float,
            'max_gen': int or 'min_fit': float or 'max_time': float,
            'report': int,
        }
        """
        # n, the population size of fireflies
        self.pop_num = settings.get('n', 80)
        # alpha, the step size
        self.alpha = settings.get('alpha', 0.01)
        # beta_min, the minimal attraction, must not less than this
        self.beta_min = settings.get('beta_min', 0.2)
        # beta0, the attraction of two firefly in 0 distance
        self.beta0 = settings.get('beta0', 1.)
        # gamma
        self.gamma = settings.get('gamma', 1.)
        # all fireflies, depended on population n
        self.new_pop()

    cdef inline void initialize(self):
        cdef uint i, j
        for i in range(self.pop_num):
            for j in range(self.dim):
                self.pool[i, j] = rand_v(self.func.lb[j], self.func.ub[j])
        self.evaluate()
        self.set_best(0)

    cdef inline void evaluate(self) nogil:
        for i in range(self.pop_num):
            self.fitness[i] = self.func.fitness(self.pool[i, :])

    cdef inline void move_fireflies(self) nogil:
        cdef bint is_move
        cdef uint i, j, s
        cdef double scale, tmp_v
        for i in range(self.pop_num):
            moved = False
            for j in range(self.pop_num):
                if i == j or self.fitness[i] <= self.fitness[j]:
                    continue
                self.move_firefly(self.pool[i, :], self.pool[j, :])
                moved = True
            if moved:
                continue
            for s in range(self.dim):
                self.pool[i, s] = self.check(s, self.pool[i, s] + self.alpha * (
                    self.func.ub[s] - self.func.lb[s]) * rand_v(-0.5, 0.5))

    cdef inline void move_firefly(self, double[:] me, double[:] she) nogil:
        cdef double r = _distance(me, she, self.dim)
        cdef double beta = ((self.beta0 - self.beta_min)
                            * exp(-self.gamma * r * r) + self.beta_min)
        cdef uint i
        for i in range(self.dim):
            me[i] = self.check(i, me[i] + beta * (she[i] - me[i]) + self.alpha
                               * (self.func.ub[i] - self.func.lb[i])
                               * rand_v(-0.5, 0.5))

    cdef inline double check(self, int i, double v) nogil:
        """Check the bounds."""
        if v > self.func.ub[i]:
            return self.func.ub[i]
        elif v < self.func.lb[i]:
            return self.func.lb[i]
        else:
            return v

    cdef inline void find_firefly(self) nogil:
        cdef uint best = self.find_best()
        if self.fitness[best] < self.best_f:
            self.set_best(best)

    cdef inline void generation_process(self) nogil:
        self.move_fireflies()
        self.evaluate()
        self.find_firefly()
