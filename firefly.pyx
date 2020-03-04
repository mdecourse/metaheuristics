# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True

"""Firefly Algorithm

author: Yuan Chang
copyright: Copyright (C) 2016-2020
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from libc.math cimport exp, log10, sqrt, HUGE_VAL
from .utility cimport (
    rand_v,
    Chromosome,
    Objective,
    AlgorithmBase,
)

ctypedef unsigned int uint


cdef double _distance(Chromosome me, Chromosome she, uint dim):
    """Distance of two fireflies."""
    cdef double dist = 0
    cdef uint i
    cdef double diff
    for i in range(dim):
        diff = me.v[i] - she.v[i]
        dist += diff * diff
    return sqrt(dist)


@cython.final
cdef class Firefly(AlgorithmBase):
    """The implementation of Firefly Algorithm."""
    cdef uint n
    cdef double alpha, alpha0, beta_min, beta0, gamma
    cdef Chromosome[:] fireflies

    def __cinit__(
        self,
        Objective func,
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
        self.n = settings.get('n', 80)
        # alpha, the step size
        self.alpha = settings.get('alpha', 0.01)
        # alpha0, use to calculate_new_alpha
        self.alpha0 = self.alpha
        # beta_min, the minimal attraction, must not less than this
        self.beta_min = settings.get('beta_min', 0.2)
        # beta0, the attraction of two firefly in 0 distance.
        self.beta0 = settings.get('beta0', 1.)
        # gamma
        self.gamma = settings.get('gamma', 1.)
        # all fireflies, depended on population n
        self.fireflies = Chromosome.new_pop(self.dim, self.n)

    cdef inline void initialize(self):
        cdef uint i, j
        cdef Chromosome tmp
        for i in range(self.n):
            # initialize the Chromosome
            tmp = self.fireflies[i]
            for j in range(self.dim):
                tmp.v[j] = rand_v(self.lb[j], self.ub[j])
        self.evaluate()
        self.last_best.assign(self.fireflies[0])

    cdef inline void move_fireflies(self):
        cdef uint i
        cdef bint is_move
        cdef double scale, tmp_v
        cdef Chromosome tmp, other
        for tmp in self.fireflies:
            is_move = False
            for other in self.fireflies:
                if tmp is other:
                    continue
                is_move |= self.move_firefly(tmp, other)
            if is_move:
                continue
            for i in range(self.dim):
                scale = self.ub[i] - self.lb[i]
                tmp_v = tmp.v[i] + self.alpha * scale * rand_v(-0.5, 0.5)
                tmp.v[i] = self.check(i, tmp_v)

    cdef inline void evaluate(self):
        cdef Chromosome firefly
        for firefly in self.fireflies:
            firefly.f = self.func.fitness(firefly.v)

    cdef inline bint move_firefly(self, Chromosome me, Chromosome she):
        if me.f <= she.f:
            return False
        cdef double r = _distance(me, she, self.dim)
        cdef double beta = (self.beta0 - self.beta_min) * exp(-self.gamma * r * r) + self.beta_min
        cdef uint i
        cdef double scale, me_v
        for i in range(self.dim):
            scale = self.ub[i] - self.lb[i]
            me_v = me.v[i] + beta * (she.v[i] - me.v[i]) + self.alpha * scale * rand_v(-0.5, 0.5)
            me.v[i] = self.check(i, me_v)
        return True

    cdef inline double check(self, int i, double v):
        if v > self.ub[i]:
            return self.ub[i]
        elif v < self.lb[i]:
            return self.lb[i]
        else:
            return v

    cdef inline Chromosome find_firefly(self):
        cdef int index = 0
        cdef double f = HUGE_VAL

        cdef int i
        cdef Chromosome tmp
        for i, tmp in enumerate(self.fireflies):
            tmp = self.fireflies[i]
            if tmp.f < f:
                index = i
                f = tmp.f
        return self.fireflies[index]

    cdef inline void generation_process(self):
        self.move_fireflies()
        self.evaluate()
        # adjust alpha, depended on fitness value
        # if fitness value is larger, then alpha should larger
        # if fitness value is small, then alpha should smaller
        cdef Chromosome current_best = self.find_firefly()
        if self.last_best.f > current_best.f:
            self.last_best.assign(current_best)
        self.alpha = self.alpha0 * log10(current_best.f + 1)
