# -*- coding: utf-8 -*-
# cython: language_level=3, embedsignature=True, cdivision=True

"""Firefly Algorithm.

author: Yuan Chang
copyright: Copyright (C) 2016-2019
license: AGPL
email: pyslvs@gmail.com
"""

from time import time
cimport cython
from libc.math cimport (
    exp,
    log10,
    sqrt,
    HUGE_VAL,
)
from numpy cimport ndarray
from .verify cimport (
    Limit,
    MAX_GEN,
    MIN_FIT,
    MAX_TIME,
    rand_v,
    Chromosome,
    Verification,
)


cdef inline double _distance(Chromosome me, Chromosome she):
    """Distance of two fireflies."""
    cdef double dist = 0
    cdef int i
    cdef double diff
    for i in range(me.n):
        diff = me.v[i] - she.v[i]
        dist += diff * diff
    return sqrt(dist)


@cython.final
cdef class Firefly:

    """Algorithm class."""

    cdef Limit option
    cdef int D, n, max_gen, max_time, rpt, gen
    cdef double alpha, alpha0, beta_min, beta0, gamma, min_fit, time_start
    cdef Verification func
    cdef object progress_fun, interrupt_fun
    cdef ndarray lb, ub, fireflies
    cdef Chromosome last_best
    cdef list fitness_time

    def __cinit__(
        self,
        func: Verification,
        settings: dict,
        progress_fun: object = None,
        interrupt_fun: object = None
    ):
        """
        settings = {
            'n',
            'alpha',
            'beta_min',
            'beta0',
            'gamma',
            'max_gen', 'min_fit' or 'max_time',
            'report'
        }
        """
        # object function
        self.func = func

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

        # low bound
        self.lb = self.func.get_lower()
        # up bound
        self.ub = self.func.get_upper()
        if len(self.lb) != len(self.ub):
            raise ValueError("length of upper and lower bounds must be equal")
        # D, the dimension of question and each firefly will random place position in this landscape
        self.D = len(self.lb)

        # Algorithm will stop when the limitation has happened.
        self.max_gen = 0
        self.min_fit = 0
        self.max_time = 0
        if 'max_gen' in settings:
            self.option = MAX_GEN
            self.max_gen = settings['max_gen']
        elif 'min_fit' in settings:
            self.option = MIN_FIT
            self.min_fit = settings['min_fit']
        elif 'max_time' in settings:
            self.option = MAX_TIME
            self.max_time = settings['max_time']
        else:
            raise ValueError("please give 'max_gen', 'min_fit' or 'max_time' limit")
        # Report function
        self.rpt = settings.get('report', 0)
        if self.rpt <= 0:
            self.rpt = 10
        self.progress_fun = progress_fun
        self.interrupt_fun = interrupt_fun

        # all fireflies, depend on population n
        self.fireflies = ndarray(self.n, dtype=object)
        cdef int i
        for i in range(self.n):
            self.fireflies[i] = Chromosome.__new__(Chromosome, self.D)

        # generation of current
        self.gen = 0
        # best firefly so far
        self.last_best = Chromosome.__new__(Chromosome, self.D)

        # setup benchmark
        self.time_start = -1
        self.fitness_time = []

    cdef inline void initialize(self):
        cdef int i, j
        cdef Chromosome tmp
        for i in range(self.n):
            # initialize the Chromosome
            tmp = self.fireflies[i]
            for j in range(self.D):
                tmp.v[j] = rand_v(self.lb[j], self.ub[j])

    cdef inline void move_fireflies(self):
        cdef int i
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
            for i in range(self.D):
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
        cdef double r = _distance(me, she)
        cdef double beta = (self.beta0 - self.beta_min) * exp(-self.gamma * r * r) + self.beta_min
        cdef int i
        cdef double scale, me_v
        for i in range(me.n):
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

    cdef inline void report(self):
        self.fitness_time.append((self.gen, self.last_best.f, time() - self.time_start))

    cdef inline void generation_process(self):
        self.gen += 1

        self.move_fireflies()
        self.evaluate()
        # adjust alpha, depend on fitness value
        # if fitness value is larger, then alpha should larger
        # if fitness value is small, then alpha should smaller
        cdef Chromosome current_best = self.find_firefly()
        if self.last_best.f > current_best.f:
            self.last_best.assign(current_best)

        self.alpha = self.alpha0 * log10(current_best.f + 1)

        if self.gen % self.rpt == 0:
            self.report()

    cpdef tuple run(self):
        self.time_start = time()
        self.initialize()
        self.evaluate()
        self.last_best.assign(self.fireflies[0])
        self.report()

        while True:
            self.generation_process()

            if self.option == MAX_GEN:
                if self.gen >= self.max_gen > 0:
                    break
            elif self.option == MIN_FIT:
                if self.last_best.f <= self.min_fit:
                    break
            elif self.option == MAX_TIME:
                if time() - self.time_start >= self.max_time > 0:
                    break

            # progress
            if self.progress_fun is not None:
                self.progress_fun(self.gen, f"{self.last_best.f:.04f}")

            # interrupt
            if self.interrupt_fun is not None and self.interrupt_fun():
                break

        self.report()
        return self.func.result(self.last_best.v), self.fitness_time
