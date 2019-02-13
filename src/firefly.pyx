# -*- coding: utf-8 -*-
# cython: language_level=3

"""Firefly Algorithm.

author: Yuan Chang
copyright: Copyright (C) 2016-2019
license: AGPL
email: pyslvs@gmail.com
"""

from time import time
from numpy import array as np_array
cimport cython
from libc.math cimport exp, log10, sqrt
from numpy cimport ndarray
from verify cimport (
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
    cdef double alpha, alpha0, betaMin, beta0, gamma, min_fit, time_start
    cdef Verification func
    cdef object progress_fun, interrupt_fun
    cdef ndarray lb, ub, fireflies
    cdef Chromosome last_best, current_best
    cdef list fitness_time

    def __cinit__(
        self,
        Verification func,
        dict settings,
        object progress_fun = None,
        object interrupt_fun = None
    ):
        """
        settings = {
            'n',
            'alpha',
            'betaMin',
            'beta0',
            'gamma',
            'max_gen', 'min_fit' or 'max_time',
            'report'
        }
        """
        # object function
        self.func = func
        # D, the dimension of question and each firefly will random place position in this landscape
        self.D = self.func.length()
        # n, the population size of fireflies
        self.n = settings.get('n', 80)
        # alpha, the step size
        self.alpha = settings.get('alpha', 0.01)
        # alpha0, use to calculate_new_alpha
        self.alpha0 = self.alpha
        # betamin, the minimal attraction, must not less than this
        self.betaMin = settings.get('betaMin', 0.2)
        # beta0, the attraction of two firefly in 0 distance.
        self.beta0 = settings.get('beta0', 1.)
        # gamma
        self.gamma = settings.get('gamma', 1.)

        # low bound
        self.lb = np_array(self.func.get_lower())
        # up bound
        self.ub = np_array(self.func.get_upper())

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
            raise Exception("Please give 'max_gen', 'min_fit' or 'max_time' limit.")
        # Report function
        self.rpt = settings.get('report', 0)
        self.progress_fun = progress_fun
        self.interrupt_fun = interrupt_fun

        # all fireflies, depend on population n
        self.fireflies = ndarray(self.n, dtype=object)
        cdef int i
        for i in range(self.n):
            self.fireflies[i] = Chromosome.__new__(Chromosome, self.D)

        # generation of current
        self.gen = 0
        # best firefly of geneation
        self.current_best = Chromosome.__new__(Chromosome, self.D)
        # best firefly so far
        self.last_best = Chromosome.__new__(Chromosome, self.D)

        # setup benchmark
        self.time_start = -1
        self.fitness_time = []

    cdef inline void initialize(self):
        cdef int i, j
        for i in range(self.n):
            # initialize the Chromosome
            for j in range(self.D):
                self.fireflies[i].v[j] = rand_v(self.lb[j], self.ub[j])

    cdef inline void move_fireflies(self):
        cdef int i, j
        cdef bint is_move
        for i in range(self.n):
            is_move = False
            for j in range(self.n):
                is_move |= self.move_firefly(self.fireflies[i], self.fireflies[j])
            if is_move:
                continue
            for j in range(self.D):
                scale = self.ub[j] - self.lb[j]
                self.fireflies[i].v[j] += self.alpha * (rand_v() - 0.5) * scale
                self.fireflies[i].v[j] = self.check(j, self.fireflies[i].v[j])

    cdef inline void evaluate(self):
        cdef Chromosome firefly
        for firefly in self.fireflies:
            firefly.f = self.func.fitness(firefly.v)

    cdef inline bint move_firefly(self, Chromosome me, Chromosome she):
        if me.f <= she.f:
            return False
        cdef double r = _distance(me, she)
        cdef double beta = (self.beta0 - self.betaMin) * exp(-self.gamma * r * r) + self.betaMin
        cdef int i
        for i in range(me.n):
            scale = self.ub[i] - self.lb[i]
            me.v[i] += beta * (she.v[i] - me.v[i]) + self.alpha * (rand_v() - 0.5) * scale
            me.v[i] = self.check(i, me.v[i])
        return True

    cdef inline double check(self, int i, double v):
        if v > self.ub[i]:
            return self.ub[i]
        elif v < self.lb[i]:
            return self.lb[i]
        else:
            return v

    cdef inline Chromosome find_firefly(self):
        cdef int i
        cdef int index = 0
        cdef double f = self.fireflies[0].f
        for i in range(1, len(self.fireflies)):
            if self.fireflies[i].f < f:
                index = i
                f = self.fireflies[i].f
        return self.fireflies[index]

    cdef inline void report(self):
        self.fitness_time.append((self.gen, self.last_best.f, time() - self.time_start))

    cdef inline void generation_process(self):
        self.move_fireflies()
        self.evaluate()
        # adjust alpha, depend on fitness value
        # if fitness value is larger, then alpha should larger
        # if fitness value is small, then alpha should smaller
        self.current_best.assign(self.find_firefly())
        if self.last_best.f > self.current_best.f:
            self.last_best.assign(self.current_best)

        self.alpha = self.alpha0 * log10(self.current_best.f + 1)

        if self.rpt:
            if self.gen % self.rpt == 0:
                self.report()
        else:
            if self.gen % 10 == 0:
                self.report()

    cpdef tuple run(self):
        self.time_start = time()
        self.initialize()
        self.evaluate()
        self.last_best.assign(self.fireflies[0])
        self.report()

        while True:
            self.gen += 1
            self.generation_process()

            if self.option == MAX_GEN:
                if 0 < self.max_gen <= self.gen:
                    break
            elif self.option == MIN_FIT:
                if self.last_best.f <= self.min_fit:
                    break
            elif self.option == MAX_TIME:
                if 0 < self.max_time <= time() - self.time_start:
                    break

            # progress
            if self.progress_fun is not None:
                self.progress_fun(self.gen, f"{self.last_best.f:.04f}")

            # interrupt
            if self.interrupt_fun is not None and self.interrupt_fun():
                break

        self.report()
        return self.func.result(self.last_best.v), self.fitness_time
