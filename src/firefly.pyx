# -*- coding: utf-8 -*-
# cython: language_level=3

"""Firefly Algorithm.

author: Yuan Chang
copyright: Copyright (C) 2016-2018
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from libc.math cimport exp, log10
import numpy as np
cimport numpy as np
from verify cimport (
    limit,
    maxGen,
    minFit,
    maxTime,
    Chromosome,
    Verification,
)
from libc.stdlib cimport (
    rand,
    RAND_MAX,
    srand,
)
from time import time
srand(int(time()))


cdef double rand_v():
    return rand() / (RAND_MAX * 1.01)


@cython.final
cdef class Firefly:

    """Algorithm class."""

    cdef limit option
    cdef int D, n, maxGen, maxTime, rpt, gen
    cdef double alpha, alpha0, betaMin, beta0, gamma, minFit, time_start
    cdef Verification func
    cdef object progress_fun, interrupt_fun
    cdef np.ndarray lb, ub, fireflys
    cdef Chromosome genbest, bestFirefly
    cdef list fitnessTime

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
            'betaMin',
            'beta0',
            'gamma',
            'maxGen', 'minFit' or 'maxTime',
            'report'
        }
        """
        # object function
        self.func = func
        # D, the dimension of question and each firefly will random place position in this landscape
        self.D = self.func.get_nParm()
        # n, the population size of fireflies
        self.n = settings['n']
        # alpha, the step size
        self.alpha = settings['alpha']
        # alpha0, use to calculate_new_alpha
        self.alpha0 = settings['alpha']
        # betamin, the minimal attraction, must not less than this
        self.betaMin = settings['betaMin']
        # beta0, the attraction of two firefly in 0 distance
        self.beta0 = settings['beta0']
        # gamma
        self.gamma = settings['gamma']

        # low bound
        self.lb = np.array(self.func.get_lower())
        # up bound
        self.ub = np.array(self.func.get_upper())

        # Algorithm will stop when the limitation has happened.
        self.maxGen = 0
        self.minFit = 0
        self.maxTime = 0
        if 'maxGen' in settings:
            self.option = maxGen
            self.maxGen = settings['maxGen']
        elif 'minFit' in settings:
            self.option = minFit
            self.minFit = settings['minFit']
        elif 'maxTime' in settings:
            self.option = maxTime
            self.maxTime = settings['maxTime']
        else:
            raise Exception("Please give 'maxGen', 'minFit' or 'maxTime' limit.")
        # Report function
        self.rpt = settings['report']
        self.progress_fun = progress_fun
        self.interrupt_fun = interrupt_fun

        # all fireflies, depend on population n
        self.fireflys = np.ndarray(self.n, dtype=np.object)
        cdef int i
        for i in range(self.n):
            self.fireflys[i] = Chromosome(self.D)

        # generation of current
        self.gen = 0
        # best firefly of geneation
        self.genbest = Chromosome(self.D)
        # best firefly so far
        self.bestFirefly = Chromosome(self.D)

        # setup benchmark
        self.time_start = time()
        self.fitnessTime = []

    cdef inline void init(self):
        cdef int i, j
        for i in range(self.n):
            # initialize the Chromosome
            for j in range(self.D):
                self.fireflys[i].v[j] = rand_v() * (self.ub[j] - self.lb[j]) + self.lb[j]

    cdef inline void move_fireflies(self):
        cdef int i, j
        cdef bint is_move
        for i in range(self.n):
            is_move = False
            for j in range(self.n):
                is_move |= self.move_firefly(self.fireflys[i], self.fireflys[j])
            if is_move:
                continue
            for j in range(self.D):
                scale = self.ub[j] - self.lb[j]
                self.fireflys[i].v[j] += self.alpha * (rand_v() - 0.5) * scale
                self.fireflys[i].v[j] = self.check(j, self.fireflys[i].v[j])

    cdef inline void evaluate(self):
        cdef Chromosome firefly
        for firefly in self.fireflys:
            firefly.f = self.func.fitness(firefly.v)

    cdef inline bint move_firefly(self, Chromosome me, Chromosome she):
        if me.f <= she.f:
            return False
        cdef double r = me.distance(she)
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
        cdef double f = self.fireflys[0].f
        for i in range(1, len(self.fireflys)):
            if self.fireflys[i].f < f:
                index = i
                f = self.fireflys[i].f
        return self.fireflys[index]

    cdef inline void report(self):
        self.fitnessTime.append((self.gen, self.bestFirefly.f, time() - self.time_start))

    cdef inline void generation_process(self):
        self.move_fireflies()
        self.evaluate()
        # adjust alpha, depend on fitness value
        # if fitness value is larger, then alpha should larger
        # if fitness value is small, then alpha should smaller
        self.genbest.assign(self.find_firefly())
        if self.bestFirefly.f > self.genbest.f:
            self.bestFirefly.assign(self.genbest)

        self.alpha = self.alpha0 * log10(self.genbest.f + 1)

        if self.rpt:
            if self.gen % self.rpt == 0:
                self.report()
        else:
            if self.gen % 10 == 0:
                self.report()

    cpdef tuple run(self):
        self.init()
        self.evaluate()
        self.bestFirefly.assign(self.fireflys[0])
        self.report()
        while True:
            self.gen += 1
            if self.option == maxGen:
                if 0 < self.maxGen < self.gen:
                    break
            elif self.option == minFit:
                if self.bestFirefly.f <= self.minFit:
                    break
            elif self.option == maxTime:
                if 0 < self.maxTime <= time() - self.time_start:
                    break
            self.generation_process()
            # progress
            if self.progress_fun:
                self.progress_fun(self.gen, f"{self.bestFirefly.f:.04f}")
            # interrupt
            if self.interrupt_fun and self.interrupt_fun():
                break
        self.report()
        return self.func.result(self.bestFirefly.v), self.fitnessTime
