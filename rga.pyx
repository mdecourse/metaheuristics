# -*- coding: utf-8 -*-
# cython: language_level=3, embedsignature=True, cdivision=True

"""Real-coded Genetic Algorithm.

author: Yuan Chang
copyright: Copyright (C) 2016-2019
license: AGPL
email: pyslvs@gmail.com
"""

from time import time
from libc.math cimport pow, HUGE_VAL
cimport cython
from numpy cimport ndarray
from .verify cimport (
    Limit,
    MAX_GEN,
    MIN_FIT,
    MAX_TIME,
    rand_v,
    rand_i,
    Chromosome,
    Verification,
)


@cython.final
cdef class Genetic:

    """Algorithm class."""

    cdef Limit option
    cdef int nParm, nPop, max_gen, max_time, gen, rpt
    cdef double pCross, pMute, pWin, bDelta, min_fit, time_start
    cdef Verification func
    cdef object progress_fun, interrupt_fun
    cdef ndarray chromosome, new_chromosome, ub, lb
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
            'nPop',
            'pCross',
            'pMute',
            'pWin',
            'bDelta',
            'max_gen' or 'min_fit' or 'max_time',
            'report'
        }
        """
        self.func = func

        self.nPop = settings.get('nPop', 500)
        self.pCross = settings.get('pCross', 0.95)
        self.pMute = settings.get('pMute', 0.05)
        self.pWin = settings.get('pWin', 0.95)
        self.bDelta = settings.get('bDelta', 5.)
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
        self.rpt = settings.get('report', 0)
        if self.rpt <= 0:
            self.rpt = 10
        self.progress_fun = progress_fun
        self.interrupt_fun = interrupt_fun

        # low bound
        self.lb = self.func.get_lower()
        # up bound
        self.ub = self.func.get_upper()
        if len(self.lb) != len(self.ub):
            raise ValueError("length of upper and lower bounds must be equal")
        self.nParm = len(self.lb)

        self.chromosome = ndarray(self.nPop, dtype=object)
        self.new_chromosome = ndarray(self.nPop, dtype=object)
        cdef int i
        for i in range(self.nPop):
            self.chromosome[i] = Chromosome.__new__(Chromosome, self.nParm)
        for i in range(self.nPop):
            self.new_chromosome[i] = Chromosome.__new__(Chromosome, self.nParm)

        self.last_best = Chromosome.__new__(Chromosome, self.nParm)

        # generations
        self.gen = 0

        # setup benchmark
        self.time_start = -1
        self.fitness_time = []

    cdef inline double check(self, int i, double v):
        """If a variable is out of bound, replace it with a random value."""
        if v > self.ub[i] or v < self.lb[i]:
            return rand_v(self.lb[i], self.ub[i])
        return v

    cdef inline void initialize(self):
        cdef int i, j
        cdef Chromosome tmp
        for i in range(self.nPop):
            tmp = self.chromosome[i]
            for j in range(self.nParm):
                tmp.v[j] = rand_v(self.lb[j], self.ub[j])

        tmp = self.chromosome[0]
        tmp.f = self.func.fitness(tmp.v)
        self.last_best.assign(tmp)

    cdef inline void cross_over(self):
        cdef Chromosome c1 = Chromosome.__new__(Chromosome, self.nParm)
        cdef Chromosome c2 = Chromosome.__new__(Chromosome, self.nParm)
        cdef Chromosome c3 = Chromosome.__new__(Chromosome, self.nParm)

        cdef int i, s, j
        cdef Chromosome b1, b2
        for i in range(0, self.nPop - 1, 2):
            if not rand_v() < self.pCross:
                continue

            b1 = self.chromosome[i]
            b2 = self.chromosome[i + 1]
            for s in range(self.nParm):
                # first baby, half father half mother
                c1.v[s] = 0.5 * b1.v[s] + 0.5 * b2.v[s]
                # second baby, three quarters of father and quarter of mother
                c2.v[s] = self.check(s, 1.5 * b1.v[s] - 0.5 * b2.v[s])
                # third baby, quarter of father and three quarters of mother
                c3.v[s] = self.check(s, -0.5 * b1.v[s] + 1.5 * b2.v[s])
            # evaluate new baby
            c1.f = self.func.fitness(c1.v)
            c2.f = self.func.fitness(c2.v)
            c3.f = self.func.fitness(c3.v)
            # maybe use bubble sort? smaller -> larger
            if c1.f > c2.f:
                c1, c2 = c2, c1
            if c1.f > c3.f:
                c1, c3 = c3, c1
            if c2.f > c3.f:
                c2, c3 = c3, c2
            # replace first two baby to parent, another one will be
            b1.assign(c1)
            b2.assign(c2)

    cdef inline double delta(self, double y):
        cdef double r
        if self.max_gen > 0:
            r = <double>self.gen / self.max_gen
        else:
            r = 1
        return y * rand_v() * pow(1.0 - r, self.bDelta)

    cdef inline void fitness(self):
        cdef int i
        cdef Chromosome tmp
        for i in range(self.nPop):
            tmp = self.chromosome[i]
            tmp.f = self.func.fitness(tmp.v)

        cdef int index = 0
        cdef double f = HUGE_VAL

        for i, tmp in enumerate(self.chromosome):
            if tmp.f < f:
                index = i
                f = tmp.f
        if f < self.last_best.f:
            self.last_best.assign(self.chromosome[index])

    cdef inline void mutate(self):
        cdef int i, s
        cdef Chromosome tmp
        for i in range(self.nPop):
            if not rand_v() < self.pMute:
                continue
            s = rand_i(self.nParm)
            tmp = self.chromosome[i]
            if rand_v() < 0.5:
                tmp.v[s] += self.delta(self.ub[s] - tmp.v[s])
            else:
                tmp.v[s] -= self.delta(tmp.v[s] - self.lb[s])

    cdef inline void report(self):
        self.fitness_time.append((self.gen, self.last_best.f, time() - self.time_start))

    cdef inline void select(self):
        """
        roulette wheel selection
        """
        cdef int i, j, k
        cdef Chromosome baby, b1, b2
        for i in range(self.nPop):
            j = rand_i(self.nPop)
            k = rand_i(self.nPop)
            b1 = self.chromosome[j]
            b2 = self.chromosome[k]
            baby = self.new_chromosome[i]
            if b1.f > b2.f and rand_v() < self.pWin:
                baby.assign(b2)
            else:
                baby.assign(b1)
        # in this stage, new_chromosome is select finish
        # now replace origin chromosome
        for i in range(self.nPop):
            baby = self.chromosome[i]
            baby.assign(self.new_chromosome[i])
        # select random one chromosome to be best chromosome, make best chromosome still exist
        baby = self.chromosome[rand_i(self.nPop)]
        baby.assign(self.last_best)

    cdef inline void generation_process(self):
        self.gen += 1

        self.select()
        self.cross_over()
        self.mutate()
        self.fitness()
        if self.gen % self.rpt == 0:
            self.report()

    cpdef tuple run(self):
        """Init and run GA for max_gen times."""
        self.time_start = time()
        self.initialize()
        self.fitness()
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
