# -*- coding: utf-8 -*-
# cython: language_level=3

"""Real-coded Genetic Algorithm.

author: Yuan Chang
copyright: Copyright (C) 2016-2019
license: AGPL
email: pyslvs@gmail.com
"""

from time import time
from libc.math cimport pow
cimport cython
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


@cython.final
cdef class Genetic:

    """Algorithm class."""

    cdef Limit option
    cdef int nParm, nPop, max_gen, max_time, gen, rpt
    cdef double pCross, pMute, pWin, bDelta, min_fit, time_start
    cdef Verification func
    cdef object progress_fun, interrupt_fun
    cdef ndarray chromosome, new_chromosome, baby_chromosome, ub, lb
    cdef Chromosome current_best, last_best
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
        self.nParm = self.func.length()
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
            raise Exception("Please give 'max_gen', 'min_fit' or 'max_time' limit.")
        self.rpt = settings.get('report', 0)
        self.progress_fun = progress_fun
        self.interrupt_fun = interrupt_fun

        # low bound
        self.lb = self.func.get_lower()
        # up bound
        self.ub = self.func.get_upper()

        self.chromosome = ndarray(self.nPop, dtype=object)
        self.new_chromosome = ndarray(self.nPop, dtype=object)
        self.baby_chromosome = ndarray(3, dtype=object)
        cdef int i
        for i in range(self.nPop):
            self.chromosome[i] = Chromosome.__new__(Chromosome, self.nParm)
        for i in range(self.nPop):
            self.new_chromosome[i] = Chromosome.__new__(Chromosome, self.nParm)
        for i in range(3):
            self.baby_chromosome[i] = Chromosome.__new__(Chromosome, self.nParm)

        self.current_best = Chromosome.__new__(Chromosome, self.nParm)
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
        for j in range(self.nPop):
            for i in range(self.nParm):
                self.chromosome[j].v[i] = rand_v(self.lb[i], self.ub[i])

    cdef inline void cross_over(self):
        cdef int i, s, j
        cdef Chromosome baby
        for i in range(0, self.nPop - 1, 2):
            if not rand_v() < self.pCross:
                continue
            for s in range(self.nParm):
                # first baby, half father half mother
                self.baby_chromosome[0].v[s] = 0.5 * self.chromosome[i].v[s] + 0.5 * self.chromosome[i + 1].v[s]
                # second baby, three quarters of father and quarter of mother
                self.baby_chromosome[1].v[s] = self.check(s, 1.5 * self.chromosome[i].v[s] - 0.5 * self.chromosome[i + 1].v[s])
                # third baby, quarter of father and three quarters of mother
                self.baby_chromosome[2].v[s] = self.check(s, -0.5 * self.chromosome[i].v[s] + 1.5 * self.chromosome[i + 1].v[s])
            # evaluate new baby
            for j in range(3):
                self.baby_chromosome[j].f = self.func.fitness(self.baby_chromosome[j].v)
            # maybe use bubble sort? smaller -> larger
            if self.baby_chromosome[1].f < self.baby_chromosome[0].f:
                self.baby_chromosome[0], self.baby_chromosome[1] = self.baby_chromosome[1], self.baby_chromosome[0]
            if self.baby_chromosome[2].f < self.baby_chromosome[0].f:
                self.baby_chromosome[2], self.baby_chromosome[0] = self.baby_chromosome[0], self.baby_chromosome[2]
            if self.baby_chromosome[2].f < self.baby_chromosome[1].f:
                self.baby_chromosome[2], self.baby_chromosome[1] = self.baby_chromosome[1], self.baby_chromosome[2]
            # replace first two baby to parent, another one will be
            baby = self.chromosome[i]
            baby.assign(self.baby_chromosome[0])
            baby = self.chromosome[i + 1]
            baby.assign(self.baby_chromosome[1])

    @cython.cdivision
    cdef inline double delta(self, double y):
        cdef double r
        if self.max_gen > 0:
            r = <double>self.gen / self.max_gen
        else:
            r = 1
        return y * rand_v() * pow(1.0 - r, self.bDelta)

    cdef inline void fitness(self):
        cdef int j
        for j in range(self.nPop):
            self.chromosome[j].f = self.func.fitness(self.chromosome[j].v)
        self.last_best.assign(self.chromosome[0])
        for j in range(1, self.nPop):
            if self.chromosome[j].f < self.last_best.f:
                self.last_best.assign(self.chromosome[j])
        if self.last_best.f < self.current_best.f:
            self.current_best.assign(self.last_best)

    cdef inline void mutate(self):
        cdef int i, s
        for i in range(self.nPop):
            if not rand_v() < self.pMute:
                continue
            s = int(rand_v() * self.nParm)
            if int(rand_v() * 2) == 0:
                self.chromosome[i].v[s] += self.delta(self.ub[s] - self.chromosome[i].v[s])
            else:
                self.chromosome[i].v[s] -= self.delta(self.chromosome[i].v[s] - self.lb[s])

    cdef inline void report(self):
        self.fitness_time.append((self.gen, self.current_best.f, time() - self.time_start))

    cdef inline void select(self):
        """
        roulette wheel selection
        """
        cdef int i, j, k
        cdef Chromosome baby
        for i in range(self.nPop):
            j = int(rand_v() * self.nPop)
            k = int(rand_v() * self.nPop)
            baby = self.new_chromosome[i]
            baby.assign(self.chromosome[j])
            if self.chromosome[k].f < self.chromosome[j].f and rand_v() < self.pWin:
                baby = self.new_chromosome[i]
                baby.assign(self.chromosome[k])
        # in this stage, new_chromosome is select finish
        # now replace origin chromosome
        for i in range(self.nPop):
            baby = self.chromosome[i]
            baby.assign(self.new_chromosome[i])
        # select random one chromosome to be best chromosome, make best chromosome still exist
        j = int(rand_v() * self.nPop)
        baby = self.chromosome[j]
        baby.assign(self.current_best)

    cdef inline void generation_process(self):
        self.select()
        self.cross_over()
        self.mutate()
        self.fitness()
        if self.rpt:
            if self.gen % self.rpt == 0:
                self.report()
        else:
            if self.gen % 10 == 0:
                self.report()

    cpdef tuple run(self):
        """Init and run GA for max_gen times."""
        self.time_start = time()
        self.initialize()
        self.chromosome[0].f = self.func.fitness(self.chromosome[0].v)
        self.current_best.assign(self.chromosome[0])
        self.fitness()
        self.report()

        while True:
            self.gen += 1
            self.generation_process()

            if self.option == MAX_GEN:
                if 0 < self.max_gen <= self.gen:
                    break
            elif self.option == MIN_FIT:
                if self.current_best.f <= self.min_fit:
                    break
            elif self.option == MAX_TIME:
                if 0 < self.max_time <= time() - self.time_start:
                    break

            # progress
            if self.progress_fun is not None:
                self.progress_fun(self.gen, f"{self.current_best.f:.04f}")

            # interrupt
            if self.interrupt_fun is not None and self.interrupt_fun():
                break

        self.report()
        return self.func.result(self.current_best.v), self.fitness_time
