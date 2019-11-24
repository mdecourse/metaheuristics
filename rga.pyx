# -*- coding: utf-8 -*-
# cython: language_level=3, embedsignature=True, cdivision=True

"""Real-coded Genetic Algorithm

author: Yuan Chang
copyright: Copyright (C) 2016-2019
license: AGPL
email: pyslvs@gmail.com
"""

from libc.math cimport pow, HUGE_VAL
cimport cython
from numpy import array as np_array
from .verify cimport (
    MAX_GEN,
    rand_v,
    rand_i,
    Chromosome,
    Verification,
    AlgorithmBase,
)

ctypedef unsigned int uint


@cython.final
cdef class Genetic(AlgorithmBase):

    """Algorithm class."""

    cdef uint nParm, nPop
    cdef double pCross, pMute, pWin, bDelta
    cdef Chromosome[:] chromosome, new_chromosome

    def __cinit__(
        self,
        Verification func,
        dict settings,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        """
        settings = {
            'nPop': int,
            'pCross': float,
            'pMute': float,
            'pWin': float,
            'bDelta': float,
            'max_gen': int or 'min_fit': float or 'max_time': float,
            'report': int,
        }
        """
        self.nPop = settings.get('nPop', 500)
        self.pCross = settings.get('pCross', 0.95)
        self.pMute = settings.get('pMute', 0.05)
        self.pWin = settings.get('pWin', 0.95)
        self.bDelta = settings.get('bDelta', 5.)
        self.nParm = len(self.lb)

        self.chromosome = Chromosome.new_pop(self.nParm, self.nPop)
        self.new_chromosome = Chromosome.new_pop(self.nParm, self.nPop)
        self.last_best = Chromosome.__new__(Chromosome, self.nParm)

    cdef inline double check(self, int i, double v):
        """If a variable is out of bound, replace it with a random value."""
        if v > self.ub[i] or v < self.lb[i]:
            return rand_v(self.lb[i], self.ub[i])
        return v

    cdef inline void initialize(self):
        cdef uint i, j
        cdef Chromosome tmp
        for i in range(self.nPop):
            tmp = self.chromosome[i]
            for j in range(self.nParm):
                tmp.v[j] = rand_v(self.lb[j], self.ub[j])

        tmp = self.chromosome[0]
        tmp.f = self.func.fitness(tmp.v)
        self.last_best.assign(tmp)
        self.fitness()

    cdef inline void cross_over(self):
        cdef Chromosome c1 = Chromosome.__new__(Chromosome, self.nParm)
        cdef Chromosome c2 = Chromosome.__new__(Chromosome, self.nParm)
        cdef Chromosome c3 = Chromosome.__new__(Chromosome, self.nParm)

        cdef uint i, s
        cdef Chromosome b1, b2
        for i in range(0, <uint>(self.nPop - 1), 2):
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
            # bubble sort: smaller -> larger
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
        if self.stop_at == MAX_GEN and self.stop_at_i > 0:
            r = <double>self.gen / self.stop_at_i
        else:
            r = 1
        return y * rand_v() * pow(1.0 - r, self.bDelta)

    cdef inline void fitness(self):
        cdef uint i
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
        cdef uint i, s
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

    cdef inline void select(self):
        """roulette wheel selection"""
        cdef uint i, j, k
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
        self.select()
        self.cross_over()
        self.mutate()
        self.fitness()
