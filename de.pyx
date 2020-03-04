# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False

"""Differential Evolution

author: Yuan Chang
copyright: Copyright (C) 2016-2020
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from .utility cimport (
    rand_v,
    rand_i,
    Chromosome,
    Objective,
    AlgorithmBase,
)

ctypedef unsigned int uint
ctypedef void (*Eq)(Differential, int, Chromosome)


cdef enum Strategy:
    STRATEGY0
    STRATEGY1
    STRATEGY2
    STRATEGY3
    STRATEGY4
    STRATEGY5
    STRATEGY6
    STRATEGY7
    STRATEGY8
    STRATEGY9


@cython.final
cdef class Differential(AlgorithmBase):
    """The implementation of Differential Evolution."""
    cdef Strategy strategy
    cdef uint np, r1, r2, r3, r4, r5
    cdef double F, CR
    cdef Chromosome[:] pool

    def __cinit__(
        self,
        Objective func,
        dict settings,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        """
        settings = {
            'strategy': int,
            'NP': int,
            'F': float,
            'CR': float,
            'max_gen': int or 'min_fit': float or 'max_time': float,
            'report': int,
        }
        """
        # strategy 0~9, choice what strategy to generate new member in temporary
        cdef uint strategy = settings.get('strategy', STRATEGY1)
        if strategy > 9:
            raise ValueError(f"invalid strategy: {strategy}")
        self.strategy = <Strategy>strategy
        # population size
        # To start off np = 10*dim is a reasonable choice. Increase np if misconvergence
        self.np = settings.get('NP', 400)
        # weight factor F is usually between 0.5 and 1 (in rare cases > 1)
        self.F = settings.get('F', 0.6)
        if not (0.5 <= self.F <= 1):
            raise ValueError('CR should be [0.5,1]')
        # crossover possible CR in [0,1]
        self.CR = settings.get('CR', 0.9)
        if not (0 <= self.CR <= 1):
            raise ValueError('CR should be [0,1]')
        # the vector
        self.r1 = self.r2 = self.r3 = self.r4 = self.r5 = 0
        # generation pool, depended on population size
        self.pool = Chromosome.new_pop(self.dim, self.np)

    cdef inline void initialize(self):
        """Initial population."""
        cdef uint i, j
        cdef Chromosome tmp
        for i in range(self.np):
            tmp = self.pool[i]
            for j in range(self.dim):
                tmp.v[j] = rand_v(self.lb[j], self.ub[j])
            tmp.f = self.func.fitness(tmp.v)
        self.last_best.assign(self.find_best())

    cdef inline Chromosome find_best(self):
        """Find member that have minimum fitness value from pool."""
        cdef Chromosome best = self.pool[0]
        cdef Chromosome tmp
        for tmp in self.pool[1:]:
            if tmp.f < best.f:
                best = tmp
        return best

    cdef inline void generate_random_vector(self, uint i):
        """Generate new vector."""
        self.r1 = self.r2 = self.r3 = self.r4 = self.r5 = i
        while self.r1 == i:
            self.r1 = rand_i(self.np)
        while self.r2 in {i, self.r1}:
            self.r2 = rand_i(self.np)
        if self.strategy in {STRATEGY1, STRATEGY3, STRATEGY6, STRATEGY8}:
            return
        while self.r3 in {i, self.r1, self.r2}:
            self.r3 = rand_i(self.np)
        if self.strategy in {STRATEGY2, STRATEGY7}:
            return
        while self.r4 in {i, self.r1, self.r2, self.r3}:
            self.r4 = rand_i(self.np)
        if self.strategy in {STRATEGY4, STRATEGY9}:
            return
        while self.r5 in {i, self.r1, self.r2, self.r3, self.r4}:
            self.r5 = rand_i(self.np)

    cdef inline void type1(self, Chromosome tmp, Eq func):
        cdef uint n = rand_i(self.dim)
        cdef uint l_v = 0
        while True:
            func(self, n, tmp)
            n = (n + 1) % self.dim
            l_v += 1
            if not (rand_v() < self.CR and l_v < self.dim):
                break

    cdef inline void type2(self, Chromosome tmp, Eq func):
        cdef uint n = rand_i(self.dim)
        cdef uint l_v
        for l_v in range(self.dim):
            if rand_v() < self.CR or l_v == self.dim - 1:
                func(self, n, tmp)
            n = (n + 1) % self.dim

    cdef void eq1(self, int n, Chromosome tmp):
        cdef Chromosome c1 = self.pool[self.r1]
        cdef Chromosome c2 = self.pool[self.r2]
        tmp.v[n] = self.last_best.v[n] + self.F * (c1.v[n] - c2.v[n])

    cdef void eq2(self, int n, Chromosome tmp):
        cdef Chromosome c1 = self.pool[self.r1]
        cdef Chromosome c2 = self.pool[self.r2]
        cdef Chromosome c3 = self.pool[self.r3]
        tmp.v[n] = c1.v[n] + self.F * (c2.v[n] - c3.v[n])

    cdef void eq3(self, int n, Chromosome tmp):
        cdef Chromosome c1 = self.pool[self.r1]
        cdef Chromosome c2 = self.pool[self.r2]
        tmp.v[n] = tmp.v[n] + self.F * (self.last_best.v[n] - tmp.v[n]) + self.F * (c1.v[n] - c2.v[n])

    cdef void eq4(self, int n, Chromosome tmp):
        cdef Chromosome c1 = self.pool[self.r1]
        cdef Chromosome c2 = self.pool[self.r2]
        cdef Chromosome c3 = self.pool[self.r3]
        cdef Chromosome c4 = self.pool[self.r4]
        tmp.v[n] = self.last_best.v[n] + (c1.v[n] + c2.v[n] - c3.v[n] - c4.v[n]) * self.F

    cdef void eq5(self, int n, Chromosome tmp):
        cdef Chromosome c1 = self.pool[self.r1]
        cdef Chromosome c2 = self.pool[self.r2]
        cdef Chromosome c3 = self.pool[self.r3]
        cdef Chromosome c4 = self.pool[self.r4]
        cdef Chromosome c5 = self.pool[self.r5]
        tmp.v[n] = c5.v[n] + (c1.v[n] + c2.v[n] - c3.v[n] - c4.v[n]) * self.F

    cdef inline Chromosome recombination(self, int i):
        """use new vector, recombination the new one member to tmp."""
        cdef Chromosome tmp = Chromosome.__new__(Chromosome, self.dim)
        tmp.assign(self.pool[i])
        if self.strategy == 1:
            self.type1(tmp, Differential.eq1)
        elif self.strategy == 2:
            self.type1(tmp, Differential.eq2)
        elif self.strategy == 3:
            self.type1(tmp, Differential.eq3)
        elif self.strategy == 4:
            self.type1(tmp, Differential.eq4)
        elif self.strategy == 5:
            self.type1(tmp, Differential.eq5)
        elif self.strategy == 6:
            self.type2(tmp, Differential.eq1)
        elif self.strategy == 7:
            self.type2(tmp, Differential.eq2)
        elif self.strategy == 8:
            self.type2(tmp, Differential.eq3)
        elif self.strategy == 9:
            self.type2(tmp, Differential.eq4)
        elif self.strategy == 0:
            self.type2(tmp, Differential.eq5)
        return tmp

    cdef inline bint over_bound(self, Chromosome member):
        """check the member's chromosome that is out of bound?"""
        cdef uint i
        for i in range(self.dim):
            if member.v[i] > self.ub[i] or member.v[i] < self.lb[i]:
                return True
        return False

    cdef inline void generation_process(self):
        cdef uint i
        cdef Chromosome tmp, baby
        for i in range(self.np):
            # generate new vector
            self.generate_random_vector(i)
            # use the vector recombine the member to temporary
            tmp = self.recombination(i)
            # check the one is out of bound?
            if self.over_bound(tmp):
                # if it is, then abandon it
                continue
            # is not out of bound, that mean it's qualify of environment
            # then evaluate the one
            tmp.f = self.func.fitness(tmp.v)
            # if temporary one is better than origin(fitness value is smaller)
            baby = self.pool[i]
            if tmp.f <= baby.f:
                # copy the temporary one to origin member
                baby.assign(tmp)
                # check the temporary one is better than the current_best
                if tmp.f < self.last_best.f:
                    # copy the temporary one to current_best
                    self.last_best.assign(tmp)
