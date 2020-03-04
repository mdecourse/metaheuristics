# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True

"""Real-coded Genetic Algorithm

author: Yuan Chang
copyright: Copyright (C) 2016-2020
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from libc.math cimport pow, HUGE_VAL
from .utility cimport (
    MAX_GEN,
    rand_v,
    rand_i,
    Chromosome,
    Objective,
    AlgorithmBase,
)

ctypedef unsigned int uint


@cython.final
cdef class Genetic(AlgorithmBase):
    """The implementation of Real-coded Genetic Algorithm."""
    cdef uint pop_num
    cdef double cross, mute, win, delta
    cdef Chromosome[:] chromosome, new_chromosome

    def __cinit__(
        self,
        Objective func,
        dict settings,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        """
        settings = {
            'pop_num': int,
            'cross': float,
            'mute': float,
            'win': float,
            'delta': float,
            'max_gen': int or 'min_fit': float or 'max_time': float,
            'report': int,
        }
        """
        self.pop_num = settings.get('pop_num', 500)
        self.cross = settings.get('cross', 0.95)
        self.mute = settings.get('mute', 0.05)
        self.win = settings.get('win', 0.95)
        self.delta = settings.get('delta', 5.)
        self.chromosome = Chromosome.new_pop(self.dim, self.pop_num)
        self.new_chromosome = Chromosome.new_pop(self.dim, self.pop_num)

    cdef inline double check(self, int i, double v):
        """If a variable is out of bound, replace it with a random value."""
        if v > self.ub[i] or v < self.lb[i]:
            return rand_v(self.lb[i], self.ub[i])
        return v

    cdef inline void initialize(self):
        cdef uint i, j
        cdef Chromosome tmp
        for i in range(self.pop_num):
            tmp = self.chromosome[i]
            for j in range(self.dim):
                tmp.v[j] = rand_v(self.lb[j], self.ub[j])
        tmp = self.chromosome[0]
        tmp.f = self.func.fitness(tmp.v)
        self.last_best.assign(tmp)
        self.fitness()

    cdef inline void cross_over(self):
        cdef Chromosome c1 = Chromosome.__new__(Chromosome, self.dim)
        cdef Chromosome c2 = Chromosome.__new__(Chromosome, self.dim)
        cdef Chromosome c3 = Chromosome.__new__(Chromosome, self.dim)
        cdef uint i, s
        cdef Chromosome b1, b2
        for i in range(0, <uint>(self.pop_num - 1), 2):
            if not rand_v() < self.cross:
                continue

            b1 = self.chromosome[i]
            b2 = self.chromosome[i + 1]
            for s in range(self.dim):
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

    cdef inline double get_delta(self, double y):
        cdef double r
        if self.stop_at == MAX_GEN and self.stop_at_i > 0:
            r = <double>self.gen / self.stop_at_i
        else:
            r = 1
        return y * rand_v() * pow(1.0 - r, self.delta)

    cdef inline void fitness(self):
        cdef uint i
        cdef Chromosome tmp
        for i in range(self.pop_num):
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
        for i in range(self.pop_num):
            if not rand_v() < self.mute:
                continue
            s = rand_i(self.dim)
            tmp = self.chromosome[i]
            if rand_v() < 0.5:
                tmp.v[s] += self.get_delta(self.ub[s] - tmp.v[s])
            else:
                tmp.v[s] -= self.get_delta(tmp.v[s] - self.lb[s])

    cdef inline void select(self):
        """roulette wheel selection"""
        cdef uint i, j, k
        cdef Chromosome baby, b1, b2
        for i in range(self.pop_num):
            j = rand_i(self.pop_num)
            k = rand_i(self.pop_num)
            b1 = self.chromosome[j]
            b2 = self.chromosome[k]
            baby = self.new_chromosome[i]
            if b1.f > b2.f and rand_v() < self.win:
                baby.assign(b2)
            else:
                baby.assign(b1)
        # in this stage, new_chromosome is select finish
        # now replace origin chromosome
        for i in range(self.pop_num):
            baby = self.chromosome[i]
            baby.assign(self.new_chromosome[i])
        # select random one chromosome to be best chromosome, make best chromosome still exist
        baby = self.chromosome[rand_i(self.pop_num)]
        baby.assign(self.last_best)

    cdef inline void generation_process(self):
        self.select()
        self.cross_over()
        self.mutate()
        self.fitness()
