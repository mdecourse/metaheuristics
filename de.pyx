# -*- coding: utf-8 -*-
# cython: language_level=3, embedsignature=True, cdivision=True

"""Differential Evolution.

author: Yuan Chang
copyright: Copyright (C) 2016-2019
license: AGPL
email: pyslvs@gmail.com
"""

from time import time
cimport cython
from libc.math cimport HUGE_VAL
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

ctypedef void (*Eq)(Differential, int, Chromosome)


@cython.final
cdef class Differential:

    """Algorithm class."""

    cdef Limit option
    cdef int strategy, D, NP, max_gen, max_time, rpt, gen, r1, r2, r3, r4, r5
    cdef double F, CR, min_fit, time_start
    cdef ndarray lb, ub, pool
    cdef Verification func
    cdef object progress_fun, interrupt_fun
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
            'strategy',
            'NP',
            'F',
            'CR',
            'max_gen', 'min_fit' or 'max_time',
            'report'
        }
        """
        # object function, or environment
        self.func = func

        # strategy 0~9, choice what strategy to generate new member in temporary
        self.strategy = settings.get('strategy', 1)
        # population size
        # To start off NP = 10*D is a reasonable choice. Increase NP if misconvergence
        self.NP = settings.get('NP', 400)
        # weight factor
        # F is usually between 0.5 and 1 (in rare cases > 1)
        self.F = settings.get('F', 0.6)
        # crossover possible
        # CR in [0,1]
        self.CR = settings.get('CR', 0.9)
        # low bound
        self.lb = self.func.get_lower()
        # up bound
        self.ub = self.func.get_upper()
        if len(self.lb) != len(self.ub):
            raise ValueError("length of upper and lower bounds must be equal")
        # dimension of question
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

        # check parameter is set properly
        self.check_parameter()

        # generation pool, depend on population size
        self.pool = ndarray(self.NP, dtype=object)
        cdef int i
        for i in range(self.NP):
            self.pool[i] = Chromosome.__new__(Chromosome, self.D)

        # last generation best member
        self.last_best = Chromosome.__new__(Chromosome, self.D)

        # the generation count
        self.gen = 0

        # the vector
        self.r1 = self.r2 = self.r3 = self.r4 = self.r5 = 0

        # setup benchmark
        self.time_start = -1
        self.fitness_time = []

    cdef inline void check_parameter(self):
        """Check parameter is set properly."""
        if self.D <= 0:
            raise ValueError('D should be integer and larger than 0')
        if self.NP <= 0:
            raise ValueError('NP should be integer and larger than 0')
        if not (0 <= self.CR <= 1):
            raise ValueError('CR should be [0,1]')
        if self.strategy not in range(10):
            raise ValueError('strategy should be [0,9]')

    cdef inline void initialize(self):
        """Initial population."""
        cdef int i, j
        cdef Chromosome tmp
        for i in range(self.NP):
            tmp = self.pool[i]
            for j in range(self.D):
                tmp.v[j] = rand_v(self.lb[j], self.ub[j])
            tmp.f = self.func.fitness(tmp.v)

    cdef inline Chromosome find_best(self):
        """Find member that have minimum fitness value from pool."""
        cdef int index = 0
        cdef double f = HUGE_VAL

        cdef int i
        cdef Chromosome tmp
        for i, tmp in enumerate(self.pool):
            if tmp.f < f:
                index = i
                f = tmp.f
        return self.pool[index]

    cdef inline void generate_random_vector(self, int i):
        """Generate new vector."""
        self.r1 = self.r2 = self.r3 = self.r4 = self.r5 = i
        while self.r1 == i:
            self.r1 = rand_i(self.NP)
        while self.r2 in {i, self.r1}:
            self.r2 = rand_i(self.NP)

        if self.strategy in {1, 3, 6, 8}:
            return

        while self.r3 in {i, self.r1, self.r2}:
            self.r3 = rand_i(self.NP)

        if self.strategy in {2, 7}:
            return

        while self.r4 in {i, self.r1, self.r2, self.r3}:
            self.r4 = rand_i(self.NP)

        if self.strategy in {4, 9}:
            return

        while self.r5 in {i, self.r1, self.r2, self.r3, self.r4}:
            self.r5 = rand_i(self.NP)

    cdef inline void type1(self, Chromosome tmp, Eq func):
        cdef int n = rand_i(self.D)
        cdef int l_v = 0
        while True:
            func(self, n, tmp)
            n = (n + 1) % self.D
            l_v += 1
            if not (rand_v() < self.CR and l_v < self.D):
                break

    cdef inline void type2(self, Chromosome tmp, Eq func):
        cdef int n = rand_i(self.D)
        cdef int l_v
        for l_v in range(self.D):
            if rand_v() < self.CR or l_v == self.D - 1:
                func(self, n, tmp)
            n = (n + 1) % self.D

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
        cdef Chromosome tmp = Chromosome.__new__(Chromosome, self.D)
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

    cdef inline void report(self):
        self.fitness_time.append((self.gen, self.last_best.f, time() - self.time_start))

    cdef inline bint over_bound(self, Chromosome member):
        """check the member's chromosome that is out of bound?"""
        cdef int i
        for i in range(self.D):
            if member.v[i] > self.ub[i] or member.v[i] < self.lb[i]:
                return True
        return False

    cdef inline void generation_process(self):
        self.gen += 1

        cdef int i
        cdef Chromosome tmp, baby
        for i in range(self.NP):
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

        if self.gen % self.rpt == 0:
            self.report()

    cpdef tuple run(self):
        """Run the algorithm."""
        self.time_start = time()
        # initialize the member's chromosome
        self.initialize()
        # find the best one (smallest fitness value)
        cdef Chromosome tmp = self.find_best()
        # copy to last_best
        self.last_best.assign(tmp)
        # report status
        self.report()

        # the evolution journey is begin ...
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

        # the evolution journey is done, report the final status
        self.report()
        return self.func.result(self.last_best.v), self.fitness_time
