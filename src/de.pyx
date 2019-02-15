# -*- coding: utf-8 -*-
# cython: language_level=3

"""Differential Evolution.

author: Yuan Chang
copyright: Copyright (C) 2016-2019
license: AGPL
email: pyslvs@gmail.com
"""

from time import time
from numpy import array as np_array
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
cdef class Differential:

    """Algorithm class."""

    cdef Limit option
    cdef int strategy, D, NP, max_gen, max_time, rpt, gen, r1, r2, r3, r4, r5
    cdef double F, CR, min_fit, time_start
    cdef ndarray lb, ub, pop
    cdef Verification func
    cdef object progress_fun, interrupt_fun
    cdef Chromosome last_best, current_best
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
        # dimension of question
        self.D = self.func.length()
        # strategy 1~10, choice what strategy to generate new member in temporary
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
            raise ValueError("Please give 'max_gen', 'min_fit' or 'max_time' limit.")
        # Report function
        self.rpt = settings.get('report', 0)
        self.progress_fun = progress_fun
        self.interrupt_fun = interrupt_fun

        # check parameter is set properly
        self.check_parameter()

        # generation pool, depend on population size
        self.pop = ndarray(self.NP, dtype=object)
        cdef int i
        for i in range(self.NP):
            self.pop[i] = Chromosome.__new__(Chromosome, self.D)

        # last generation best member
        self.last_best = Chromosome.__new__(Chromosome, self.D)
        # current best member
        self.current_best = Chromosome.__new__(Chromosome, self.D)

        # the generation count
        self.gen = 0

        # the vector
        self.r1 = 0
        self.r2 = 0
        self.r3 = 0
        self.r4 = 0
        self.r5 = 0

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
        for i in range(self.NP):
            for j in range(self.D):
                self.pop[i].v[j] = rand_v(self.lb[j], self.ub[j])
            self.pop[i].f = self.func.fitness(self.pop[i].v)

    cdef inline Chromosome find_best(self):
        """Find member that have minimum fitness value from pool."""
        cdef int index = 0
        cdef double f = self.pop[0].f

        cdef int i
        cdef Chromosome c
        for i in range(len(self.pop)):
            c = self.pop[i]
            if c.f < f:
                index = i
                f = c.f
        return self.pop[index]

    cdef inline void generate_random_vector(self, int i):
        """Generate new vector."""
        self.r1 = self.r2 = self.r3 = self.r4 = self.r5 = i
        cdef set compare_set = {i}
        cdef int np = self.NP - 1
        while self.r1 in compare_set:
            self.r1 = <int>rand_v(0, np)
        compare_set.add(self.r1)
        while self.r2 in compare_set:
            self.r2 = <int>rand_v(0, np)
        compare_set.add(self.r2)
        while self.r3 in compare_set:
            self.r3 = <int>rand_v(0, np)
        compare_set.add(self.r3)
        while self.r4 in compare_set:
            self.r4 = <int>rand_v(0, np)
        compare_set.add(self.r5)
        while self.r5 in compare_set:
            self.r5 = <int>rand_v(0, np)

    cdef inline Chromosome recombination(self, int i):
        """use new vector, recombination the new one member to tmp."""
        cdef Chromosome tmp = Chromosome.__new__(Chromosome, self.D)
        tmp.assign(self.pop[i])
        cdef int n = <int>rand_v(0, self.D - 1)
        cdef int l_v = 0
        if self.strategy == 1:
            while True:
                tmp.v[n] = self.last_best.v[n] + self.F * (self.pop[self.r2].v[n] - self.pop[self.r3].v[n])
                n = (n + 1) % self.D
                l_v += 1
                if not (rand_v() < self.CR and l_v < self.D):
                    break
        elif self.strategy == 2:
            while True:
                tmp.v[n] = self.pop[self.r1].v[n] + self.F * (self.pop[self.r2].v[n] - self.pop[self.r3].v[n])
                n = (n + 1) % self.D
                l_v += 1
                if not (rand_v() < self.CR and l_v < self.D):
                    break
        elif self.strategy == 3:
            while True:
                tmp.v[n] = tmp.v[n] + self.F * (self.last_best.v[n] - tmp.v[n]) + self.F*(self.pop[self.r1].v[n] - self.pop[self.r2].v[n])
                n = (n + 1) % self.D
                l_v += 1
                if not (rand_v() < self.CR and l_v < self.D):
                    break
        elif self.strategy == 4:
            while True:
                tmp.v[n] = self.last_best.v[n] + (self.pop[self.r1].v[n] + self.pop[self.r2].v[n] - self.pop[self.r3].v[n] - self.pop[self.r4].v[n]) * self.F
                n = (n + 1) % self.D
                l_v += 1
                if not (rand_v() < self.CR and l_v < self.D):
                    break
        elif self.strategy == 5:
            while True:
                tmp.v[n] = self.pop[self.r5].v[n] + (self.pop[self.r1].v[n] + self.pop[self.r2].v[n] - self.pop[self.r3].v[n] - self.pop[self.r4].v[n]) * self.F
                n = (n + 1) % self.D
                l_v += 1
                if not (rand_v() < self.CR and l_v < self.D):
                    break
        elif self.strategy == 6:
            for l_v in range(self.D):
                if rand_v() < self.CR or l_v == self.D - 1:
                    tmp.v[n] = self.last_best.v[n] + self.F * (self.pop[self.r2].v[n] - self.pop[self.r3].v[n])
                n = (n + 1) % self.D
        elif self.strategy == 7:
            for l_v in range(self.D):
                if rand_v() < self.CR or l_v == self.D - 1:
                    tmp.v[n] = self.pop[self.r1].v[n] + self.F * (self.pop[self.r2].v[n] - self.pop[self.r3].v[n])
                n = (n + 1) % self.D
        elif self.strategy == 8:
            for l_v in range(self.D):
                if rand_v() < self.CR or l_v == self.D - 1:
                    tmp.v[n] = tmp.v[n] + self.F*(self.last_best.v[n] - tmp.v[n]) + self.F*(self.pop[self.r1].v[n] - self.pop[self.r2].v[n])
                n = (n + 1) % self.D
        elif self.strategy == 9:
            for l_v in range(self.D):
                if rand_v() < self.CR or l_v == self.D - 1:
                    tmp.v[n] = self.last_best.v[n] + (self.pop[self.r1].v[n] + self.pop[self.r2].v[n] - self.pop[self.r3].v[n] - self.pop[self.r4].v[n]) * self.F
                n = (n + 1) % self.D
        else:
            for l_v in range(self.D):
                if rand_v() < self.CR or l_v == self.D - 1:
                    tmp.v[n] = self.pop[self.r5].v[n] + (self.pop[self.r1].v[n] + self.pop[self.r2].v[n] - self.pop[self.r3].v[n] - self.pop[self.r4].v[n]) * self.F
                n = (n + 1) % self.D
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
            # is not out of bound, that mean it's qualify of enviorment
            # then evaluate the one
            tmp.f = self.func.fitness(tmp.v)
            # if temporary one is better than origin(fitness value is smaller)
            if tmp.f <= self.pop[i].f:
                # copy the temporary one to origin member
                baby = self.pop[i]
                baby.assign(tmp)
                # check the temporary one is better than the current_best
                if tmp.f < self.current_best.f:
                    # copy the temporary one to current_best
                    self.current_best.assign(tmp)
        # copy the current_best to last_best
        self.last_best.assign(self.current_best)
        # if report generation is set, report
        if self.rpt:
            if self.gen % self.rpt == 0:
                self.report()
        else:
            if self.gen % 10 == 0:
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
        # copy to current_best
        self.current_best.assign(tmp)
        # report status
        self.report()

        # the evolution journey is begin ...
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

        # the evolution journey is done, report the final status
        self.report()
        return self.func.result(self.last_best.v), self.fitness_time
