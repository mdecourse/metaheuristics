# -*- coding: utf-8 -*-
# cython: language_level=3

"""Differential Evolution.

author: Yuan Chang
copyright: Copyright (C) 2016-2018
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
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
cdef class Differential:

    """Algorithm class."""

    cdef limit option
    cdef int strategy, D, NP, maxGen, maxTime, rpt, gen, r1, r2, r3, r4, r5
    cdef double F, CR, minFit, time_start
    cdef np.ndarray lb, ub, pop
    cdef Verification func
    cdef object progress_fun, interrupt_fun
    cdef Chromosome lastgenbest, currentbest
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
            'strategy',
            'NP',
            'F',
            'CR',
            'maxGen', 'minFit' or 'maxTime',
            'report'
        }
        """
        # object function, or environment
        self.func = func
        # dimension of question
        self.D = self.func.get_nParm()
        # strategy 1~10, choice what strategy to generate new member in temporary
        self.strategy = settings['strategy']
        # population size
        # To start off NP = 10*D is a reasonable choice. Increase NP if misconvergence
        self.NP = settings['NP']
        # weight factor
        # F is usually between 0.5 and 1 (in rare cases > 1)
        self.F = settings['F']
        # crossover possible
        # CR in [0,1]
        self.CR = settings['CR']
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

        # check parameter is set properly
        self.check_parameter()

        # generation pool, depend on population size
        self.pop = np.ndarray(self.NP, dtype=np.object)
        cdef int i
        for i in range(self.NP):
            self.pop[i] = Chromosome(self.D)

        # last generation best member
        self.lastgenbest = Chromosome(self.D)
        # current best member
        self.currentbest = Chromosome(self.D)

        # the generation count
        self.gen = 0

        # the vector
        self.r1 = 0
        self.r2 = 0
        self.r3 = 0
        self.r4 = 0
        self.r5 = 0

        # setup benchmark
        self.time_start = time()
        self.fitnessTime = []

    cdef inline void check_parameter(self):
        """Check parameter is set properly."""
        if self.D <= 0:
            raise Exception('D should be integer and larger than 0')
        if self.NP <= 0:
            raise Exception('NP should be integer and larger than 0')
        if not (0 <= self.CR <= 1):
            raise Exception('CR should be [0,1]')
        if self.strategy not in range(10):
            raise Exception('strategy should be [0,9]')
        for lower, upper in zip(self.lb, self.ub):
            if lower > upper:
                raise Exception('upper bound should be larger than lower bound')

    cdef inline void initialize(self):
        """Initial population."""
        cdef int i, j
        for i in range(self.NP):
            for j in range(self.D):
                self.pop[i].v[j] = self.lb[j] + rand_v() * (self.ub[j] - self.lb[j])
            self.pop[i].f = self.evaluate(self.pop[i])

    cdef inline double evaluate(self, Chromosome member):
        """Evalute the member in environment."""
        return self.func(member.v)

    cdef inline Chromosome find_best(self):
        """Find member that have minimum fitness value from pool."""
        cdef int i
        cdef int index = 0
        cdef Chromosome chromosome
        cdef double f = self.pop[0].f
        for i in range(len(self.pop)):
            chromosome = self.pop[i]
            if chromosome.f < f:
                index = i
                f = chromosome.f
        return self.pop[index]

    cdef inline void generate_random_vector(self, int i):
        """Generate new vector."""
        self.r1 = self.r2 = self.r3 = self.r4 = self.r5 = i
        cdef set compare_set = {i}
        while self.r1 in compare_set:
            self.r1 = int(rand_v() * self.NP)
        compare_set.add(self.r1)
        while self.r2 in compare_set:
            self.r2 = int(rand_v() * self.NP)
        compare_set.add(self.r2)
        while self.r3 in compare_set:
            self.r3 = int(rand_v() * self.NP)
        compare_set.add(self.r3)
        while self.r4 in compare_set:
            self.r4 = int(rand_v() * self.NP)
        compare_set.add(self.r5)
        while self.r5 in compare_set:
            self.r5 = int(rand_v() * self.NP)

    cdef inline Chromosome recombination(self, int i):
        """use new vector, recombination the new one member to tmp."""
        cdef Chromosome tmp = Chromosome(self.D)
        tmp.assign(self.pop[i])
        cdef int n = int(rand_v() * self.D)
        cdef int l_v = 0
        if self.strategy == 1:
            while True:
                tmp.v[n] = self.lastgenbest.v[n] + self.F * (self.pop[self.r2].v[n] - self.pop[self.r3].v[n])
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
                tmp.v[n] = tmp.v[n] + self.F * (self.lastgenbest.v[n] - tmp.v[n]) + self.F*(self.pop[self.r1].v[n] - self.pop[self.r2].v[n])
                n = (n + 1) % self.D
                l_v += 1
                if not (rand_v() < self.CR and l_v < self.D):
                    break
        elif self.strategy == 4:
            while True:
                tmp.v[n] = self.lastgenbest.v[n] + (self.pop[self.r1].v[n] + self.pop[self.r2].v[n] - self.pop[self.r3].v[n] - self.pop[self.r4].v[n]) * self.F
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
                    tmp.v[n] = self.lastgenbest.v[n] + self.F * (self.pop[self.r2].v[n] - self.pop[self.r3].v[n])
                n = (n + 1) % self.D
        elif self.strategy == 7:
            for l_v in range(self.D):
                if rand_v() < self.CR or l_v == self.D - 1:
                    tmp.v[n] = self.pop[self.r1].v[n] + self.F * (self.pop[self.r2].v[n] - self.pop[self.r3].v[n])
                n = (n + 1) % self.D
        elif self.strategy == 8:
            for l_v in range(self.D):
                if rand_v() < self.CR or l_v == self.D - 1:
                    tmp.v[n] = tmp.v[n] + self.F*(self.lastgenbest.v[n] - tmp.v[n]) + self.F*(self.pop[self.r1].v[n] - self.pop[self.r2].v[n])
                n = (n + 1) % self.D
        elif self.strategy == 9:
            for l_v in range(self.D):
                if rand_v() < self.CR or l_v == self.D - 1:
                    tmp.v[n] = self.lastgenbest.v[n] + (self.pop[self.r1].v[n] + self.pop[self.r2].v[n] - self.pop[self.r3].v[n] - self.pop[self.r4].v[n]) * self.F
                n = (n + 1) % self.D
        else:
            for l_v in range(self.D):
                if rand_v() < self.CR or l_v == self.D - 1:
                    tmp.v[n] = self.pop[self.r5].v[n] + (self.pop[self.r1].v[n] + self.pop[self.r2].v[n] - self.pop[self.r3].v[n] - self.pop[self.r4].v[n]) * self.F
                n = (n + 1) % self.D
        return tmp

    cdef inline void report(self):
        self.fitnessTime.append((self.gen, self.lastgenbest.f, time() - self.time_start))

    cdef inline bint over_bound(self, Chromosome member):
        """check the member's chromosome that is out of bound?"""
        cdef int i
        for i in range(self.D):
            if member.v[i] > self.ub[i] or member.v[i] < self.lb[i]:
                return True
        return False

    cdef inline void generation_process(self):
        cdef int i
        cdef Chromosome tmp
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
            tmp.f = self.evaluate(tmp)
            # if temporary one is better than origin(fitness value is smaller)
            if tmp.f <= self.pop[i].f:
                # copy the temporary one to origin member
                self.pop[i].assign(tmp)
                # check the temporary one is better than the currentbest
                if tmp.f < self.currentbest.f:
                    # copy the temporary one to currentbest
                    self.currentbest.assign(tmp)
        # copy the currentbest to lastgenbest
        self.lastgenbest.assign(self.currentbest)
        # if report generation is set, report
        if self.rpt:
            if self.gen % self.rpt == 0:
                self.report()
        else:
            if self.gen % 10 == 0:
                self.report()

    cpdef tuple run(self):
        """Run the algorithm ..."""
        # initialize the member's chromosome
        self.initialize()
        # find the best one (smallest fitness value)
        cdef Chromosome tmp = self.find_best()
        # copy to lastgenbest
        self.lastgenbest.assign(tmp)
        # copy to currentbest
        self.currentbest.assign(tmp)
        # report status
        self.report()
        # end initial step
        # the evolution journey is begin ...
        while True:
            self.gen += 1
            if self.option == maxGen:
                if 0 < self.maxGen < self.gen:
                    break
            elif self.option == minFit:
                if self.lastgenbest.f <= self.minFit:
                    break
            elif self.option == maxTime:
                if 0 < self.maxTime <= time() - self.time_start:
                    break
            self.generation_process()
            # progress
            if self.progress_fun:
                self.progress_fun(self.gen, f"{self.lastgenbest.f:.04f}")
            # interrupt
            if self.interrupt_fun and self.interrupt_fun():
                break
        # the evolution journey is done, report the final status
        self.report()
        return self.func.result(self.lastgenbest.v), self.fitnessTime
