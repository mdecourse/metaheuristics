# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False

"""Teaching Learning Based Optimization

author: Yuan Chang
copyright: Copyright (C) 2016-2020
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from cython.parallel cimport prange
from libc.math cimport round
from .utility cimport rand_v, rand_i, ObjFunc, Algorithm

ctypedef unsigned int uint


@cython.final
cdef class TeachingLearning(Algorithm):
    """The implementation of Teaching Learning Based Optimization."""

    def __cinit__(
        self,
        ObjFunc func,
        dict settings,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        """
        settings = {
            'class_size': int,
            'max_gen': int or 'min_fit': float or 'max_time': float,
            'report': int,
        }
        """
        self.pop_num = settings.get('class_size', 50)
        self.new_pop()

    cdef inline void initialize(self) nogil:
        """Initial population: Sorted students."""
        cdef uint i, j, s
        for i in prange(self.pop_num, num_threads=4, nogil=True):
            for s in range(self.dim):
                self.pool[i, s] = rand_v(self.func.lb[s], self.func.ub[s])
            self.fitness[i] = self.func.fitness(self.pool[i, :])
            self.set_best(i)

    cdef inline void teaching(self, uint i) nogil:
        """Teaching phase. The last best is the teacher."""
        cdef double tf = round(1 + rand_v())
        cdef uint s, j
        cdef double mean
        for s in range(self.dim):
            mean = 0
            for j in range(self.pop_num):
                mean += self.pool[j, s]
            mean /= self.dim
            self.tmp[s] = self.pool[i, s] + rand_v(1, self.dim) * (
                self.best[s] - tf * mean)
            if self.tmp[s] < self.func.lb[s]:
                self.tmp[s] = self.func.lb[s]
            elif self.tmp[s] > self.func.ub[s]:
                self.tmp[s] = self.func.ub[s]
        cdef double f_new = self.func.fitness(self.tmp)
        if f_new < self.fitness[i]:
            self.pool[i, :] = self.tmp
            self.fitness[i] = f_new
        self.set_best(i)

    cdef inline void learning(self, uint i) nogil:
        """Learning phase."""
        cdef uint j = rand_i(self.pop_num - 1)
        if j >= i:
            j += 1
        cdef uint s
        cdef double diff
        for s in range(self.dim):
            if self.fitness[j] < self.fitness[i]:
                diff = self.pool[i, s] - self.pool[j, s]
            else:
                diff = self.pool[j, s] - self.pool[i, s]
            self.tmp[s] = self.pool[i, s] + diff * rand_v(1, self.dim)
            if self.tmp[s] < self.func.lb[s]:
                self.tmp[s] = self.func.lb[s]
            elif self.tmp[s] > self.func.ub[s]:
                self.tmp[s] = self.func.ub[s]
        cdef double f_new = self.func.fitness(self.tmp)
        if f_new < self.fitness[i]:
            self.pool[i, :] = self.tmp
            self.fitness[i] = f_new
        self.set_best(i)

    cdef inline void generation_process(self) nogil:
        """The process of each generation."""
        cdef uint i
        for i in range(self.pop_num):
            self.teaching(i)
            self.learning(i)
