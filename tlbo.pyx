# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False

"""Teaching Learning Based Optimization

author: Yuan Chang
copyright: Copyright (C) 2016-2020
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from numpy cimport ndarray
from numpy import zeros, float64 as np_float
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

    cdef inline void initialize(self):
        """Initial population: Sorted students."""
        cdef uint end = self.dim + 1
        cdef ndarray[double, ndim=2] s = zeros((self.pop_num, end), dtype=np_float)
        cdef uint i, j
        for i in range(self.pop_num):
            for j in range(self.dim):
                s[i, j] = rand_v(self.func.lb[j], self.func.ub[j])
            s[i, end - 1] = self.func.fitness(s[i, :end - 1])
        s = s[s[:, end - 1].argsort()][::-1]
        cdef double[:] tmp
        for i in range(self.pop_num):
            tmp = s[i, :end - 1]
            self.pool[i, :] = tmp
            self.fitness[i] = s[i, end - 1]
        self.set_best(self.pop_num - 1)

    cdef inline void teaching(self, uint index):
        """Teaching phase. The last best is the teacher."""
        cdef double[:] tmp = self.make_tmp()
        cdef double tf = round(1 + rand_v())
        cdef uint i, j
        cdef double mean
        for i in range(self.dim):
            if self.state_check():
                return
            mean = 0
            for j in range(self.pop_num):
                mean += self.pool[j, i]
            mean /= self.dim
            tmp[i] = self.pool[index, i] + rand_v(1, self.dim) * (
                self.best[i] - tf * mean)
            if tmp[i] < self.func.lb[i]:
                tmp[i] = self.func.lb[i]
            elif tmp[i] > self.func.ub[i]:
                tmp[i] = self.func.ub[i]
        cdef double f_new = self.func.fitness(tmp)
        if f_new < self.fitness[index]:
            self.pool[index, :] = tmp
            self.fitness[index] = f_new
        if self.fitness[index] < self.best_f:
            self.set_best(index)

    cdef inline void learning(self, uint index):
        """Learning phase."""
        cdef uint cmp_index = rand_i(self.pop_num - 1)
        if cmp_index >= index:
            cmp_index += 1
        cdef double[:] v = self.make_tmp()
        cdef uint s
        cdef double diff
        for s in range(self.dim):
            if self.state_check():
                return
            if self.fitness[cmp_index] < self.fitness[index]:
                diff = self.pool[index, s] - self.pool[cmp_index, s]
            else:
                diff = self.pool[cmp_index, s] - self.pool[index, s]
            v[s] = self.pool[index, s] + diff * rand_v(1, self.dim)
            if v[s] < self.func.lb[s]:
                v[s] = self.func.lb[s]
            elif v[s] > self.func.ub[s]:
                v[s] = self.func.ub[s]
        cdef double f_new = self.func.fitness(v)
        if f_new < self.fitness[index]:
            self.pool[index, :] = v
            self.fitness[index] = f_new
        if self.fitness[index] < self.best_f:
            self.set_best(index)

    cdef inline bint state_check(self) nogil:
        """Check status."""
        if self.progress_fun is not None:
            with gil:
                self.progress_fun(self.func.gen, f"{self.best_f:.04f}")
        if self.interrupt_fun is not None:
            with gil:
                if self.interrupt_fun():
                    return True
        return False

    cdef inline void generation_process(self):
        """The process of each generation."""
        cdef uint i
        for i in range(self.pop_num):
            if self.state_check():
                break
            self.teaching(i)
            self.learning(i)
