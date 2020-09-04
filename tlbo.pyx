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
from .utility cimport (
    rand_v,
    rand_i,
    Chromosome,
    ObjFunc,
    Algorithm,
)

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
        self.pool = Chromosome.new_pop(self.dim, self.pop_num)

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
        for i in range(self.pop_num):
            self.pool[i].v = s[i, :end - 1]
            self.pool[i].f = s[i, end - 1]
        self.last_best.assign(self.pool[self.pop_num - 1])

    cdef inline void teaching(self, uint index):
        """Teaching phase. The last best is the teacher."""
        cdef Chromosome student = self.pool[index]
        cdef double[:] v = zeros(self.dim, dtype=np_float)
        cdef double tf = round(1 + rand_v())
        cdef uint i, j
        cdef double mean
        cdef Chromosome tmp
        for i in range(self.dim):
            if self.state_check():
                return
            mean = 0
            for j in range(self.pop_num):
                mean += self.pool[j].v[i]
            mean /= self.dim
            v[i] = student.v[i] + rand_v(1, self.dim) * (self.last_best.v[i] - tf * mean)
            if v[i] < self.func.lb[i]:
                v[i] = self.func.lb[i]
            elif v[i] > self.func.ub[i]:
                v[i] = self.func.ub[i]
        cdef double f_new = self.func.fitness(v)
        if f_new < student.f:
            student.v[:] = v
            student.f = f_new
        if student.f < self.last_best.f:
            self.last_best.assign(student)

    cdef inline void learning(self, uint index):
        """Learning phase."""
        cdef Chromosome student_a = self.pool[index]
        cdef uint cmp_index = rand_i(self.pop_num - 1)
        if cmp_index >= index:
            cmp_index += 1
        cdef Chromosome student_b = self.pool[cmp_index]
        cdef double[:] v = zeros(self.dim, dtype=np_float)
        cdef uint i
        cdef double diff
        for i in range(self.dim):
            if self.state_check():
                return
            if student_b.f < student_a.f:
                diff = student_a.v[i] - student_b.v[i]
            else:
                diff = student_b.v[i] - student_a.v[i]
            v[i] = student_a.v[i] + diff * rand_v(1, self.dim)
            if v[i] < self.func.lb[i]:
                v[i] = self.func.lb[i]
            elif v[i] > self.func.ub[i]:
                v[i] = self.func.ub[i]
        cdef double f_new = self.func.fitness(v)
        if f_new < student_a.f:
            student_a.v[:] = v
            student_a.f = f_new
        if student_a.f < self.last_best.f:
            self.last_best.assign(student_a)

    cdef inline bint state_check(self):
        """Check status."""
        if self.progress_fun is not None:
            self.progress_fun(self.func.gen, f"{self.last_best.f:.04f}")
        if (self.interrupt_fun is not None) and self.interrupt_fun():
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
