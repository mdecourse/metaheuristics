# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True

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
    Objective,
    AlgorithmBase,
)

ctypedef unsigned int uint


@cython.final
cdef class TeachingLearning(AlgorithmBase):
    """The implementation of Teaching Learning Based Optimization."""
    cdef uint class_size
    cdef Chromosome[:] students

    def __cinit__(
        self,
        Objective func,
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
        self.class_size = settings.get('class_size', 50)
        self.students = Chromosome.new_pop(self.dim, self.class_size)

    cdef inline void initialize(self):
        """Initial population: Sorted students."""
        cdef ndarray[double, ndim=2] s = zeros((self.class_size, self.dim + 1), dtype=np_float)
        cdef uint i, j
        for i in range(self.class_size):
            for j in range(self.dim):
                s[i, j] = rand_v(self.lb[j], self.ub[j])
            s[i, -1] = self.func.fitness(s[i, :-1])
        s = s[s[:, -1].argsort()][::-1]
        for i in range(self.class_size):
            self.students[i].v = s[i, :-1]
            self.students[i].f = s[i, -1]
        self.last_best.assign(self.students[-1])

    cdef inline void teaching(self, uint index):
        """Teaching phase. The last best is the teacher."""
        cdef Chromosome student = self.students[index]
        cdef double[:] v = zeros(self.dim, dtype=np_float)
        cdef double tf = round(1 + rand_v())
        cdef uint i, j
        cdef double mean
        cdef Chromosome tmp
        for i in range(self.dim):
            if self.state_check():
                return
            mean = 0
            for j in range(self.class_size):
                mean += self.students[j].v[i]
            mean /= self.dim
            v[i] = student.v[i] + rand_v(1, self.dim) * (self.last_best.v[i] - tf * mean)
            if v[i] < self.lb[i]:
                v[i] = self.lb[i]
            elif v[i] > self.ub[i]:
                v[i] = self.ub[i]
        cdef double f_new = self.func.fitness(v)
        if f_new < student.f:
            student.v[:] = v
            student.f = f_new
        if student.f < self.last_best.f:
            self.last_best.assign(student)

    cdef inline void learning(self, uint index):
        """Learning phase."""
        cdef Chromosome student_a = self.students[index]
        cdef uint cmp_index = rand_i(self.class_size - 1)
        if cmp_index >= index:
            cmp_index += 1
        cdef Chromosome student_b = self.students[cmp_index]
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
            if v[i] < self.lb[i]:
                v[i] = self.lb[i]
            elif v[i] > self.ub[i]:
                v[i] = self.ub[i]
        cdef double f_new = self.func.fitness(v)
        if f_new < student_a.f:
            student_a.v[:] = v
            student_a.f = f_new
        if student_a.f < self.last_best.f:
            self.last_best.assign(student_a)

    cdef inline bint state_check(self):
        """Check status."""
        if self.progress_fun is not None:
            self.progress_fun(self.gen, f"{self.last_best.f:.04f}")
        if (self.interrupt_fun is not None) and self.interrupt_fun():
            return True
        return False

    cdef inline void generation_process(self):
        """The process of each generation."""
        cdef uint i
        for i in range(self.class_size):
            if self.state_check():
                break
            self.teaching(i)
            self.learning(i)
