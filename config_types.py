# -*- coding: utf-8 -*-

from typing import TypedDict


class AlgorithmConfig(TypedDict, total=False):
    max_gen: int
    min_fit: float
    max_time: float
    slow_down: float
    report: int
    parallel: bool


class DEConfig(AlgorithmConfig):
    strategy: int
    NP: int
    F: float
    CR: float


class GAConfig(AlgorithmConfig):
    pop_num: int
    cross: float
    mutate: float
    win: float
    delta: float


class FAConfig(AlgorithmConfig):
    n: int
    alpha: float
    beta_min: float
    beta0: float
    gamma: float


class TOBLConfig(AlgorithmConfig):
    class_size: int
