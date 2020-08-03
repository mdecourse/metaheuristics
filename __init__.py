# -*- coding: utf-8 -*-

"""Kernel of Metaheuristic Algorithm."""

__author__ = "Yuan Chang"
__copyright__ = "Copyright (C) 2016-2020"
__license__ = "AGPL"
__email__ = "pyslvs@gmail.com"

from typing import Mapping, Union, Type
from enum import unique, Enum
from .utility import ObjFunc, AlgorithmBase
from .rga import Genetic
from .firefly import Firefly
from .de import Differential
from .tlbo import TeachingLearning


@unique
class AlgorithmType(str, Enum):
    """Enum type of algorithms."""
    RGA = "Real-coded Genetic Algorithm"
    Firefly = "Firefly Algorithm"
    DE = "Differential Evolution"
    TLBO = "Teaching Learning Based Optimization"


ALGORITHM: Mapping[AlgorithmType, Type[AlgorithmBase]] = {
    AlgorithmType.RGA: Genetic,
    AlgorithmType.Firefly: Firefly,
    AlgorithmType.DE: Differential,
    AlgorithmType.TLBO: TeachingLearning,
}
PARAMS: Mapping[AlgorithmType, Mapping[str, Union[int, float]]] = {
    AlgorithmType.RGA: {
        'nPop': 500,
        'pCross': 0.95,
        'pMute': 0.05,
        'pWin': 0.95,
        'bDelta': 5.,
    },
    AlgorithmType.Firefly: {
        'n': 80,
        'alpha': 0.01,
        'beta_min': 0.2,
        'gamma': 1.,
        'beta0': 1.,
    },
    AlgorithmType.DE: {
        'strategy': 1,
        'NP': 400,
        'F': 0.6,
        'CR': 0.9,
    },
    AlgorithmType.TLBO: {
        'class_size': 50,
    }
}
DEFAULT_PARAMS: Mapping[str, int] = {'max_gen': 1000, 'report': 50}
