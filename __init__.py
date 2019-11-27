# -*- coding: utf-8 -*-

"""Kernel of Metaheuristic Algorithm."""

from enum import unique, Enum
from .utility import Objective, AlgorithmBase
from .rga import Genetic
from .firefly import Firefly
from .de import Differential
from .tlbo import TeachingLearning

__all__ = [
    'Objective',
    'AlgorithmBase',
    'Genetic',
    'Firefly',
    'Differential',
    'TeachingLearning',
    'AlgorithmType',
    'PARAMS',
    'DEFAULT_PARAMS',
]


@unique
class AlgorithmType(Enum):
    """Enum type of algorithms."""

    RGA = "Real-coded Genetic Algorithm"
    Firefly = "Firefly Algorithm"
    DE = "Differential Evolution"
    TLBO = "Teaching Learning Based Optimization"

    def __str__(self) -> str:
        return str(self.value)


PARAMS = {
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
DEFAULT_PARAMS = {'max_gen': 1000, 'report': 50}
