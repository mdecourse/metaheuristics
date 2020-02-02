# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import TypeVar, Tuple, List, Dict, Callable, Optional, Generic, Any
from numpy import ndarray, double

FVal = TypeVar('FVal')


class Objective(Generic[FVal]):

    """Objective function base class.

    It is used to build the objective function for Metaheuristic Random Algorithms.
    """

    @abstractmethod
    def fitness(self, v: ndarray) -> double:
        """(`cdef` function) Return the fitness from the variable list `v`.
        This function will be directly called in the algorithms.
        """
        ...

    @abstractmethod
    def get_upper(self) -> ndarray:
        """Return upper bound."""
        ...

    @abstractmethod
    def get_lower(self) -> ndarray:
        """Return lower bound."""
        ...

    @abstractmethod
    def result(self, v: ndarray) -> FVal:
        """Return the result from the variable list `v`."""
        ...


class AlgorithmBase(Generic[FVal]):

    """Algorithm base class.

    It is used to build the Metaheuristic Random Algorithms.
    """

    func: Objective[FVal]

    def __class_getitem__(cls, item):
        # PEP 560
        raise NotImplemented

    @abstractmethod
    def __init__(
        self,
        func: Objective[FVal],
        settings: Dict[str, Any],
        progress_fun: Optional[Callable[[int, str], None]] = None,
        interrupt_fun: Optional[Callable[[], bool]] = None
    ):
        """The argument `func` is a object inherit from [Objective],
        and all abstract methods should be implemented.

        The format of argument `settings` can be customized.

        The argument `progress_fun` will be called when update progress,
        and the argument `interrupt_fun` will check the interrupt status from GUI or subprocess.
        """
        ...

    def history(self) -> List[Tuple[int, float, float]]:
        """Return the history of the process.

        The first value is generation (iteration);
        the second value is fitness;
        the third value is time in second.
        """
        ...

    @abstractmethod
    def run(self) -> FVal:
        """Run and return the result and convergence history.

        The first place of `return` is came from calling [`Objective.result()`](#objectiveresult).

        The second place of `return` is a list of generation data,
        which type is `Tuple[int, float, float]]`.
        The first of them is generation,
        the second is fitness, and the last one is time in second.
        """
        ...
