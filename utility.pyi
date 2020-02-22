# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import TypeVar, Tuple, List, Dict, Callable, Optional, Generic, Any
from numpy import ndarray, double

FVal = TypeVar('FVal')


class Objective(Generic[FVal]):

    @abstractmethod
    def fitness(self, v: ndarray) -> double:
        """(`cdef` function) Return the fitness from the variable list `v`.
        This function will be directly called in the algorithms.
        """
        ...

    @abstractmethod
    def get_upper(self) -> ndarray:
        ...

    @abstractmethod
    def get_lower(self) -> ndarray:
        ...

    @abstractmethod
    def result(self, v: ndarray) -> FVal:
        ...


class AlgorithmBase(Generic[FVal]):

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
        ...

    def run(self) -> FVal:
        ...
