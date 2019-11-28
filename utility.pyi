# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import TypeVar, Tuple, List, Dict, Callable, Optional, Generic, Any
from numpy import ndarray

FVal = TypeVar('FVal')


class Objective(Generic[FVal]):

    """Objective function base class."""

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
        """Show the result."""
        ...


class AlgorithmBase(Generic[FVal]):

    """Algorithm base class."""

    func: Objective[FVal]

    def __class_getitem__(cls, item):
        # PEP 560
        raise NotImplemented

    def __init__(
        self,
        func: Objective[FVal],
        settings: Dict[str, Any],
        progress_fun: Optional[Callable[[int, str], None]] = None,
        interrupt_fun: Optional[Callable[[], bool]] = None
    ):
        ...

    def history(self) -> List[Tuple[int, float, float]]:
        """Return the history of the process."""
        ...

    def run(self) -> FVal:
        """Init and run GA for max_gen times."""
        ...
