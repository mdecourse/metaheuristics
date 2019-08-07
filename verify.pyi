# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Callable, Optional, Any
import numpy as np


class Verification(ABC):

    """Verification function base class."""

    @abstractmethod
    def result(self, v: np.ndarray) -> Any:
        """Show the result."""
        ...


class AlgorithmBase(ABC):

    """Algorithm base class."""

    def __init__(
        self,
        func: Verification,
        settings: Dict[str, Any],
        progress_fun: Optional[Callable[[int, str], None]] = None,
        interrupt_fun: Optional[Callable[[], bool]] = None
    ):
        ...

    def run(self) -> Tuple[Any, List[Tuple[int, float, float]]]:
        ...
