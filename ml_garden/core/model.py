from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import pandas as pd

from ml_garden.core.constants import Task

# ruff: noqa: N803 N806


class Model(ABC):
    """Base class for models."""

    TASKS: List[Task] = []

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
        *,
        verbose: Optional[bool] = True,
    ) -> None:
        """Abstract method for fitting the model."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Abstract method for making predictions."""
