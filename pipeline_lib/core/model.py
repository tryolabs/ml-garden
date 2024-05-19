from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import pandas as pd


class Model(ABC):
    """Base class for models."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
        verbose: Optional[bool] = True,
    ) -> None:
        """Abstract method for fitting the model."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Abstract method for making predictions."""
