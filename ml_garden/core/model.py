from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import pandas as pd

from ml_garden.core.constants import Task


class Model(ABC):
    """Base class for models."""

    TASKS: List[Task] = []

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

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict class probabilities with the trained model.

        Parameters
        ----------
        X : pd.DataFrame
            Features to make probability predictions on.

        Returns
        -------
        pd.DataFrame
            Predicted class probabilities for the input features.
        """
        pass
