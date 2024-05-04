from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
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

    def save(self, path: str) -> None:
        """Save the model."""
        if not path.endswith(".joblib"):
            raise ValueError("The path must end with .joblib")
        joblib.dump(self, path)

    @classmethod
    def from_file(cls, path: str) -> "Model":
        """Load the model from a .joblib file."""
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.endswith(".joblib"):
            raise ValueError("The path must end with .joblib")

        return joblib.load(path)
