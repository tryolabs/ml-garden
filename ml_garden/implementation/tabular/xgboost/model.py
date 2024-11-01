import logging
from typing import Any, List, Optional, Tuple

import pandas as pd
import xgboost as xgb

from ml_garden.core.constants import Task
from ml_garden.core.model import Model

logger = logging.getLogger(__name__)


class XGBoostRegressor(Model):
    TASKS = [Task.REGRESSION]

    def __init__(self, **params: dict[str, Any]) -> None:
        self.model = xgb.XGBRegressor(**params)

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
        *,
        verbose: bool = True,
    ) -> None:
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)

    def predict(self, X: pd.DataFrame) -> pd.Series:  # noqa: N803
        return self.model.predict(X)


class XGBoostClassifier(Model):
    TASKS = [Task.CLASSIFICATION]

    def __init__(self, **params: dict[str, Any]) -> None:
        self.model = xgb.XGBClassifier(**params)

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
        *,
        verbose: bool = True,
    ) -> None:
        """
        Train the XGBoost classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Training labels.
        eval_set : list of (X, y), optional
            A list of (X, y) tuple pairs to use as validation sets.
        verbose : bool, default True
            If True, print evaluation messages to the console.
        """
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)

    def predict(self, X: pd.DataFrame) -> pd.Series:  # noqa: N803
        """
        Make predictions with the trained XGBoost classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Features to make predictions on.

        Returns
        -------
        pd.Series
            Predictions for the input features.
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """
        Predict class probabilities with the trained XGBoost classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Features to make probability predictions on.

        Returns
        -------
        pd.DataFrame
            Predicted class probabilities for the input features.
        """
        return self.model.predict_proba(X)
