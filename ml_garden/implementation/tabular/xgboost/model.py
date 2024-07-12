import logging

import pandas as pd
import xgboost as xgb

from ml_garden.core.model import Model

logger = logging.getLogger(__file__)


class XGBoostRegressor(Model):
    TASK = "regression"

    def __init__(self, **params):
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, verbose=True) -> None:
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)


class XGBoostClassifier(Model):
    TASK = "classification"

    def __init__(self, **params):
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, verbose=True) -> None:
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

    def predict(self, X: pd.DataFrame) -> pd.Series:
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

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
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
