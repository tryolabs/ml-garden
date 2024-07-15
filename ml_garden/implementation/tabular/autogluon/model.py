import logging

import pandas as pd
from autogluon.tabular import TabularPredictor

from ml_garden.core.constants import Task
from ml_garden.core.model import Model

logger = logging.getLogger(__file__)


class AutoGluon(Model):
    TASKS = [Task.REGRESSION, Task.CLASSIFICATION]

    def __init__(self, **params):
        self.params = params
        self.model: TabularPredictor = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, verbose=True) -> None:
        train_data = X.copy()
        train_data["label"] = y
        if eval_set is not None and len(eval_set) > 0:
            eval_X, eval_y = eval_set[0]
            val_data = eval_X.copy()
            val_data["label"] = eval_y
            self.model = TabularPredictor(label="label", verbosity=3 if verbose else 0).fit(
                train_data=train_data, tuning_data=val_data, **self.params
            )
        else:
            self.model = TabularPredictor(label="label", verbosity=3 if verbose else 0).fit(
                train_data=train_data, **self.params
            )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X)
        return predictions
