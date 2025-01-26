import logging
from typing import Any, List, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor

from ml_garden.core.constants import Task
from ml_garden.core.model import Model

logger = logging.getLogger(__file__)


class AutoGluon(Model):
    TASKS = [Task.REGRESSION, Task.CLASSIFICATION, Task.QUANTILE_REGRESSION]

    def __init__(
        self,
        autogluon_create_params: dict[str, Any],
        autogluon_fit_params: dict[str, Any],
    ):
        self.autogluon_create_params = autogluon_create_params
        self.autogluon_fit_params = autogluon_fit_params
        self.model: Optional[TabularPredictor] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
        verbose: Optional[bool] = True,
    ) -> None:
        # AutoGluon performs it's own splits so we need to concatenate the train and validation data
        dfs = [pd.concat([X, y], axis=1)]
        if eval_set is not None:
            dfs.extend(
                [pd.concat([X_eval, y_eval], axis=1) for X_eval, y_eval in eval_set]
            )
        concatenated_df = pd.concat(dfs, axis=0)
        if not verbose:
            self.autogluon_create_params["verbosity"] = 0

        # label = self.autogluon_create_params["label"]
        # train_data = X.copy()
        # train_data[label] = y
        # if eval_set is not None and len(eval_set) > 0:
        #     eval_X, eval_y = eval_set[0]
        #     val_data = eval_X.copy()
        #     val_data[label] = eval_y
        #     self.model = TabularPredictor(
        #         **self.autogluon_create_params,
        #     ).fit(
        #         train_data=train_data, tuning_data=val_data, **self.autogluon_fit_params
        #     )
        # else:
        self.model = TabularPredictor(
            **self.autogluon_create_params,
        ).fit(train_data=concatenated_df, **self.autogluon_fit_params)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X, as_pandas=True)
        return pd.Series(predictions, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model.problem_type == "regression":
            raise ValueError("predict_proba is not available for regression tasks.")
        probabilities = self.model.predict_proba(X, as_pandas=True)
        return pd.DataFrame(probabilities, index=X.index)
