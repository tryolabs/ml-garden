import json
import logging
from typing import Optional, Tuple, Type

import optuna
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)

from pipeline_lib.core import DataContainer
from pipeline_lib.core.model import Model
from pipeline_lib.core.steps.base import PipelineStep


class OptunaOptimizer:
    def __init__(self, optuna_params: dict, logger: logging.Logger) -> None:
        self.optuna_params = optuna_params
        self.logger = logger

    def optimize(
        self, X_train, y_train, X_valid, y_valid, model_class: Type[Model], model_params: dict
    ) -> dict:
        def objective(trial):
            param = self._create_trial_params(trial)
            param.update(model_params)

            model = model_class(**param)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            preds = model.predict(X_valid)

            objective_metric = self.optuna_params.get("objective_metric", "mean_absolute_error")
            error = self._calculate_error(y_valid, preds, objective_metric)
            return error

        study = self._create_study()
        study.optimize(objective, n_trials=self.optuna_params.get("trials", 20))

        best_params = study.best_params
        return best_params

    def _create_trial_params(self, trial) -> dict:
        param = {}
        for key, value in self.optuna_params.get("search_space", {}).items():
            if isinstance(value, dict):
                suggest_func = getattr(trial, f"suggest_{value['type']}")
                kwargs = value.get("kwargs", {})
                param[key] = suggest_func(key, *value["args"], **kwargs)
            else:
                param[key] = value
        return param

    def _create_study(self) -> optuna.Study:
        study_name = self.optuna_params.get("study_name")
        storage = self.optuna_params.get("storage", "sqlite:///db.sqlite3")
        load_if_exists = self.optuna_params.get("load_if_exists", False)

        self.logger.info(
            f"Creating Optuna study with parameters: \n {json.dumps(self.optuna_params, indent=4)}"
        )

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
            direction="minimize",
        )
        return study

    @staticmethod
    def _calculate_error(y_true, y_pred, metric):
        metrics = {
            "mae": mean_absolute_error,
            "mse": mean_squared_error,
            "rmse": root_mean_squared_error,
        }

        if metric not in metrics:
            raise ValueError(f"Unsupported objective metric: {metric}")

        return metrics[metric](y_true, y_pred)


class FitModelStep(PipelineStep):
    used_for_training = True
    used_for_prediction = False

    def __init__(
        self,
        model_class: Type[Model],
        model_params: Optional[dict] = None,
        drop_columns: Optional[list[str]] = None,
        optuna_params: Optional[dict] = None,
        save_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.init_logger()
        self.model_class = model_class
        self.model_params = model_params or {}
        self.drop_columns = drop_columns or []
        self.optuna_params = optuna_params
        self.save_path = save_path

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.info(f"Fitting the {self.model_class.__name__} model")

        df_train, df_valid = self._prepare_data(data)
        X_train, y_train, X_valid, y_valid = self._extract_target(df_train, df_valid, data.target)

        model_params = self.model_params

        if self.optuna_params:
            optimizer = OptunaOptimizer(self.optuna_params, self.logger)
            model_params = optimizer.optimize(
                X_train, y_train, X_valid, y_valid, self.model_class, model_params
            )
            model_params.update(self.model_params)
            self.logger.info(f"Optimized model parameters: \n{json.dumps(model_params)}")
            self.logger.info("Re-fitting the model with optimized parameters")

        model = self.model_class(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=True)

        data.model = model
        data._drop_columns = self.drop_columns

        if self.save_path:
            self.logger.info(f"Saving the model to {self.save_path}")
            model.save(self.save_path)

        return data

    def _prepare_data(self, data: DataContainer) -> tuple:
        df_train = data.train
        df_valid = data.validation

        if self.drop_columns:
            df_train = df_train.drop(columns=self.drop_columns)
            df_valid = df_valid.drop(columns=self.drop_columns)

        return df_train, df_valid

    def _extract_target(
        self, df_train: pd.DataFrame, df_valid: pd.DataFrame, target: str
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Extract target column from the dataframes, to be used in model fitting."""
        X_train = df_train.drop(columns=[target])
        y_train = df_train[target]

        X_valid = df_valid.drop(columns=[target])
        y_valid = df_valid[target]

        return X_train, y_train, X_valid, y_valid
