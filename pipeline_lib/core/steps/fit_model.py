import json
import logging
from typing import Optional, Type

import optuna
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
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_class: Type[Model],
        model_parameters: dict,
    ) -> dict:
        def objective(trial):
            # Create a copy of model_parameters, then update with the optuna hyperparameters
            param = {}
            param.update(model_parameters)
            param.update(self._create_trial_params(trial))

            model = model_class(**param)
            model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False)
            preds = model.predict(X_validation)

            objective_metric = self.optuna_params.get("objective_metric", "mean_absolute_error")
            error = self._calculate_error(y_validation, preds, objective_metric)
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


class ModelStep(PipelineStep):
    used_for_training = True
    used_for_prediction = True

    def __init__(
        self,
        model_class: Type[Model],
        model_parameters: Optional[dict] = None,
        optuna_params: Optional[dict] = None,
        save_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.init_logger()
        self.model_class = model_class
        self.model_parameters = model_parameters or {}
        self.optuna_params = optuna_params
        self.save_path = save_path

    def execute(self, data: DataContainer) -> DataContainer:
        if data.is_train:
            return self.train(data)

        return self.predict(data)

    def train(self, data: DataContainer) -> DataContainer:
        self.logger.info(f"Fitting the {self.model_class.__name__} model")
        model_parameters = self.model_parameters

        assert data.X_train is not None and data.y_train is not None, (
            "Encoded train data not found in the DataContainer, make sure the EncodeStep was"
            " executed before FitModelStep"
        )

        if self.optuna_params:
            optimizer = OptunaOptimizer(self.optuna_params, self.logger)
            optuna_model_params = optimizer.optimize(
                data.X_train,
                data.y_train,
                data.X_validation,
                data.y_validation,
                self.model_class,
                model_parameters,
            )
            model_parameters.update(optuna_model_params)
            self.logger.info(f"Optimized model parameters: \n{json.dumps(model_parameters)}")
            self.logger.info("Re-fitting the model with optimized parameters")

        model = self.model_class(**model_parameters)
        model.fit(
            data.X_train,
            data.y_train,
            eval_set=[(data.X_validation, data.y_validation)],
            verbose=True,
        )

        data.model = model

        if self.save_path:
            self.logger.info(f"Saving the model to {self.save_path}")
            model.save(self.save_path)

        # save dataset predictions for metrics calculation
        self._save_datasets_predictions(data)

        return data

    def _save_datasets_predictions(self, data: DataContainer) -> None:
        """Save the predictions for each dataset (train, val, test) in the DataContainer."""
        for dataset_name in ["train", "validation", "test"]:
            dataset = getattr(data, dataset_name, None)
            encoded_dataset = getattr(data, f"X_{dataset_name}", None)

            if dataset is None or encoded_dataset is None:
                self.logger.warning(
                    f"Dataset '{dataset_name}' not found. Skipping metric calculation."
                )
                continue

            dataset[data.prediction_column] = data.model.predict(encoded_dataset)

    def predict(self, data: DataContainer) -> DataContainer:
        self.logger.info(f"Predicting with {self.model_class.__name__} model")
        data.flow[data.prediction_column] = data.model.predict(data.X_prediction)
        return data
