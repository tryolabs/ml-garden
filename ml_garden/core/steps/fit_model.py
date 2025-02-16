import json
import logging
from typing import Optional, Type

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    root_mean_squared_error,
)

from ml_garden.core import DataContainer
from ml_garden.core.constants import Task
from ml_garden.core.model import Model
from ml_garden.core.steps.base import PipelineStep

# ruff: noqa: N803 N806


class OptunaOptimizer:
    """Optuna optimizer for hyperparameter tuning."""

    def __init__(self, optuna_params: dict, logger: logging.Logger) -> None:
        """Initialize OptunaOptimizer.

        Parameters
        ----------
        optuna_params : dict
            Dictionary containing the Optuna parameters
        logger : logging.Logger
            The logger object
        """
        self.optuna_params = optuna_params
        self.logger = logger

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        model_class: Type[Model],
        model_parameters: dict,
        task: Task,
    ) -> dict:
        """Optimize the model hyperparameters using Optuna.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training dataset
        y_train : pd.Series
            The training target
        X_validation : pd.DataFrame
            The validation dataset
        y_validation : pd.Series
            The validation target
        model_class : Type[Model]
            The model class to optimize
        model_parameters : dict
            The model parameters to optimize
        task : Task
            The type of task: "regression" or "classification"

        Returns
        -------
        dict
            The best hyperparameters found by Optuna
        """

        def objective(trial: optuna.Trial) -> float:
            # Create a copy of model_parameters, then update with the optuna hyperparameters
            param = {}
            param.update(model_parameters)
            param.update(self._create_trial_params(trial))

            model = model_class(**param)
            model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False)
            preds = model.predict(X_validation)

            objective_metric = self.optuna_params["objective_metric"]
            error = self._calculate_error(y_validation, preds, objective_metric, task)
            return error

        study = self._create_study()
        study.optimize(objective, n_trials=self.optuna_params.get("trials", 20))

        best_params = study.best_params
        return best_params

    def _create_trial_params(self, trial: optuna.Trial) -> dict:
        """Create a dictionary of hyperparameters for a single Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object

        Returns
        -------
        dict
            The hyperparameters for the trial
        """
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
        """Create an Optuna study.

        Returns
        -------
        optuna.Study
            The Optuna study object
        """
        study_name = self.optuna_params.get("study_name")
        storage = self.optuna_params.get("storage", "sqlite:///db.sqlite3")
        load_if_exists = self.optuna_params.get("load_if_exists", False)

        self.logger.info(
            "Creating Optuna study with parameters: \n %s", json.dumps(self.optuna_params, indent=4)
        )

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
            direction="minimize",
        )
        return study

    @staticmethod
    def _calculate_error(y_true: np.ndarray, y_pred: np.ndarray, metric: str, task: Task) -> float:
        """Calculate the error between the true and predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            The true target values
        y_pred : np.ndarray
            The predicted target values
        metric : str
            The error metric to calculate
        task : Task
            The type of task: "regression" or "classification"

        Returns
        -------
        float
            The error value
        """
        regression_metrics = {
            "mae": mean_absolute_error,
            "mse": mean_squared_error,
            "rmse": root_mean_squared_error,
        }
        classification_metrics = {
            "accuracy": accuracy_score,
            "f1": f1_score,
            "auc": roc_auc_score,
        }

        if task == Task.REGRESSION:
            metrics = regression_metrics
        elif task == Task.CLASSIFICATION:
            metrics = classification_metrics
        else:
            error_message = f"Unsupported task: {task}"
            raise ValueError(error_message)

        if metric not in metrics:
            error_message = f"Unsupported objective metric: {metric}"
            raise ValueError(error_message)

        return metrics[metric](y_true, y_pred)


class ModelStep(PipelineStep):
    """Fit and predict with a model."""

    used_for_training = True
    used_for_prediction = True

    def __init__(
        self,
        model_class: Type[Model],
        model_parameters: Optional[dict] = None,
        optuna_params: Optional[dict] = None,
    ) -> None:
        """Initialize ModelStep.

        Parameters
        ----------
        model_class : Type[Model]
            The model class to use
        model_parameters : dict, optional
            The model parameters, by default None
        optuna_params : dict, optional
            The Optuna parameters for hyperparameter tuning, by default None
        """
        super().__init__()
        self.init_logger()
        self.model_class = model_class
        self.model_parameters = model_parameters or {}
        self.optuna_params = optuna_params

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step.

        Parameters
        ----------
        data : DataContainer
            The data container

        Returns
        -------
        DataContainer
            The updated data container
        """
        if data.is_train:
            return self.train(data)

        return self.predict(data)

    def train(self, data: DataContainer) -> DataContainer:
        """Train the model.

        Parameters
        ----------
        data : DataContainer
            The data container

        Returns
        -------
        DataContainer
            The updated data container
        """
        self.logger.info("Fitting the %s model", self.model_class.__name__)
        model_parameters = self.model_parameters

        if data.X_train is None:
            error_message = (
                "Encoded train data not found in the DataContainer, make sure the EncodeStep was"
                " executed before FitModelStep"
            )
            raise ValueError(error_message)
        if data.y_train is None:
            error_message = (
                "Encoded target data not found in the DataContainer, make sure the EncodeStep was"
                " executed before FitModelStep"
            )
            raise ValueError(error_message)
        if self.optuna_params:
            optimizer = OptunaOptimizer(self.optuna_params, self.logger)
            optuna_model_params = optimizer.optimize(
                data.X_train,
                data.y_train,
                data.X_validation,
                data.y_validation,
                self.model_class,
                model_parameters,
                data.task,
            )
            model_parameters.update(optuna_model_params)
            self.logger.info(
                "Optimized model parameters: \n%s", json.dumps(model_parameters, indent=4)
            )
            self.logger.info("Re-fitting the model with optimized parameters")

        model = self.model_class(**model_parameters)
        model.fit(
            data.X_train,
            data.y_train,
            eval_set=[(data.X_validation, data.y_validation)],
            verbose=True,
        )

        data.model = model

        # save dataset predictions for metrics calculation
        self._save_datasets_predictions(data)

        return data

    def _save_datasets_predictions(self, data: DataContainer) -> None:
        """Save the predictions for each dataset (train, val, test) in the DataContainer.

        Parameters
        ----------
        data : DataContainer
            The data container
        """
        for dataset_name in ["train", "validation", "test"]:
            dataset = getattr(data, dataset_name, None)
            encoded_dataset = getattr(data, f"X_{dataset_name}", None)

            if dataset is None or encoded_dataset is None:
                self.logger.warning(
                    "Dataset '%s' not found. Skipping metric calculation.", dataset_name
                )
                continue

            dataset[data.prediction_column] = data.model.predict(encoded_dataset)

    def predict(self, data: DataContainer) -> DataContainer:
        """Predict with the model.

        Parameters
        ----------
        data : DataContainer
            The data container

        Returns
        -------
        DataContainer
            The updated data container
        """
        self.logger.info("Predicting with %s model", self.model_class.__name__)
        data.flow[data.prediction_column] = data.model.predict(data.X_prediction)
        data.predictions = data.flow[data.prediction_column]
        return data
