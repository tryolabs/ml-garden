from typing import Any, Optional, Type

from ml_garden.core import DataContainer
from ml_garden.core.constants import Task
from ml_garden.core.model import Model
from ml_garden.core.steps.fit_model import ModelStep
from ml_garden.implementation.tabular.autogluon.model import AutoGluon


class AutoGluonModelStep(ModelStep):
    """Fit and predict with a model."""

    used_for_training = True
    used_for_prediction = True

    def __init__(
        self,
        model_class: Type[Model],
        autogluon_create_params: Optional[dict[str, Any]] = None,
        autogluon_fit_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize ModelStep.
        Parameters
        ----------
        model_class : Type[Model]
            The model class to use
        autogluon_create_params : dict
            The AutoGluon create params
        autogluon_fit_params : dict, optional
            The AutoGluon fit params, by default None
        """
        super().__init__(
            model_class=AutoGluon, model_parameters=None, optuna_params=None
        )
        self.autogluon_create_params = autogluon_create_params or {}
        self.autogluon_fit_params = autogluon_fit_params or {}

    def _get_autogluon_problem_type_from_task(
        self, task: Task, data: DataContainer
    ) -> str:
        """Get the AutoGluon problem type from the task.
        Parameters
        ----------
        task : Task
            The task
        data : DataContainer
            The data container
        Returns
        -------
        str
            The AutoGluon problem type
        """

        if task == Task.CLASSIFICATION:
            if len(data.y_train.unique()) == 2:
                return "binary"
            return "multiclass"
        elif task == Task.QUANTILE_REGRESSION:
            return "quantile"
        elif task == Task.REGRESSION:
            return "regression"
        else:
            raise ValueError(
                f"Task {task} not supported by AutoGluon. Supported tasks: {AutoGluon.TASKS}"
            )

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
        self.logger.info(f"Fitting the {self.model_class.__name__} model")

        assert data.X_train is not None and data.y_train is not None, (
            "Encoded train data not found in the DataContainer, make sure the EncodeStep was"
            " executed before FitModelStep"
        )

        self.autogluon_create_params["label"] = data.target
        self.autogluon_create_params["problem_type"] = data.task.value

        model = AutoGluon(
            self.autogluon_create_params,
            self.autogluon_fit_params,
        )
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
