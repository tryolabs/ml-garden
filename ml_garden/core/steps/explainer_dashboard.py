import numpy as np
import pandas as pd
from explainerdashboard import ClassifierExplainer, RegressionExplainer

from ml_garden.core import DataContainer
from ml_garden.core.constants import Task
from ml_garden.core.random_state_generator import RandomStateManager
from ml_garden.core.steps.base import PipelineStep

# ruff: noqa: N803 N806


class ExplainerDashboardStep(PipelineStep):
    """Create an explainer dashboard for the model."""

    used_for_prediction = False
    used_for_training = True

    def __init__(
        self,
        max_samples: int = 1000,
        X_background_samples: int = 100,
        *,
        enable_step: bool = True,
    ) -> None:
        """Initialize ExplainerDashboardStep.

        Parameters
        ----------
        max_samples : int, optional
            Maximum number of samples to use for the explainer dashboard, by default 1000
        X_background_samples : int, optional
            Number of samples to use for the background dataset, by default 100
        enable_step : bool, optional
            Enable or disable the step, by default True
        """
        self.init_logger()
        self.max_samples = max_samples
        self.X_background_samples = X_background_samples
        self.enable_step = enable_step

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
        if not self.enable_step:
            self.logger.info("ExplainerDashboardStep disabled, skipping execution")
            return data

        self.logger.debug("Starting explainer dashboard")

        model = data.model
        if model is None:
            error_message = "Model not found in data container."
            raise ValueError(error_message)

        target = data.target
        if target is None:
            error_message = "Target column not found in any parameter."
            raise ValueError(error_message)

        if data.is_train:
            # Explainer dashboard is calculated only during training
            # We use all the available data for this purpose, optionally a sample if the data is too
            # large
            X = data.X_validation
            y = data.y_validation

            if data.X_test is not None:
                X = pd.concat([X, data.X_test])
                y = pd.concat([y, data.y_test])

            # Some Shap explainers require a "background dataset" with the original distribution
            # of the data.
            if self.X_background_samples > 0 and len(X) > self.X_background_samples:
                X_background = X.sample(
                    n=self.max_samples, random_state=RandomStateManager.get_state(), replace=False
                )
            else:
                X_background = X

            if self.max_samples > 0 and len(X) > self.max_samples:
                # Randomly sample a subset of data points if the dataset is larger than max_samples
                self.logger.info(
                    "Dataset contains %s data points and max_samples is set to %s.",
                    len(X),
                    self.max_samples,
                )
                self.logger.info("Sampling %s data points from the dataset.", self.max_samples)
                rng = np.random.default_rng()  # Create a random number generator
                sample_rows = rng.choice(
                    len(X), replace=False, size=self.max_samples
                )  # Use the generator
                X = X.iloc[sample_rows, :]
                y = y.iloc[sample_rows]

            # To avoid this potential long wait to a crash we add this check here
            if len(X) != len(y):
                error_message = "Mismatch in number of samples and labels"
                raise ValueError(error_message)

            # Choose the appropriate explainer based on the task
            explainer_class = {
                Task.REGRESSION: RegressionExplainer,
                Task.CLASSIFICATION: ClassifierExplainer,
            }.get(data.task)

            if explainer_class is None:
                error_message = f"Unsupported task type: {data.task}"
                raise ValueError(error_message)

            explainer = explainer_class(
                model,
                X_background=X_background,
                X=X,
                y=y,
            )

            explainer.calculate_properties()
            data.explainer = explainer

        return data
