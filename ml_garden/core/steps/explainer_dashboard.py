import numpy as np
import pandas as pd
from explainerdashboard import RegressionExplainer

from pipeline_lib.core import DataContainer
from pipeline_lib.core.random_state_generator import get_random_state
from pipeline_lib.core.steps.base import PipelineStep


class ExplainerDashboardStep(PipelineStep):
    """Create an explainer dashboard for the model."""

    used_for_prediction = False
    used_for_training = True

    def __init__(
        self,
        max_samples: int = 1000,
        X_background_samples: int = 100,
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
            raise ValueError("Model not found in data container.")

        target = data.target
        if target is None:
            raise ValueError("Target column not found in any parameter.")

        if data.is_train:
            # Explainer dashboard is calculated only during training
            # We use all the available data for this purpose, optionally a sample if the data is too
            # large
            X = data.X_train
            y = data.y_train
            if data.validation is not None:
                X = pd.concat([X, data.X_validation])
                y = pd.concat([y, data.y_validation])
            if data.test is not None:
                X = pd.concat([X, data.X_test])
                y = pd.concat([y, data.y_test])

            # Some Shap explainers require a "background dataset" with the original distribution
            # of the data.
            if self.X_background_samples > 0 and len(X) > self.X_background_samples:
                X_backround = X.sample(
                    n=self.max_samples, random_state=get_random_state(), replace=False
                )
            else:
                X_backround = X

            if self.max_samples > 0 and len(X) > self.max_samples:
                # Randomly sample a subset of data points if the dataset is larger than max_samples
                self.logger.info(
                    f"Dataset contains {len(X)} data points and max_samples is set to"
                    f" {self.max_samples}."
                )
                self.logger.info(f"Sampling {self.max_samples} data points from the dataset.")
                sample_rows = np.random.choice(range(len(X)), replace=False, size=self.max_samples)
                X = X.iloc[sample_rows, :]
                y = y.iloc[sample_rows]

            # This can happen if there are duplicate indices, the Shap values will run, taking a
            # long time, but it will crash when calculating the shap dependence plots.
            # To avoid this potential long wait to a crash we add this assertion here
            assert len(X) == len(y), "Mismatch in number of samples and labels"

            explainer = RegressionExplainer(
                model,
                X_background=X_backround,
                X=X,
                y=y,
            )
            explainer.calculate_properties()
            data.explainer = explainer

        return data
