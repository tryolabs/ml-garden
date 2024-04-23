import pandas as pd
from explainerdashboard import RegressionExplainer

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class ExplainerDashboardStep(PipelineStep):
    """Scale the target using Quantile Transformer."""

    used_for_prediction = False
    used_for_training = True

    def __init__(
        self,
        max_samples: int = 100,
        X_background_samples: int = 100,
        enable_step: bool = True,
    ) -> None:
        self.init_logger()
        self.max_samples = max_samples
        self.X_background_samples = X_background_samples
        self.enable_step = enable_step

    def execute(self, data: DataContainer) -> DataContainer:
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
            df = data.train
            if data.validation is not None:
                df = pd.concat([df, data.validation])
            if data.test is not None:
                df = pd.concat([df, data.test])

            if target not in df.columns:
                raise ValueError(
                    f"Target column `{target}` not found in the dataset. It must be present for the"
                    " explainer dashboard."
                )

            # Some Shap explainers require a "background dataset" with the original distribution
            # of the data.
            if self.X_background_samples > 0 and len(df) > self.X_background_samples:
                X_backround = df.sample(n=self.max_samples, random_state=42)
            else:
                X_backround = df

            if self.max_samples > 0 and len(df) > self.max_samples:
                # Randomly sample a subset of data points if the dataset is larger than max_samples
                self.logger.info(
                    f"Dataset contains {len(df)} data points and max_samples is set to"
                    f" {self.max_samples}."
                )
                self.logger.info(f"Sampling {self.max_samples} data points from the dataset.")
                df = df.sample(n=self.max_samples, random_state=42)

            drop_columns = (
                data._drop_columns + ["predictions"] if data._drop_columns else ["predictions"]
            )
            drop_columns = [col for col in drop_columns if col in df.columns]

            df = df.drop(columns=drop_columns)
            X_backround = X_backround.drop(columns=drop_columns + [target])

            X = df.drop(columns=[target])
            y = df[target]

            explainer = RegressionExplainer(
                model,
                X_background=X_backround,
                X=X,
                y=y,
            )
            explainer.calculate_properties()
            data.explainer = explainer

        return data
