from explainerdashboard import RegressionExplainer

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class ExplainerDashboardStep(PipelineStep):
    """Scale the target using Quantile Transformer."""

    used_for_prediction = True
    used_for_training = False

    def __init__(
        self,
        max_samples: int = 1000,
    ) -> None:
        self.init_logger()
        self.max_samples = max_samples

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.debug("Starting explainer dashboard")

        model = data.model
        if model is None:
            raise ValueError("Model not found in data container.")

        target = data.target
        if target is None:
            raise ValueError("Target column not found in any parameter.")

        if target not in data.flow.columns:
            raise ValueError(
                f"Target column `{target}` not found in the dataset. It must be present for the"
                " explainer dashboard."
            )

        df = data.flow

        if len(df) > self.max_samples:
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

        df = df.drop(columns=drop_columns)

        X_test = df.drop(columns=[target])
        y_test = df[target]

        explainer = RegressionExplainer(
            model,
            X_test,
            y_test,
        )

        data.explainer = explainer

        return data
