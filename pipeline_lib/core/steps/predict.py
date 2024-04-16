from typing import Optional

import mlflow.pyfunc

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class PredictStep(PipelineStep):
    """Obtain the predictions."""

    used_for_prediction = True
    used_for_training = True

    def __init__(
        self,
        mlflow_model_name: Optional[str] = None,
        mlflow_model_version: Optional[int] = None,
    ) -> None:
        """Initialize Predict Step."""
        super().__init__()
        self.init_logger()
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_model_version = mlflow_model_version

        # ensure if model name is provided version exist and vice versa
        if self.mlflow_model_name and not self.mlflow_model_version:
            raise ValueError("Model version must be provided if model name is provided.")

        if not self.mlflow_model_name and self.mlflow_model_version:
            raise ValueError("Model name must be provided if model version is provided.")

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Obtaining predictions")

        if not data.model and not self.mlflow_model_name:
            raise ValueError("Model is not present on the data container.")

        if data.is_train:
            for dataset_name in ["train", "validation", "test"]:
                dataset = getattr(data, dataset_name, None)
                encoded_dataset = getattr(data, f"X_{dataset_name}", None)

                if dataset is None or encoded_dataset is None:
                    self.logger.warning(
                        f"Dataset '{dataset_name}' not found. Skipping metric calculation."
                    )
                    continue

                dataset[data.prediction_column] = data.model.predict(encoded_dataset)
        else:
            data.flow[data.prediction_column] = data.model.predict(data.X_prediction)

        return data
