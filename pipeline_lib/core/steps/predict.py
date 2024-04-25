import time

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class PredictStep(PipelineStep):
    """Obtain the predictions."""

    used_for_prediction = True
    used_for_training = True

    def __init__(
        self,
    ) -> None:
        """Initialize Predict Step."""
        super().__init__()
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Obtaining predictions")

        if not data.model:
            raise ValueError("Model is not present on the data container.")

        if data.is_train:
            # Metrics are only calculated during training
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
