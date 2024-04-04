from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class PredictStep(PipelineStep):
    """Obtain the predictions."""

    used_for_prediction = True
    used_for_training = False

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

        drop_columns = data._drop_columns + [data.target]

        missing_columns = [col for col in drop_columns if col not in data.flow.columns]
        if missing_columns:
            error_message = (
                f"The following columns do not exist in the DataFrame: {', '.join(missing_columns)}"
            )
            self.logger.warning(error_message)
            raise KeyError(error_message)

        data.predictions = data.model.predict(data.flow.drop(columns=drop_columns))

        data.flow["predictions"] = data.predictions

        return data
