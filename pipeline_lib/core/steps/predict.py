from pathlib import Path
from typing import List, Optional

from pipeline_lib.core import DataContainer
from pipeline_lib.core.model import Model
from pipeline_lib.core.steps.base import PipelineStep


class PredictStep(PipelineStep):
    """Obtain the predictions."""

    used_for_prediction = True
    used_for_training = False

    def __init__(
        self,
        load_path: str,
        target: str,
        drop_columns: Optional[List[str]] = None,
    ) -> None:
        """Initialize Predict Step."""
        super().__init__()
        self.init_logger()
        self.load_path = load_path
        if not Path(load_path).is_file():
            self.logger.warning(f"Model file not found at {load_path}")
            self.model = None
        else:
            self.model = Model.from_file(load_path)
        self.target = target
        self.drop_columns = drop_columns or []

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Obtaining predictions")
        if not self.model:
            raise ValueError("Model not found. Please check the path.")

        drop_columns = self.drop_columns + [self.target]

        missing_columns = [col for col in drop_columns if col not in data.flow.columns]
        if missing_columns:
            error_message = (
                f"The following columns do not exist in the DataFrame: {', '.join(missing_columns)}"
            )
            self.logger.warning(error_message)
            raise KeyError(error_message)

        data.predictions = self.model.predict(data.flow.drop(columns=drop_columns))

        data.flow["predictions"] = data.predictions

        data.target = self.target

        return data
