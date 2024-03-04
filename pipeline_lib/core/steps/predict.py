from pipeline_lib.core import DataContainer

from .base import PipelineStep


class PredictStep(PipelineStep):
    """Obtain the predictions."""

    def __init__(self) -> None:
        """Initialize Predict Step."""
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Obtaining predictions")
        return data
