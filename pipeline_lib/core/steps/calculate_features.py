from pipeline_lib.core import DataContainer

from .base import PipelineStep


class CalculateFeaturesStep(PipelineStep):
    """Calculate features."""

    def __init__(self) -> None:
        """Initialize CalculateFeaturesStep."""
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Calculating features")
        return data
