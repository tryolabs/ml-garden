from pipeline_lib.core import DataContainer

from .base import PipelineStep


class CalculateReportsStep(PipelineStep):
    """Calculate reports."""

    def __init__(self) -> None:
        """Initialize CalculateReportsStep."""
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Calculating reports")
        return data
