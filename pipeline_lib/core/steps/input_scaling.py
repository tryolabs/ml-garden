from pipeline_lib.core import DataContainer

from .base import PipelineStep


class InputScalingStep(PipelineStep):
    """Scale the input."""

    def __init__(self) -> None:
        """Initialize InputScalingStep."""
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Scaling input")
        return data
