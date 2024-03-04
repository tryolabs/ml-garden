from pipeline_lib.core import DataContainer

from .base import PipelineStep


class EncodeStep(PipelineStep):
    """Encode the data."""

    def __init__(self) -> None:
        """Initialize EncodeStep."""
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Encoding data")
        return data
