from pipeline_lib.core import DataContainer

from .base import PipelineStep


class AugmentStep(PipelineStep):
    """Augment the data."""

    def __init__(self) -> None:
        """Initialize AugmentStep."""
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Augmenting data")
        return data
