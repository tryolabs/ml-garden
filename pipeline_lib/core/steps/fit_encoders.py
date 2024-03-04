from pipeline_lib.core import DataContainer

from .base import PipelineStep


class FitEncodersStep(PipelineStep):
    """Fit encoders."""

    def __init__(self) -> None:
        """Initialize FitEncodersStep."""
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Fitting encoders")
        return data
