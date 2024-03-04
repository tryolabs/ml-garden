from typing import Optional

from pipeline_lib.core import DataContainer

from .base import PipelineStep


class FitEncodersStep(PipelineStep):
    """Fit encoders."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize FitEncodersStep."""
        super().__init__(config=config)
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Fitting encoders")
        return data
