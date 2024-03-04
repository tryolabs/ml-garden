from typing import Optional

from pipeline_lib.core import DataContainer

from .base import PipelineStep


class FitModelStep(PipelineStep):
    """Fit the model."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize FitModelStep."""
        super().__init__(config=config)
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Fitting the model")
        return data
