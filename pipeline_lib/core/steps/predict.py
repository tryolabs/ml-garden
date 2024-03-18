from typing import Optional

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class PredictStep(PipelineStep):
    """Obtain the predictions."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize Predict Step."""
        super().__init__(config=config)
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Obtaining predictions")
        return data
