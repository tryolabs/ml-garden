from typing import Optional

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class CalculateReportsStep(PipelineStep):
    """Calculate reports."""

    used_for_prediction = True
    used_for_training = False

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize CalculateReportsStep."""
        super().__init__(config=config)
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Calculating reports")
        return data
