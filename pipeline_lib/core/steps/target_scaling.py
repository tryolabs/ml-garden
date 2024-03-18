from typing import Optional

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class TargetScalingStep(PipelineStep):
    """Scale the target."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize TargetScalingStep."""
        super().__init__(config=config)
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        if not self.config:
            self.logger.info("No target scaling configs found. Skipping target scaling.")
            return data

        self.logger.info("Scaling target data.")
        return data
