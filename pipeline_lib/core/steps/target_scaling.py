from typing import Optional

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class TargetScalingStep(PipelineStep):
    """Scale the target."""

    used_for_prediction = True
    used_for_training = True

    def __init__(self) -> None:
        """Initialize TargetScalingStep."""
        super().__init__()
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        if not self.config:
            self.logger.info("No target scaling configs found. Skipping target scaling.")
            return data

        self.logger.info("Scaling target data.")
        return data
