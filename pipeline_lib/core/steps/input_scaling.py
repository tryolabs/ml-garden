from typing import Optional

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class InputScalingStep(PipelineStep):
    """Scale the input."""

    used_for_prediction = True
    used_for_training = True

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize InputScalingStep."""
        super().__init__(config=config)
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Scaling input")
        return data