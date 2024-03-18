from typing import Optional

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class EncodeStep(PipelineStep):
    """Encode the data."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize EncodeStep."""
        super().__init__(config=config)
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Encoding data")
        return data
