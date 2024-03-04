from pipeline_lib.core import DataContainer

from .base import PipelineStep


class ExplainerDashboardStep(PipelineStep):
    """Explainer Dashboard."""

    def __init__(self) -> None:
        """Initialize ExplainerDashboardStep."""
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Creating explainer dashboard.")
        return data
