from pipeline_lib.core import DataContainer

from .base import PipelineStep


class TargetScalingStep(PipelineStep):
    """Scale the target."""

    def __init__(self) -> None:
        """Initialize TargetScalingStep."""
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        target_scaling_configs = data.get(DataContainer.TARGET_SCALING_CONFIGS)

        if target_scaling_configs is None:
            self.logger.info("No target scaling configs found. Skipping target scaling.")
            return data

        self.logger.info("Scaling target data.")
        return data
