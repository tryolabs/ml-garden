from typing import Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pipeline_lib.core import DataContainer

from .base import PipelineStep


class CalculateMetricsStep(PipelineStep):
    """Calculate metrics."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize CalculateMetricsStep."""
        super().__init__(config=config)
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.debug("Starting metric calculation")
        model_output = data[DataContainer.MODEL_OUTPUT]
        model_configs = data[DataContainer.MODEL_CONFIGS]
        target_column_name = model_configs.get("target")

        if target_column_name is None:
            raise ValueError("Target column not found in model_configs.")

        true_values = model_output[target_column_name]
        predictions = model_output[DataContainer.PREDICTIONS]

        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))

        results = {"MAE": str(mae), "RMSE": str(rmse)}
        data[DataContainer.METRICS] = results
        return data
