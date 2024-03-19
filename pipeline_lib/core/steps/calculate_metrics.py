import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class CalculateMetricsStep(PipelineStep):
    """Calculate metrics."""

    def __init__(self) -> None:
        """Initialize CalculateMetricsStep."""
        super().__init__()
        self.init_logger()

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.debug("Starting metric calculation")
        model_output = data.model_output

        target_column_name = data.target

        if target_column_name is None:
            raise ValueError("Target column not found on any configuration.")

        true_values = model_output[target_column_name]
        predictions = model_output["predictions"]

        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        r2 = r2_score(true_values, predictions)

        # Additional metrics
        me = np.mean(true_values - predictions)  # Mean Error
        mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
        max_error = np.max(np.abs(true_values - predictions))
        median_absolute_error = np.median(np.abs(true_values - predictions))

        results = {
            "MAE": str(mae),
            "RMSE": str(rmse),
            "R^2": str(r2),
            "Mean Error": str(me),
            "MAPE": str(mape),
            "Max Error": str(max_error),
            "Median Absolute Error": str(median_absolute_error),
        }
        self.logger.info(results)
        data.metrics = results
        return data
