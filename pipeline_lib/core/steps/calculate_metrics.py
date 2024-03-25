import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class CalculateMetricsStep(PipelineStep):
    """Calculate metrics."""

    used_for_prediction = True
    used_for_training = False

    def __init__(self, mape_threshold: float = 0.01) -> None:
        """Initialize CalculateMetricsStep."""
        super().__init__()
        self.init_logger()
        self.mape_threshold = mape_threshold

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.debug("Starting metric calculation")

        target_column_name = data.target

        if target_column_name is None:
            raise ValueError("Target column nsot found on any configuration.")

        true_values = data.flow[target_column_name]
        predictions = data.predictions

        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        r2 = r2_score(true_values, predictions)

        # Additional metrics
        me = np.mean(true_values - predictions)  # Mean Error
        max_error = np.max(np.abs(true_values - predictions))
        median_absolute_error = np.median(np.abs(true_values - predictions))

        # MAPE calculation with threshold
        mask = (true_values > self.mape_threshold) & (predictions > self.mape_threshold)
        mape_true_values = true_values[mask]
        mape_predictions = predictions[mask]
        if len(mape_true_values) > 0:
            mape = np.mean(np.abs((mape_true_values - mape_predictions) / mape_true_values)) * 100
        else:
            mape = np.nan

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
