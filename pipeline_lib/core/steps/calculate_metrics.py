import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class CalculateMetricsStep(PipelineStep):
    """Calculate metrics."""

    used_for_prediction = False
    used_for_training = True

    def __init__(self, mape_threshold: float = 0.01) -> None:
        """Initialize CalculateMetricsStep."""
        super().__init__()
        self.init_logger()
        self.mape_threshold = mape_threshold

    def _calculate_metrics(self, true_values: pd.Series, predictions: pd.Series) -> dict:
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

        return {
            "MAE": str(mae),
            "RMSE": str(rmse),
            "R^2": str(r2),
            "Mean Error": str(me),
            "MAPE": str(mape),
            "Max Error": str(max_error),
            "Median Absolute Error": str(median_absolute_error),
        }

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.debug("Starting metric calculation")

        metrics = {}
        if data.is_train:
            # Metrics are only calculated during training
            for dataset_name in ["train", "validation", "test"]:
                dataset = getattr(data, dataset_name, None)

                if dataset is None:
                    self.logger.warning(
                        f"Dataset '{dataset_name}' not found. Skipping metric calculation."
                    )
                    continue

                metrics[dataset_name] = self._calculate_metrics(
                    true_values=dataset[data.target],
                    predictions=dataset[data.prediction_column],
                )

            # pretty print metrics
            self.logger.info(f"Metrics: {json.dumps(metrics, indent=4)}")

            data.metrics = metrics

        return data
