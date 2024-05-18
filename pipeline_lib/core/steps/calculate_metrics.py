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

    def __init__(self) -> None:
        """Initialize CalculateMetricsStep."""
        super().__init__()
        self.init_logger()

    def _calculate_metrics(self, true_values: pd.Series, predictions: pd.Series) -> dict:
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        r2 = r2_score(true_values, predictions)

        # Additional metrics
        me = np.mean(true_values - predictions)  # Mean Error
        max_error = np.max(np.abs(true_values - predictions))
        median_absolute_error = np.median(np.abs(true_values - predictions))

        return {
            "MAE": str(mae),
            "RMSE": str(rmse),
            "R_2": str(r2),
            "Mean Error": str(me),
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
