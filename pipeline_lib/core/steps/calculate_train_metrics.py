import json
import time
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline_lib.core import DataContainer
from pipeline_lib.core.model import Model
from pipeline_lib.core.steps.base import PipelineStep


class CalculateTrainMetricsStep(PipelineStep):
    """Calculate metrics."""

    used_for_prediction = False
    used_for_training = True

    def __init__(self) -> None:
        """Initialize CalculateMetricsStep."""
        super().__init__()
        self.init_logger()

    def _calculate_metrics(self, true_values: pd.Series, predictions: pd.Series) -> dict:
        return {
            "MAE": str(mean_absolute_error(true_values, predictions)),
            "RMSE": str(np.sqrt(mean_squared_error(true_values, predictions))),
            "R^2": str(r2_score(true_values, predictions)),
            "Mean Error": str(np.mean(true_values - predictions)),
            "Max Error": str(np.max(np.abs(true_values - predictions))),
            "Median Absolute Error": str(np.median(np.abs(true_values - predictions))),
        }

    def _get_predictions(
        self, model: Model, df: pd.DataFrame, target: str, drop_columns: Optional[List[str]] = None
    ) -> pd.Series:
        drop_columns = (drop_columns or []) + [target]
        return model.predict(df.drop(columns=drop_columns))

    def _log_metrics(self, dataset_name: str, metrics: dict) -> None:
        self.logger.info(f"Metrics for {dataset_name} dataset:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value}")

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.debug("Starting metric calculation")

        target_column_name = data.target
        if target_column_name is None:
            raise ValueError("Target column not found on any configuration.")

        metrics = {}

        for dataset_name in ["train", "validation", "test"]:
            start_time = time.time()
            dataset = getattr(data, dataset_name, None)

            if dataset is None:
                self.logger.warning(
                    f"Dataset '{dataset_name}' not found. Skipping metric calculation."
                )
                continue

            predictions = self._get_predictions(
                model=data.model,
                df=dataset,
                target=target_column_name,
                drop_columns=data._drop_columns,
            )
            metrics[dataset_name] = self._calculate_metrics(
                true_values=dataset[target_column_name],
                predictions=predictions,
            )
            elapsed_time = time.time() - start_time
            self.logger.info(f"Elapsed time for {dataset_name} dataset: {elapsed_time:.2f} seconds")

        # pretty print metrics
        self.logger.info(f"Metrics: {json.dumps(metrics, indent=4)}")

        data.metrics = metrics

        return data
